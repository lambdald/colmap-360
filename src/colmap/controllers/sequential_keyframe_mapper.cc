// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/controllers/sequential_keyframe_mapper.h"

#include "colmap/util/misc.h"

namespace colmap {
namespace {

size_t TriangulateImage(const SequentialKeyframeMapperOptions& options,
                        const Image& image,
                        SequentialKeyframeMapper* mapper) {
  LOG(INFO) << "=> Continued observations: " << image.NumPoints3D();
  const size_t num_tris =
      mapper->TriangulateImage(options.Triangulation(), image.ImageId());
  LOG(INFO) << "=> Added observations: " << num_tris;
  return num_tris;
}

void AdjustGlobalBundle(const SequentialKeyframeMapperOptions& options,
                        SequentialKeyframeMapper* mapper) {
  BundleAdjustmentOptions custom_ba_options = options.GlobalBundleAdjustment();

  const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();

  // Use stricter convergence criteria for first registered images.
  const size_t kMinNumRegImagesForFastBA = 10;
  if (num_reg_images < kMinNumRegImagesForFastBA) {
    custom_ba_options.solver_options.function_tolerance /= 10;
    custom_ba_options.solver_options.gradient_tolerance /= 10;
    custom_ba_options.solver_options.parameter_tolerance /= 10;
    custom_ba_options.solver_options.max_num_iterations *= 2;
    custom_ba_options.solver_options.max_linear_solver_iterations = 200;
  }

  PrintHeading1("Global bundle adjustment");
  mapper->AdjustGlobalBundle(options.Mapper(), custom_ba_options);
}

void IterativeLocalRefinement(const SequentialKeyframeMapperOptions& options,
                              const image_t image_id,
                              SequentialKeyframeMapper* mapper) {
  auto ba_options = options.LocalBundleAdjustment();
  for (int i = 0; i < options.ba_local_max_refinements; ++i) {
    const auto report =
        mapper->AdjustLocalBundle(options.Mapper(),
                                  ba_options,
                                  options.Triangulation(),
                                  image_id,
                                  mapper->GetModifiedPoints3D());
    LOG(INFO) << "=> Merged observations: " << report.num_merged_observations;
    LOG(INFO) << "=> Completed observations: "
              << report.num_completed_observations;
    LOG(INFO) << "=> Filtered observations: "
              << report.num_filtered_observations;
    const double changed =
        report.num_adjusted_observations == 0
            ? 0
            : (report.num_merged_observations +
               report.num_completed_observations +
               report.num_filtered_observations) /
                  static_cast<double>(report.num_adjusted_observations);
    LOG(INFO) << StringPrintf("=> Changed observations: %.6f", changed);
    if (changed < options.ba_local_max_refinement_change) {
      break;
    }
    // Only use robust cost function for first iteration.
    ba_options.loss_function_type =
        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  }
  mapper->ClearModifiedPoints3D();
}

void IterativeGlobalRefinement(const SequentialKeyframeMapperOptions& options,
                               SequentialKeyframeMapper* mapper) {
  PrintHeading1("Retriangulation");
  CompleteAndMergeTracks(options, mapper);
  LOG(INFO) << "=> Retriangulated observations: "
            << mapper->Retriangulate(options.Triangulation());

  for (int i = 0; i < options.ba_global_max_refinements; ++i) {
    const size_t num_observations =
        mapper->GetReconstruction().ComputeNumObservations();
    size_t num_changed_observations = 0;
    AdjustGlobalBundle(options, mapper);
    num_changed_observations += CompleteAndMergeTracks(options, mapper);
    num_changed_observations += FilterPoints(options, mapper);
    const double changed =
        num_observations == 0
            ? 0
            : static_cast<double>(num_changed_observations) / num_observations;
    LOG(INFO) << StringPrintf("=> Changed observations: %.6f", changed);
    if (changed < options.ba_global_max_refinement_change) {
      break;
    }
  }

  FilterImages(options, mapper);
}

void ExtractColors(const std::string& image_path,
                   const image_t image_id,
                   Reconstruction* reconstruction) {
  if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
    LOG(WARNING) << StringPrintf("Could not read image %s at path %s.",
                                 reconstruction->Image(image_id).Name().c_str(),
                                 image_path.c_str());
  }
}

void WriteSnapshot(const Reconstruction& reconstruction,
                   const std::string& snapshot_path) {
  PrintHeading1("Creating snapshot");
  // Get the current timestamp in milliseconds.
  const size_t timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  // Write reconstruction to unique path with current timestamp.
  const std::string path =
      JoinPaths(snapshot_path, StringPrintf("%010d", timestamp));
  CreateDirIfNotExists(path);
  LOG(INFO) << "=> Writing to " << path;
  reconstruction.Write(path);
}

}  // namespace

size_t FilterPoints(const SequentialKeyframeMapperOptions& options,
                    SequentialKeyframeMapper* mapper) {
  const size_t num_filtered_observations =
      mapper->FilterPoints(options.Mapper());
  LOG(INFO) << "=> Filtered observations: " << num_filtered_observations;
  return num_filtered_observations;
}

size_t FilterImages(const SequentialKeyframeMapperOptions& options,
                    SequentialKeyframeMapper* mapper) {
  const size_t num_filtered_images = mapper->FilterImages(options.Mapper());
  LOG(INFO) << "=> Filtered images: " << num_filtered_images;
  return num_filtered_images;
}

size_t CompleteAndMergeTracks(const SequentialKeyframeMapperOptions& options,
                              SequentialKeyframeMapper* mapper) {
  const size_t num_completed_observations =
      mapper->CompleteTracks(options.Triangulation());
  LOG(INFO) << "=> Completed observations: " << num_completed_observations;
  const size_t num_merged_observations =
      mapper->MergeTracks(options.Triangulation());
  LOG(INFO) << "=> Merged observations: " << num_merged_observations;
  return num_completed_observations + num_merged_observations;
}

SequentialKeyframeMapper::Options SequentialKeyframeMapperOptions::Mapper() const {
  SequentialKeyframeMapper::Options options = mapper;
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  options.local_ba_num_images = ba_local_num_images;
  options.fix_existing_images = fix_existing_images;
  return options;
}

IncrementalTriangulator::Options SequentialKeyframeMapperOptions::Triangulation()
    const {
  IncrementalTriangulator::Options options = triangulation;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  return options;
}

BundleAdjustmentOptions SequentialKeyframeMapperOptions::LocalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = ba_local_function_tolerance;
  options.solver_options.gradient_tolerance = 10.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_local_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.logging_type = ceres::LoggingType::SILENT;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.min_num_residuals_for_multi_threading =
      ba_min_num_residuals_for_multi_threading;
  options.loss_function_scale = 1.0;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  return options;
}

BundleAdjustmentOptions SequentialKeyframeMapperOptions::GlobalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = ba_global_function_tolerance;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_global_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.logging_type =
      ceres::LoggingType::PER_MINIMIZER_ITERATION;
  options.solver_options.minimizer_progress_to_stdout = true;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.min_num_residuals_for_multi_threading =
      ba_min_num_residuals_for_multi_threading;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  return options;
}

bool SequentialKeyframeMapperOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(max_num_models, 0);
  CHECK_OPTION_GT(max_model_overlap, 0);
  CHECK_OPTION_GE(min_model_size, 0);
  CHECK_OPTION_GT(init_num_trials, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GE(ba_local_num_images, 2);
  CHECK_OPTION_GE(ba_local_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_images_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_images_freq, 0);
  CHECK_OPTION_GT(ba_global_points_freq, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_local_max_refinements, 0);
  CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
  CHECK_OPTION_GT(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION_GE(snapshot_images_freq, 0);
  CHECK_OPTION(Mapper().Check());
  CHECK_OPTION(Triangulation().Check());
  return true;
}

SequentialKeyframeMapperController::SequentialKeyframeMapperController(
    std::shared_ptr<const SequentialKeyframeMapperOptions> options,
    const std::string& image_path,
    const std::string& database_path,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(std::move(options)),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_manager_(std::move(reconstruction_manager)) {
  CHECK(options_->Check());
  RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
  RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
  RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

void SequentialKeyframeMapperController::Run() {
  if (!LoadDatabase()) {
    return;
  }

  SequentialKeyframeMapper::Options init_mapper_options = options_->Mapper();
  Reconstruct(init_mapper_options);

  const size_t kNumInitRelaxations = 5;
  const float kMinNumRegImageRatio = 0.2;
  const int kMinNumRegImage = database_cache_->NumImages() * kMinNumRegImageRatio;
  for (size_t i = 0; i < kNumInitRelaxations; ++i) {
    // if (IsStopped()) {
    //   break;
    // }

    // if(reconstruction_manager_->Size() > 0 && reconstruction_manager_->Get(0)->NumRegImages() > kMinNumRegImage) {
    //     break;
    // }

    // if (reconstruction_manager_->Size() > 0) {
    //   reconstruction_manager_->Clear();
    // }


    // LOG(INFO) << "=> Relaxing the initialization constraints [init_min_num_inliers].";
    // init_mapper_options.init_min_num_inliers /= 2;
    // Reconstruct(init_mapper_options);

    if (IsStopped()) {
      break;
    }

    if(reconstruction_manager_->Size() > 0 && reconstruction_manager_->Get(0)->NumRegImages() > kMinNumRegImage) {
        break;
    }
    if (reconstruction_manager_->Size() > 0) {
      reconstruction_manager_->Clear();
    }
    LOG(INFO) << "=> Relaxing the initialization constraints [init_min_tri_angle].";
    init_mapper_options.init_min_tri_angle /= 2;
    Reconstruct(init_mapper_options);


    if (IsStopped()) {
      break;
    }

    if(reconstruction_manager_->Size() > 0 && reconstruction_manager_->Get(0)->NumRegImages() > kMinNumRegImage) {
        break;
    }
    if (reconstruction_manager_->Size() > 0) {
      reconstruction_manager_->Clear();
    }

    LOG(INFO) << "=> Relaxing the initialization constraints [num_adjacent].";
    init_mapper_options.num_adjacent /= 2;
    Reconstruct(init_mapper_options);

    if (IsStopped()) {
      break;
    }

    if(reconstruction_manager_->Size() > 0 && reconstruction_manager_->Get(0)->NumRegImages() > kMinNumRegImage) {
        break;
    }

    if (reconstruction_manager_->Size() > 0) {
      reconstruction_manager_->Clear();
    }

    LOG(INFO) << "=> Relaxing the initialization constraints [init_min_num_inliers].";
    init_mapper_options.init_min_num_inliers /= 2;
    Reconstruct(init_mapper_options);
  }

  GetTimer().PrintMinutes();
}

bool SequentialKeyframeMapperController::LoadDatabase() {
  PrintHeading1("Loading database");

  // Make sure images of the given reconstruction are also included when
  // manually specifying images for the reconstrunstruction procedure.
  std::unordered_set<std::string> image_names = options_->image_names;
  if (reconstruction_manager_->Size() == 1 && !options_->image_names.empty()) {
    const auto& reconstruction = reconstruction_manager_->Get(0);
    for (const image_t image_id : reconstruction->RegImageIds()) {
      const auto& image = reconstruction->Image(image_id);
      image_names.insert(image.Name());
    }
  }

  Database database(database_path_);
  Timer timer;
  timer.Start();
  const size_t min_num_matches = static_cast<size_t>(options_->min_num_matches);
  database_cache_ = DatabaseCache::Create(
      database, min_num_matches, options_->ignore_watermarks, image_names);
  timer.PrintMinutes();

  if (database_cache_->NumImages() == 0) {
    LOG(WARNING) << "No images with matches found in the database";
    return false;
  }

  return true;
}

void SequentialKeyframeMapperController::Reconstruct(
    const SequentialKeyframeMapper::Options& init_mapper_options) {
  //////////////////////////////////////////////////////////////////////////////
  // Main loop
  //////////////////////////////////////////////////////////////////////////////

  SequentialKeyframeMapper mapper(database_cache_);

  // Is there a sub-model before we start the reconstruction? I.e. the user
  // has imported an existing reconstruction.
  const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
  CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                  "single reconstruction, but "
                                                  "multiple are given.";

  for (int num_trials = 0; num_trials < options_->init_num_trials;
       ++num_trials) {

    LOG(INFO) << "==> Begin trial " << num_trials;
    BlockIfPaused();
    if (IsStopped()) {
      break;
    }

    size_t reconstruction_idx;
    if (!initial_reconstruction_given || num_trials > 0) {
      reconstruction_idx = reconstruction_manager_->Add();
    } else {
      reconstruction_idx = 0;
    }

    std::shared_ptr<Reconstruction> reconstruction =
        reconstruction_manager_->Get(reconstruction_idx);

    mapper.BeginReconstruction(reconstruction);

    ////////////////////////////////////////////////////////////////////////////
    // Register initial pair
    ////////////////////////////////////////////////////////////////////////////

    if (reconstruction->NumRegImages() == 0) {
      image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
      image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

      // Try to find good initial pair.
      if (options_->init_image_id1 == -1 || options_->init_image_id2 == -1) {
        PrintHeading1("Finding good initial image pair");
        const bool find_init_success = mapper.FindInitialImagePair(
            init_mapper_options, &image_id1, &image_id2);
        if (!find_init_success) {
          LOG(INFO) << "=> No good initial image pair found.";
          mapper.EndReconstruction(/*discard=*/true);
          reconstruction_manager_->Delete(reconstruction_idx);
          break;
        }
      } else {
        if (!reconstruction->ExistsImage(image_id1) ||
            !reconstruction->ExistsImage(image_id2)) {
          LOG(INFO) << StringPrintf(
              "=> Initial image pair #%d and #%d do not exist.",
              image_id1,
              image_id2);
          mapper.EndReconstruction(/*discard=*/true);
          reconstruction_manager_->Delete(reconstruction_idx);
          return;
        }
      }

      PrintHeading1(StringPrintf(
          "Initializing with image pair #%d and #%d", image_id1, image_id2));
      const bool reg_init_success = mapper.RegisterInitialImagePair(
          init_mapper_options, image_id1, image_id2);
      if (!reg_init_success) {
        LOG(INFO) << "=> Initialization failed - possible solutions:"
                  << std::endl
                  << "     - try to relax the initialization constraints"
                  << std::endl
                  << "     - manually select an initial image pair";
        mapper.EndReconstruction(/*discard=*/true);
        reconstruction_manager_->Delete(reconstruction_idx);
        break;
      }


      AdjustGlobalBundle(*options_, &mapper);
      FilterPoints(*options_, &mapper);
      FilterImages(*options_, &mapper);

      // Initial image pair failed to register.
      if (reconstruction->NumRegImages() == 0 ||
          reconstruction->NumPoints3D() == 0) {
        mapper.EndReconstruction(/*discard=*/true);
        reconstruction_manager_->Delete(reconstruction_idx);
        // If both initial images are manually specified, there is no need for
        // further initialization trials.
        if (options_->init_image_id1 != -1 && options_->init_image_id2 != -1) {
          break;
        } else {
          continue;
        }
      }
      else {
          mapper.SetKeyframe(image_id1);
          mapper.SetKeyframe(image_id2);
      }

      if (options_->extract_colors) {
        ExtractColors(image_path_, image_id1, reconstruction.get());
      }
    }

    Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

    ////////////////////////////////////////////////////////////////////////////
    // Incremental mapping
    ////////////////////////////////////////////////////////////////////////////

    size_t snapshot_prev_num_reg_images = reconstruction->NumRegImages();
    size_t ba_prev_num_reg_images = reconstruction->NumRegImages();
    size_t ba_prev_num_points = reconstruction->NumPoints3D();

    bool reg_next_success = true;
    bool prev_reg_next_success = true;
    while (reg_next_success) {
      BlockIfPaused();
      if (IsStopped()) {
        LOG(INFO) << "Stop reconstruction...";
        break;
      }

      reg_next_success = false;

      const std::vector<image_t> next_images =
          mapper.FindNextImages(options_->Mapper());

      if (next_images.empty()) {
        LOG(INFO) << "=> No images to register.";
        break;
      }

      for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
        const image_t next_image_id = next_images[reg_trial];
        const Image& next_image = reconstruction->Image(next_image_id);

        PrintHeading1(StringPrintf("Registering image #%d (%d)",
                                   next_image_id,
                                   reconstruction->NumRegImages() + 1));

        LOG(INFO) << StringPrintf("=> Image sees %d / %d points",
                                  next_image.NumVisiblePoints3D(),
                                  next_image.NumObservations());

        reg_next_success =
            mapper.RegisterNextImage(options_->Mapper(), next_image_id);

        if (reg_next_success) {
          const size_t num_tris = TriangulateImage(*options_, next_image, &mapper);
          IterativeLocalRefinement(*options_, next_image_id, &mapper);
          if(mapper.CheckKeyframe(options_->Mapper(), next_image_id, num_tris)) {
            mapper.SetKeyframe(next_image_id);
            LOG(INFO) << StringPrintf("=> Image(%d) is Keyframe",
                                      next_image_id);
          }

          if (reconstruction->NumRegImages() >=
                  options_->ba_global_images_ratio * ba_prev_num_reg_images ||
              reconstruction->NumRegImages() >=
                  options_->ba_global_images_freq + ba_prev_num_reg_images ||
              reconstruction->NumPoints3D() >=
                  options_->ba_global_points_ratio * ba_prev_num_points ||
              reconstruction->NumPoints3D() >=
                  options_->ba_global_points_freq + ba_prev_num_points) {
            IterativeGlobalRefinement(*options_, &mapper);
            ba_prev_num_points = reconstruction->NumPoints3D();
            ba_prev_num_reg_images = reconstruction->NumRegImages();
          }

          if (options_->extract_colors) {
            ExtractColors(image_path_, next_image_id, reconstruction.get());
          }

          if (options_->snapshot_images_freq > 0 &&
              reconstruction->NumRegImages() >=
                  options_->snapshot_images_freq +
                      snapshot_prev_num_reg_images) {
            snapshot_prev_num_reg_images = reconstruction->NumRegImages();
            WriteSnapshot(*reconstruction, options_->snapshot_path);
          }

          Callback(NEXT_IMAGE_REG_CALLBACK);
          LOG(INFO) << "=> Register Done.";
          break;
        } else {
          LOG(INFO) << "=> Could not register, trying another image.";

          // If initial pair fails to continue for some time,
          // abort and try different initial pair.
          const size_t kMinNumInitialRegTrials = 30;
          if (reg_trial >= kMinNumInitialRegTrials &&
              reconstruction->NumRegImages() <
                  static_cast<size_t>(options_->min_model_size)) {
            break;
          }
        }
      }

      const size_t max_model_overlap =
          static_cast<size_t>(options_->max_model_overlap);
      if (mapper.NumSharedRegImages() >= max_model_overlap) {
        break;
      }

      // If no image could be registered, try a single final global iterative
      // bundle adjustment and try again to register one image. If this fails
      // once, then exit the incremental mapping.
      if (!reg_next_success && prev_reg_next_success) {
        reg_next_success = true;
        prev_reg_next_success = false;
        IterativeGlobalRefinement(*options_, &mapper);
      } else {
        prev_reg_next_success = reg_next_success;
      }
    }

    LOG(INFO)<< "Reconstruction is stopped.";

    if (IsStopped()) {
      mapper.EndReconstruction(/*discard=*/false);
      break;
    }

    // Only run final global BA, if last incremental BA was not global.
    if (reconstruction->NumRegImages() >= 2 &&
        reconstruction->NumRegImages() != ba_prev_num_reg_images &&
        reconstruction->NumPoints3D() != ba_prev_num_points) {
      IterativeGlobalRefinement(*options_, &mapper);
    }

    // Remember the total number of registered images before potentially
    // discarding it below due to small size, so we can out of the main loop,
    // if all images were registered.
    const size_t total_num_reg_images = mapper.NumTotalRegImages();

    // If the total number of images is small then do not enforce the minimum
    // model size so that we can reconstruct small image collections.
    // Always keep the first reconstruction, independent of size.
    const size_t min_model_size = std::min<size_t>(
        0.8 * database_cache_->NumImages(), options_->min_model_size);
    if ((options_->multiple_models && reconstruction_manager_->Size() > 1 &&
         reconstruction->NumRegImages() < min_model_size) ||
        reconstruction->NumRegImages() == 0) {
      mapper.EndReconstruction(/*discard=*/true);
      reconstruction_manager_->Delete(reconstruction_idx);
    } else {
      mapper.EndReconstruction(/*discard=*/false);
    }

    Callback(LAST_IMAGE_REG_CALLBACK);

    if (initial_reconstruction_given || !options_->multiple_models ||
        reconstruction_manager_->Size() >=
            static_cast<size_t>(options_->max_num_models) ||
        total_num_reg_images >= database_cache_->NumImages() - 1) {
      LOG(INFO) << "Stop reconstruction...";
      break;
    }
  }
}

}  // namespace colmap