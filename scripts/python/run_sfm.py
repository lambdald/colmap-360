import os, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, List, Optional, Union
import cv2
from concurrent.futures import ThreadPoolExecutor, Future
import tqdm
import tyro
import shutil

BASE_DIR=Path(__file__).parent
COLMAP_DIR=BASE_DIR.parent.parent.absolute()
sys.path.append(str(BASE_DIR))

from misc import CONSOLE, get_all_files
from subprocess_controller import run_cmd_with_log

CONSOLE.print('hello sfm')


@dataclass
class ColmapCmdConfig:
    default: bool = True


@dataclass
class VideoConfig:
    fps: int = 5
    video_frame_to_image_name_fmt: str = "%05d.jpg"

    def get_cmd(self, video_path: Path, image_dir: Path):
        image_dir.mkdir(exist_ok=True, parents=True)
        cmd_fmt = "ffmpeg -i {} -r {}  -q:v 2 -f image2 {}"

        image_fmt = image_dir / self.video_frame_to_image_name_fmt
        cmd = cmd_fmt.format(video_path, self.fps, image_fmt)
        return cmd


@dataclass
class ImagePreprocessConfig:

    max_res: int = -1
    '''if max_res != -1, resize the image to max_res'''

    images_per_equirect: Literal[8, 14] = 8


    def main(self, image_dir: Path):
        # resize
        image_paths = get_all_files(image_dir, "*.jpg")

        img = cv2.imread(str(image_paths[0]))
        h, w = img.shape[:2]
        resize_img = False
        if self.max_res > 0 and (h > self.max_res or w > self.max_res):
            print("rezie image...")
            print(f"origin shape = {w}x{h}")
            resize_img = True
            if h > w:
                new_h = self.max_res
                ratio = new_h / h
                new_w = int(ratio * w)
            else:
                new_w = self.max_res
                ratio = new_w / w
                new_h = int(ratio * h)

            print(f"new shape = {new_w}x{new_h}")

        def image_process(img_path: Path):
            img = cv2.imread(str(img_path))
            if resize_img:
                img = cv2.resize(img, (new_w, new_h))

            if not is_good_image(img):
                CONSOLE.print('delete bad image:', img_path)
                img_path.unlink()
            else:
                cv2.imwrite(str(img_path), img)

        with ThreadPoolExecutor(max_workers=24) as exe:
            fs: List[Future] = []
            for img_path in tqdm.tqdm(image_paths, desc="resize image"):
                f = exe.submit(image_process, img_path)
                fs.append(f)

            for f in tqdm.tqdm(fs):
                f.result()


@dataclass
class ImageReaderConfig(ColmapCmdConfig):

    # Optional root path to folder which contains image masks. For a given image, the corresponding mask must have the same sub-path below this root as the image has below image_path. The filename must be equal, aside from the added extension .png. For example, for an image image_path/abc/012.jpg, the mask would be mask_path/abc/012.jpg.png. No features will be extracted in regions where the mask image is black (pixel intensity value 0 in grayscale).
    camera_model = "OPENCV"  # (default: SIMPLE_RADIAL)
    # Possible values: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, FOV, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, THIN_PRISM_FISHEYE
    # Name of the camera model. See: Camera Models

    single_camera = 0  # (default: 0)
    # Whether to use the same camera for all images.

    single_camera_per_folder = 1  # (default: 0)
    # Whether to use the same camera for all images in the same sub-folder.

    single_camera_per_image = 0  # (default: 0)
    # Whether to use a different camera for each image.

    existing_camera_id = -1  # (default: -1)
    # Whether to explicitly use an existing camera for all images. Note that in this case the specified camera model and parameters are ignored.

    camera_params = ""
    # Manual specification of camera parameters. If empty, camera parameters will be extracted from EXIF, i.e. principal point and focal length.

    default_focal_length_factor = 1.2  # (default: 1.2)
    # If camera parameters are not specified manually and the image does not have focal length EXIF information, the focal length is set to the value default_focal_length_factor * max(width, height).

    camera_mask_path = ""
    # Optional path to an image file specifying a mask for all images. No features will be extracted in regions where the mask is black (pixel intensity value 0 in grayscale).

    def get_cmd(self, mask_path = None) -> str:
        cmd = f""" \
            --ImageReader.single_camera 0 \
            --ImageReader.single_camera_per_folder {self.single_camera_per_folder} \
            --ImageReader.camera_model {self.camera_model} \
        """
        if self.default:
            cmd = f""" \
            --ImageReader.single_camera_per_folder 1 \
            --ImageReader.camera_model {self.camera_model} \
            """
        if mask_path is not None:
            cmd += f' --ImageReader.mask_path {mask_path}'
        return cmd


@dataclass
class SiftExtractionConfig(ColmapCmdConfig):
    num_threads = -1  # (default: -1)
    # Number of threads for feature extraction.

    use_gpu = 1  # (default: 1)
    # Whether to use the GPU for feature extraction.

    gpu_index = -1  # (default: -1)
    # Index of the GPU used for feature extraction. For multi-GPU extraction, you should separate multiple GPU indices by comma, e.g. "0,1,2,3". See: Multi-GPU support in feature extraction/matching

    max_image_size = 4800  # (default: 3200)
    # Maximum image size, otherwise image will be down-scaled.

    max_num_features: int = 8192  # (default: 8192)
    # Maximum number of features to detect, keeping larger-scale features.

    first_octave = -1  # (default: -1)
    # First octave in the pyramid, i.e. -1 upsamples the image by one level. By convention, the octave of index 0 starts with the image full resolution. Specifying an index greater than 0 starts the scale space at a lower resolution (e.g. 1 halves the resolution). Similarly, specifying a negative index starts the scale space at an higher resolution image, and can be useful to extract very small features (since this is obtained by interpolating the input image, it does not make much sense to go past -1).

    num_octaves = 4  # (default: 4)
    # Number of octaves. Increasing the scale by an octave means doubling the size of the smoothing kernel, whose effect is roughly equivalent to halving the image resolution. By default, the scale space spans as many octaves as possible (i.e. roughly log2(min(width, height))), which has the effect of searching keypoints of all possible sizes.

    octave_resolution = 3  # (default: 3)
    # Number of levels per octave. Each octave is sampled at this given number of intermediate scales. Increasing this number might in principle return more refined keypoints, but in practice can make their selection unstable due to noise.

    peak_threshold = 0.0067  # (default: 0.0067)
    # Peak threshold for detection. This is the minimum amount of contrast to accept a keypoint. Increase to eliminate more keypoints.

    edge_threshold = 10  # (default: 10)
    # Edge threshold for detection. Decrease to eliminate more keypoints.

    estimate_affine_shape: int = 0  # (default: 0)
    # Estimate affine shape of SIFT features in the form of oriented ellipses as opposed to original SIFT which estimates oriented disks.

    max_num_orientations = 2  # (default: 2)
    # aximum number of orientations per keypoint if not SiftExtraction.estimate_affine_shape.

    upright = 0  # (default: 0)
    # Fix the orientation to 0 for upright features.

    domain_size_pooling: int = 0  # (default: 0)
    # Enable the more discriminative DSP-SIFT features instead of plain SIFT. Domain-size pooling computes an average SIFT descriptor across multiple scales around the detected scale. DSP-SIFT outperforms standard SIFT in most cases.
    # This was proposed in Domain-Size Pooling in Local Descriptors: DSP-SIFT, J. Dong and S. Soatto, CVPR 2015. This has been shown to outperform other SIFT variants and learned descriptors in Comparative Evaluation of Hand-Crafted and Learned Local Features, Schönberger, Hardmeier, Sattler, Pollefeys, CVPR 2016.

    # dsp_min_scale (default: 0.1667)
    # dsp_max_scale (default: 3)
    # dsp_num_scales (default: 10)
    # Domain-size pooling parameters. See: SiftExtraction.domain_size_pooling

    def get_cmd(self) -> str:
        cmd = f"""\
            --SiftExtraction.use_gpu {self.use_gpu} \
            --SiftExtraction.estimate_affine_shape {self.estimate_affine_shape} \
            --SiftExtraction.domain_size_pooling {self.domain_size_pooling} \
            --SiftExtraction.max_num_features {self.max_num_features} \
            --SiftExtraction.num_octaves {self.num_octaves} \
            --SiftExtraction.octave_resolution {self.octave_resolution} \
            """

        if self.default:
            cmd = f"""\
            --SiftExtraction.use_gpu {self.use_gpu} \
            --SiftExtraction.max_num_features {self.max_num_features} \
            """
        return cmd


@dataclass
class FeatureExtractorConfig(ColmapCmdConfig):
    reader: ImageReaderConfig = ImageReaderConfig()
    extraction: SiftExtractionConfig = SiftExtractionConfig()

    def __post_init__(self):
        self.reader.default = self.default
        self.extraction.default = self.default

    def get_cmd(self, mask_path=None) -> str:
        cmd_feature = f"""feature_extractor \
        {self.reader.get_cmd(mask_path)} {self.extraction.get_cmd()} \
        """
        return cmd_feature


@dataclass
class SiftMatching(ColmapCmdConfig):

    num_threads: int =-1    # --SiftMatching.num_threads arg (=-1)
    # Number of threads for feature matching and geometric verification.


    use_gpu:int =1 # --SiftMatching.use_gpu arg (=1)
    # Whether to use the GPU for feature matching.


    gpu_index:int =-1  # --SiftMatching.gpu_index arg (=-1)
    # Index of the GPU used for feature matching. For multi-GPU matching, you should separate multiple GPU indices by comma, e.g. "0,1,2,3". See: Multi-GPU support in feature extraction/matching

    max_ratio:float= 0.8 # --SiftMatching.max_ratio arg (=0.80000000000000004)
    '''Maximum distance ratio between first and second best match.'''

    
    max_distance: float =0.7  # --SiftMatching.max_distance arg (=0.69999999999999996)
    # Maximum distance to best match.

    
    cross_check: int=1 # --SiftMatching.cross_check arg (=1)
    # Whether to enable cross checking in matching.

    guided_matching: int = 0 # --SiftMatching.guided_matching arg (=0)
    # Whether to perform guided matching, if geometric verification succeeds.

    max_num_matches: int =32768 # --SiftMatching.max_num_matches arg (=32768)
    # Maximum number of matches.


    def get_cmd(self) -> str:
        cmd = f""" \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.guided_matching {self.guided_matching} \
        --SiftMatching.max_num_matches {self.max_num_matches} \
        --SiftMatching.max_ratio {self.max_ratio} \
        """

        if self.default:
            cmd = """ \
            --SiftMatching.use_gpu 1 \
            """
        return cmd


@dataclass
class ExhaustiveMatching(ColmapCmdConfig):
    name: str = "exhaustive"
    block_size = 50  # (default: 50)
    # Block size, i.e. number of images to simultaneously load into memory.

    def get_cmd(self) -> str:
        cmd = f"""\
        exhaustive_matcher \
        --ExhaustiveMatching.block_size {self.block_size} \
        """

        if self.default:
            cmd = """\
            exhaustive_matcher \
            """

        return cmd


@dataclass
class SequentialMatching(ColmapCmdConfig):
    name: str = "sequential"
    overlap: int = 30
    quadratic_overlap: int = 1
    loop_detection: int = 1
    vocab_tree_path: Path = COLMAP_DIR / "vocab_tree_flickr100K_words256K.bin"

    def get_cmd(self) -> str:
        cmd = f"""\
        sequential_matcher \
        --SequentialMatching.overlap {self.overlap} \
        --SequentialMatching.quadratic_overlap {self.quadratic_overlap} \
        --SequentialMatching.loop_detection {self.loop_detection} \
        --SequentialMatching.vocab_tree_path {self.vocab_tree_path} \
        """

        if self.default:
            cmd = f"""\
            sequential_matcher \
            --SequentialMatching.quadratic_overlap {self.quadratic_overlap} \
            --SequentialMatching.loop_detection {self.loop_detection} \
            --SequentialMatching.vocab_tree_path {self.vocab_tree_path} \
            """
        return cmd


@dataclass
class TwoViewGeometry(ColmapCmdConfig):
    min_num_inliers: int = 15    #   --TwoViewGeometry.min_num_inliers arg (=15)
    # Minimum number of inliers for an image pair to be considered as geometrically verified.

    multiple_models: int = 0    #   --TwoViewGeometry.multiple_models arg (=0)
    # Whether to attempt to estimate multiple geometric models per image pair.

    compute_relative_pose: int = 0    #   --TwoViewGeometry.compute_relative_pose arg (=0)
    # Whether to estimate the relative pose between the two images and save them to the database.

    max_error: float = 4.0    #   --TwoViewGeometry.max_error arg (=4)
    # Maximum epipolar error in pixels for geometric verification.

    confidence: float =0.999    #   --TwoViewGeometry.confidence arg (=0.999)
    # Confidence threshold for geometric verification.

    max_num_trials: int = 10000    #   --TwoViewGeometry.max_num_trials arg (=10000)
    # Maximum number of RANSAC iterations. Note that this option overrules the SiftMatching.min_inlier_ratio option.

    min_inlier_ratio: float =0.25    #   --TwoViewGeometry.min_inlier_ratio arg (=0.25)


    def get_cmd(self):
        prefix = '--TwoViewGeometry'
        # return ''

        cmd = f''' \
            {prefix}.min_num_inliers {self.min_num_inliers} \
            {prefix}.multiple_models {self.multiple_models} \
            {prefix}.compute_relative_pose {self.compute_relative_pose} \
            {prefix}.max_error {self.max_error} \
            {prefix}.confidence {self.confidence} \
            {prefix}.max_num_trials {self.max_num_trials} \
            {prefix}.min_inlier_ratio {self.min_inlier_ratio} \
            '''
        
        if self.default:
            cmd = ''
        return cmd

@dataclass
class FeatureMatchingConfig(ColmapCmdConfig):
    # match mode in ["exhaustive","sequential","spatial","transitive","vocab_tree"]
    # ref: https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification

    feature_matching: SiftMatching = SiftMatching()
    two_view_geometry: TwoViewGeometry = TwoViewGeometry()
    matching_policy: Union[ExhaustiveMatching, SequentialMatching] = SequentialMatching()

    def __post_init__(self):
        self.feature_matching.default = self.default
        self.matching_policy.default = self.default
        self.two_view_geometry.default = self.default

    def get_cmd(self):
        cmd = f"{self.matching_policy.get_cmd()} {self.feature_matching.get_cmd()} {self.two_view_geometry.get_cmd()}"
        return cmd


@dataclass
class MapperConfig(ColmapCmdConfig):
    #! General
    min_num_matches: int = 15  # (default: 15)
    # The minimum number of matches for inlier matches to be considered.

    ignore_watermarks: int = 0  # (default: 0)
    # Whether to ignore the inlier matches of watermark image pairs.

    multiple_models: int = 0  # (default: 1)
    # Whether to reconstruct multiple sub-models.

    max_num_models: int = 50  # (default: 50)
    # The number of sub-models to reconstruct.

    max_model_overlap: int = 20  # (default: 20)
    # The maximum number of overlapping images between sub-models. If the current sub-models shares more than this number of images with another model, then the reconstruction is stopped.

    min_model_size: int = 10  # (default: 10)
    # The minimum number of registered images of a sub-model, otherwise the sub-model is discarded.

    extract_colors: int  = 1  # (default: 1)
    # Whether to extract colors for reconstructed points.

    num_threads: int = -1  # (default: -1)
    # The number of threads to use during reconstruction.

    snapshot_path: str = ""
    snapshot_images_freq: int = 0  # (default: 0)
    # Path to a folder with reconstruction snapshots during incremental reconstruction. Snapshots will be saved according to the specified frequency of registered images.

    fix_existing_images: int = 0  # (default: 0)
    # If reconstruction is provided as input, fix the existing image poses.

    #! Init
    init_image_id1 = -1  # (default: -1)
    init_image_id2 = -1  # (default: -1)
    # The image identifiers used to initialize the reconstruction. Note that only one or both image identifiers can be specified. In the former case, the second image is automatically determined.

    init_num_trials = 200  # (default: 200)
    # The number of trials to initialize the reconstruction.

    init_min_num_inliers = 100  # (default: 100)
    # Minimum number of inliers for initial image pair.

    init_max_error = 4  # (default: 4)
    # Maximum error in pixels for two-view geometry estimation for initial image pair.

    init_max_forward_motion = 0.95  # (default: 0.95)
    # Maximum forward motion for initial image pair.

    init_min_tri_angle = 16  # (default: 16)
    # Minimum triangulation angle for initial image pair.

    init_max_reg_trials = 5  # (default: 2)
    # Maximum number of trials to use an image for initialization.

    #! Registration
    abs_pose_max_error = 12  # (default: 12)
    # Maximum reprojection error in absolute pose estimation.

    abs_pose_min_num_inliers = 30  #  (default: 30)
    # Minimum number of inliers in absolute pose estimation.

    abs_pose_min_inlier_ratio = 0.15  #  (default: 0.25)
    # Minimum inlier ratio in absolute pose estimation.

    max_reg_trials = 3  # (default: 3)
    # Maximum number of trials to register an image.
    #! Triangulation

    tri_max_transitivity = 1  # (default: 1)
    # Maximum transitivity to search for correspondences.

    tri_create_max_angle_error = 2  # (default: 2)
    # Maximum angular error to create new triangulations.

    tri_continue_max_angle_error = 2  # (default: 2)
    # Maximum angular error to continue existing triangulations.

    tri_merge_max_reproj_error = 4  # (default: 4)
    # Maximum reprojection error in pixels to merge triangulations.

    tri_complete_max_reproj_error = 4  # (default: 4)
    # Maximum reprojection error to complete an existing triangulation.

    tri_complete_max_transitivity = 5  # (default: 5)
    # Maximum transitivity for track completion.

    tri_re_max_angle_error = 5  # (default: 5)
    # Maximum angular error to re-triangulate under-reconstructed image pairs.

    tri_re_min_ratio = 0.2  # (default: 0.2)
    # Minimum ratio of common triangulations between an image pair over the number of correspondences between that image pair to be considered as under-reconstructed.

    tri_re_max_trials = 1  # (default: 1)
    # Maximum number of trials to re-triangulate an image pair.

    tri_min_angle = 1.5  # (default: 1.5)
    # Minimum pairwise triangulation angle for a stable triangulation. If your images are taken from far distance with respect to the scene, you can try to reduce the minimum triangulation angle

    tri_ignore_two_view_tracks = 1  # (default: 1)
    # Whether to ignore two-view feature tracks in triangulation, resulting in fewer 3D points than possible. Triangulation of two-view tracks can in rare cases improve the stability of sparse image collections by providing additional constraints in bundle adjustment.

    #! BA

    ba_refine_focal_length = 1  # (default: 1)
    ba_refine_principal_point = 0  # (default: 0)
    ba_refine_extra_params = 1  # (default: 1)
    # Which intrinsic parameters to optimize during the reconstruction.

    ba_min_num_residuals_for_multi_threading = 50000  # (default: 50000)
    # The minimum number of residuals per bundle adjustment problem to enable multi-threading solving of the problems.

    ba_local_num_images = 6  # (default: 6)
    # The number of images to optimize in local bundle adjustment.

    ba_local_function_tolerance = 0  # (default: 0)
    # Ceres solver function tolerance for local bundle adjustment

    ba_local_max_num_iterations = 25  # (default: 25)
    # The maximum number of local bundle adjustment iterations.

    ba_global_use_pba = 0  # (default: 0)
    # Whether to use PBA (Parralel Bundle Adjustment) in global bundle adjustment. See: https://grail.cs.washington.edu/projects/mcba/, https://github.com/cbalint13/pba

    ba_global_pba_gpu_index = -1  # (default: -1)
    # The GPU index for PBA bundle adjustment.

    ba_global_images_ratio = 1.1  # (default: 1.1)
    ba_global_points_ratio = 1.1  # (default: 1.1)
    ba_global_images_freq = 500  # (default: 500)
    ba_global_points_freq = 250000  # (default: 250000)
    # The growth rates after which to perform global bundle adjustment.

    ba_global_function_tolerance = 0  # (default: 0)
    # Ceres solver function tolerance for global bundle adjustment

    ba_global_max_num_iterations = 50  # (default: 50)
    # The maximum number of global bundle adjustment iterations.

    ba_global_max_refinements = 5  # (default: 5)
    ba_global_max_refinement_change = 0.005  # (default: 0.0005)

    ba_local_max_refinements = 2  # (default: 2)
    ba_local_max_refinement_change = 0.001  # (default: 0.001)
    # The thresholds for iterative bundle adjustment refinements.

    local_ba_min_tri_angle = 6  # (default: 6)
    # Minimum triangulation for images to be chosen in local bundle adjustment.

    ba_use_cuda = False

    #! Filter

    min_focal_length_ratio = 0.1  # (default: 0.1)
    max_focal_length_ratio = 10  # (default: 10)
    max_extra_param = 0.1  # (default: 1)
    # Thresholds for filtering images with degenerate intrinsics.

    filter_max_reproj_error = 4  # (default: 4)
    # Maximum reprojection error in pixels for observations.

    filter_min_tri_angle = 1.5  # (default: 1.5)
    # Minimum triangulation angle in degrees for stable 3D points.

    def get_cmd(self):
        cmd_mapper = f"""mapper \
        --Mapper.multiple_models {self.multiple_models} \
        --Mapper.init_min_tri_angle {self.init_min_tri_angle} \
        --Mapper.init_max_reg_trials {self.init_max_reg_trials} \
        --Mapper.init_min_num_inliers {self.init_min_num_inliers} \
        --Mapper.init_max_error {self.init_max_error} \
        --Mapper.abs_pose_min_inlier_ratio {self.abs_pose_min_inlier_ratio} \
        --Mapper.abs_pose_min_num_inliers {self.abs_pose_min_num_inliers} \
        --Mapper.max_reg_trials {self.max_reg_trials} \
        --Mapper.local_ba_min_tri_angle {self.local_ba_min_tri_angle} \
        --Mapper.filter_min_tri_angle {self.filter_min_tri_angle} \
        --Mapper.ba_local_num_images {self.ba_local_num_images} \
        --Mapper.abs_pose_max_error {self.abs_pose_max_error} \
        --Mapper.tri_min_angle {self.tri_min_angle} \
        --Mapper.tri_re_max_trials {self.tri_re_max_trials} \
        --Mapper.tri_ignore_two_view_tracks {self.tri_ignore_two_view_tracks} \
        --Mapper.num_threads {self.num_threads} \
        --Mapper.ba_global_max_num_iterations {self.ba_global_max_num_iterations} \
        """

        if self.default:
            cmd_mapper = f"""mapper \
            --Mapper.multiple_models {self.multiple_models} \
            --Mapper.num_threads {self.num_threads} \
            """
        return cmd_mapper


@dataclass
class SfmConfig(ColmapCmdConfig):
    data_dir: Path = Path()
    video_path: Path = Path()
    workspace: Path = Path()
    capture_device: Literal['phone', '360'] = 'phone'

    panoramic_process: Literal['to_pinhole', 'keep'] = 'keep'
    use_tsift: bool = True


    video_mode: bool = False
    force: bool = False

    use_gpu: bool = True
    gpu_id: int = 0
    log_reldir: Path = Path("logs/sfm")
    origin_frames_reldir: Path = Path('images')
    semantic_reldir: Path = Path('semantics')
    mask_reldir: Path = Path('masks')
    images_reldir = Path("images")
    colmap_model_dir = Path("colmap")
    pixsfm_model_dir = Path("pixsfm")
    final_model_dir = Path("sparse")
    database_relpath = Path("database.db")
    undistorted_dir = Path("undistorted_colmap")

    num_images = -1
    dsp_sift = False
    camera_params = ""

    # colmap_bin: Path = Path("colmap")
    official_colmap_bin: Path = COLMAP_DIR / 'build_release/src/colmap/exe/colmap'

    colmap_bin: Path = Path("colmap")

    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    matcher: FeatureMatchingConfig = FeatureMatchingConfig()
    mapper: MapperConfig = MapperConfig()

    video: VideoConfig = field(default_factory=lambda: VideoConfig())
    image_process: ImagePreprocessConfig = field(default_factory=lambda: ImagePreprocessConfig())

    rig_config_relpath: Path = Path('rig_config.json')


    def __post_init__(self):
        print("*********** default:", self.default, "**************")

        if self.video_path:
            self.workspace = self.data_dir / self.video_path.stem

        self.mapper.ba_global_pba_gpu_index = self.gpu_id
        self.feature_extractor.default = self.default
        self.matcher.default = self.default
        self.mapper.default = self.default

        if self.capture_device == '360':
            if self.panoramic_process == 'to_pinhole':
                self.feature_extractor.reader.camera_model = 'PINHOLE'
                self.colmap_bin = self.official_colmap_bin
            elif self.panoramic_process == 'keep':
                self.feature_extractor.reader.camera_model = 'PANORAMIC'
                self.colmap_bin = self.panoramic_colmap_bin
            else:
                raise NotImplementedError
        elif self.capture_device == 'phone':
            self.feature_extractor.reader.camera_model = 'OPENCV'
            self.colmap_bin = self.official_colmap_bin
        else:
            raise NotImplementedError
        
        if self.video_path != Path():
            self.video_mode = True

@dataclass
class RigBundleAdjustmentConfig:

    pass

class ColmapSfm:
    config: SfmConfig

    def __init__(self, config) -> None:
        self.config = config

    @property
    def workspace(self) -> Path:
        workspace = self.config.workspace
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    @property
    def image_dir(self) -> Path:
        image_dir = self.workspace / self.config.images_reldir
        return image_dir

    @property
    def database_path(self) -> Path:
        return self.workspace / self.config.database_relpath

    @property
    def sparse_dir(self) -> Path:
        sparse_dir = self.workspace / self.config.colmap_model_dir
        return sparse_dir

    @property
    def log_dir(self) -> Path:
        log_dir = self.workspace / self.config.log_reldir
        return log_dir

    def get_cuda_prefix(self):
        cuda_prefix = f"CUDA_VISIBLE_DEVICES={self.config.gpu_id}"
        return cuda_prefix

    def run_feature_extractor(self):
        if self.config.force:
            if (self.workspace / self.config.database_relpath).exists():
                (self.workspace / self.config.database_relpath).unlink()

        image_dir: Path = self.workspace / self.config.images_reldir
        assert image_dir.exists(), image_dir
        log_dir = self.workspace / self.config.log_reldir
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd_feature = f"""{self.get_cuda_prefix()} \
        {self.config.colmap_bin} \
        {self.config.feature_extractor.get_cmd()} \
            --database_path {self.workspace/self.config.database_relpath} \
            --image_path {image_dir} \
        """
        if self.config.capture_device == '360' and self.config.panoramic_process == 'to_pinhole':
            import json
            with open(self.config.workspace / self.config.rig_config_relpath) as infile:
                data = json.load(infile)
            
            params = data[0]['cameras'][0]['params']

            params_str = ','.join(map(str, params))
            cmd_feature += ' --ImageReader.camera_params ' + params_str

        run_cmd_with_log(cmd_feature, "feature_extractor", log_dir=log_dir, timeout=100000)

        ## update feature
        ## bad result.
        # if self.config.capture_device == '360' and self.config.panoramic_process == 'keep' and self.config.use_tsift:
        #     extractor = colmap_feature.FeatureExtractor(colmap_feature.FeatureExtractorConfig())
        #     # extractor.compute(image_dir, self.database_path, self.workspace / self.config.mask_reldir)
        #     extractor.compute(image_dir, self.database_path, self.workspace/ self.config.mask_reldir)


    def run_matcher(self):
        use_gpu = 1 if self.config.use_gpu else 0
        self.config.matcher.feature_matching.use_gpu = use_gpu

        log_dir = self.workspace / self.config.log_reldir
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd_matcher = self.config.matcher.get_cmd()
        cmd_matcher = f"{self.get_cuda_prefix()} {self.config.colmap_bin} {cmd_matcher} \
        --database_path {self.database_path}"
        run_cmd_with_log(cmd_matcher, "matcher", log_dir=log_dir, timeout=100000)

    def run_mapper(self):
        
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        cmd_mapper = f""" {self.get_cuda_prefix()} \
        {self.config.colmap_bin} {self.config.mapper.get_cmd()} \
            --database_path {self.database_path} \
            --image_path {self.image_dir} \
            --output_path {self.sparse_dir} \
        """
        run_cmd_with_log(cmd_mapper, "mapper", log_dir=self.log_dir, timeout=1000000)
        os.system(f"cp {self.sparse_dir}/0/* {self.sparse_dir}")

        cmd_ana = f"""\
        {self.config.colmap_bin} model_analyzer --path {self.sparse_dir} \
        """
        run_cmd_with_log(cmd_ana, "model_analyzer", log_dir=self.log_dir, timeout=1000000)

        cmd_cvt = f"""\
        {self.config.colmap_bin} model_converter --input_path {self.sparse_dir} --output_path {self.sparse_dir} --output_type TXT\
        """
        run_cmd_with_log(cmd_cvt, "model_converter", log_dir=self.log_dir, timeout=1000000)

        if self.config.capture_device == '360' and self.config.panoramic_process == 'to_pinhole':
            self.run_rig_ba()

    def run_rig_ba(self):

        assert self.config.capture_device == '360'

        cmd_rigba = f''' {self.config.colmap_bin} rig_bundle_adjuster \
            --input_path {self.sparse_dir} \
            --output_path {self.sparse_dir} \
            --rig_config_path {self.workspace/self.config.rig_config_relpath}
        '''
        run_cmd_with_log(cmd_rigba, "rig_bundle_adjuster", log_dir=self.log_dir, timeout=1000000)



    def run_undistorter(self):
        cmd_undistort = f"""\
        {self.config.colmap_bin} image_undistorter --image_path {self.image_dir} \
        --input_path {self.sparse_dir} \
        --output_path {self.workspace / self.config.undistorted_dir} \
        """
        run_cmd_with_log(cmd_undistort, "image_undistorter", log_dir=self.log_dir, timeout=1000000)

        cmd_cvt = f"""\
        {self.config.colmap_bin} model_converter --input_path {self.workspace / self.config.undistorted_dir}/sparse --output_path {self.workspace / self.config.undistorted_dir}/sparse --output_type TXT\
        """
        run_cmd_with_log(cmd_cvt, "model_converter_undistort", log_dir=self.log_dir, timeout=1000000)

    def run_colmap_pano_to_pinhole(self):
        cmd_converter = f'''{self.get_cuda_prefix()} {sys.executable} {BASE_DIR}/sfm/panorama/colmap_panoramic_to_pinhole.py \
        --panoramic-dir {self.workspace} \
        --pinhole-dir {self.workspace / self.config.undistorted_dir}
        '''
        run_cmd_with_log(cmd_converter, "colmap_panoramic_to_pinhole", log_dir=self.log_dir, timeout=1000000)

    def run_video2images(self):
        cmd_video = self.config.video.get_cmd(self.config.video_path, self.config.workspace / self.config.origin_frames_reldir)
        run_cmd_with_log(cmd_video, "video2images", log_dir=self.log_dir, timeout=1000000)

    def run_image_preprocess(self):
        if self.image_dir.exists() and self.config.origin_frames_reldir != self.config.images_reldir:
            shutil.rmtree(self.image_dir)

        self.run_semantic()
            
        # 360 to pinhole
        if self.config.capture_device == '360':
            if self.config.panoramic_process == 'to_pinhole':
                self.config.image_process.run_pano2pinhole(self.config.workspace / self.config.origin_frames_reldir, self.image_dir, self.config.workspace/self.config.rig_config_relpath)

                new_mask_dir = str(self.config.mask_reldir)  + '_pinhole'
                self.config.image_process.run_pano2pinhole(self.config.workspace / self.config.mask_reldir, new_mask_dir, self.config.workspace/self.config.rig_config_relpath)
                self.config.mask_reldir = new_mask_dir

            elif self.config.panoramic_process == 'keep':
                shutil.copytree(self.config.workspace / self.config.origin_frames_reldir, self.workspace / self.config.images_reldir)
        else:
            if self.config.origin_frames_reldir != self.config.images_reldir:
                shutil.copytree(self.config.workspace / self.config.origin_frames_reldir, self.image_dir)

        if self.config.video_mode:
            self.config.image_process.main(self.image_dir)


    def run_semantic(self):

        cmd = f'''{self.get_cuda_prefix()} {sys.executable} {BASE_DIR}/sfm/panorama/semantic_segment.py \
            --image-dir {self.config.workspace/self.config.origin_frames_reldir} \
            --semantic-dir {self.config.workspace/self.config.semantic_reldir} \
            --mask-dir {self.config.workspace/self.config.mask_reldir}
        '''
        run_cmd_with_log(cmd, "semantic", log_dir=self.log_dir, timeout=1000000)


    def execute(self):

        if self.config.force:
            if self.log_dir.exists():
                shutil.rmtree(self.log_dir)
            if self.sparse_dir.exists():
                shutil.rmtree(self.sparse_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        print("sfm workspace:", self.workspace)

        self.run_feature_extractor()
        self.run_matcher()
        self.run_mapper()

        if self.config.capture_device == 'phone':
            self.run_undistorter()
        elif self.config.capture_device == '360':
            self.run_colmap_pano_to_pinhole()
        else:
            raise NotImplementedError


        if (self.workspace / self.config.final_model_dir).exists():
            shutil.rmtree(self.workspace / self.config.final_model_dir)

        shutil.copytree(self.sparse_dir, self.workspace / self.config.final_model_dir)

        
def run_sfm():
    config = tyro.cli(SfmConfig)
    sfm = ColmapSfm(config)
    sfm.execute()


if __name__ == '__main__':
    run_sfm()
