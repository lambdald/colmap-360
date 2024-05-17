from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional, Callable
import os
import sys

COLMAP_DIR = Path(__file__).absolute().parent.parent.parent
SCRIPT_DIR = COLMAP_DIR / Path("scripts/python")

sys.path.append(str(SCRIPT_DIR))
from cmd import run_cmd_with_log
from timer import timer


@dataclass
class ColmapConfig:
    exp_reldir: Path = Path("baseline")
    colmap_bin: Path = Path("colmap")
    camera_model: Literal["OPENCV", "OPENCV_FISHEYE"] = "OPENCV"
    database_relpath: Path = Path("database.db")
    block_size: int = 50
    sparse_reldir: Path = Path("sparse")
    max_num_features: int = 4096
    camera_params: Optional[str] = None


class ColmapRunner:
    config: ColmapConfig

    def __init__(self, exp_reldir: Path = Path("baseline"), config=None):
        if config is None:
            config = ColmapConfig()
        self.config = config
        self.config.exp_reldir = exp_reldir

    @timer
    def run(
        self,
        workspace: Path,
        image_reldir: Path,
        mask_reldir: Path,
        undistorted_colmap_reldir: Path,
        log_reldir: Path,
        failed_callback=None,
    ):

        if (workspace / self.config.exp_reldir / self.config.database_relpath).exists():
            (workspace / self.config.exp_reldir / self.config.database_relpath).unlink()

        cmd_feature = f"""{self.config.colmap_bin} feature_extractor \
            --image_path {workspace / image_reldir} \
            --database_path {workspace/self.config.exp_reldir /self.config.database_relpath} \
            --ImageReader.single_camera_per_folder 1 \
            --ImageReader.camera_model {self.config.camera_model} \
            --SiftExtraction.max_num_features {self.config.max_num_features} \
            """

        if mask_reldir is not None and (workspace / mask_reldir).exists():
            cmd_feature += f" --ImageReader.mask_path {workspace / mask_reldir}"

        if self.config.camera_params is not None:
            cmd_feature += f" --ImageReader.camera_params {self.config.camera_params}"

        run_cmd_with_log(cmd_feature, "posing_feature", log_dir=workspace / log_reldir, failed_callback=failed_callback)

        cmd_matching = f"""{self.config.colmap_bin} exhaustive_matcher \
            --database_path {workspace/self.config.exp_reldir /self.config.database_relpath} \
            --ExhaustiveMatching.block_size {self.config.block_size} \
            --SiftMatching.use_gpu 1 \
        """
        run_cmd_with_log(
            cmd_matching,
            "posing_matching",
            log_dir=workspace / log_reldir,
            timeout=36000,
            failed_callback=failed_callback,
        )

        (workspace / self.config.exp_reldir / self.config.sparse_reldir).mkdir(exist_ok=True, parents=True)

        cmd_mapper = f"""{self.config.colmap_bin} mapper \
            --database_path {workspace/self.config.exp_reldir /self.config.database_relpath} \
            --image_path {workspace / image_reldir} \
            --output_path {workspace /self.config.exp_reldir / self.config.sparse_reldir} \
            --Mapper.multiple_models 0 \
            --Mapper.init_min_num_inliers 500 \
            --Mapper.init_min_tri_angle 8 \
            --Mapper.abs_pose_min_inlier_ratio 0.15 \
            --Mapper.abs_pose_min_num_inliers 50 \
        """

        run_cmd_with_log(
            cmd_mapper, "posing_mapper", log_dir=workspace / log_reldir, timeout=36000, failed_callback=failed_callback
        )

        sparse_dir = workspace / self.config.exp_reldir / self.config.sparse_reldir / "0"

        cmd_ana = f"""\
        {self.config.colmap_bin} model_analyzer --path {sparse_dir} \
        """
        run_cmd_with_log(
            cmd_ana, "posing_report", log_dir=workspace / log_reldir, timeout=36000, failed_callback=failed_callback
        )

        cmd_undistort = f"""\
        {self.config.colmap_bin} image_undistorter --image_path {workspace / image_reldir} \
        --input_path {sparse_dir} \
        --output_path {workspace / undistorted_colmap_reldir} \
        """
        run_cmd_with_log(
            cmd_undistort,
            "image_undistorter",
            log_dir=workspace / log_reldir,
            timeout=36000,
            failed_callback=failed_callback,
        )

    @timer
    def run_v2(
        self,
        workspace: Path,
        image_reldir: Path,
        mask_reldir: Path,
        undistorted_colmap_reldir: Path,
        log_reldir: Path,
        failed_callback: Optional[Callable] = None,
    ):

        self.config.colmap_bin = COLMAP_DIR / "build/src/colmap/exe/colmap"

        if not self.config.colmap_bin.exists():
            self.config.colmap_bin = "colmap"

        if (workspace / self.config.exp_reldir / self.config.database_relpath).exists():
            (workspace / self.config.exp_reldir / self.config.database_relpath).unlink()

        cmd_feature = f"""{self.config.colmap_bin} feature_extractor \
            --image_path {workspace / image_reldir} \
            --database_path {workspace/self.config.exp_reldir /self.config.database_relpath} \
            --ImageReader.single_camera_per_folder 1 \
            --ImageReader.camera_model {self.config.camera_model} \
            --SiftExtraction.max_num_features {self.config.max_num_features} \
            """

        if mask_reldir is not None and (workspace / mask_reldir).exists():
            cmd_feature += f" --ImageReader.mask_path {workspace / mask_reldir}"

        if self.config.camera_params is not None:
            cmd_feature += f" --ImageReader.camera_params {self.config.camera_params}"

        run_cmd_with_log(cmd_feature, "posing_feature", log_dir=workspace / log_reldir, failed_callback=failed_callback)

        cmd_matching = f"""{self.config.colmap_bin} exhaustive_matcher \
            --database_path {workspace/self.config.exp_reldir /self.config.database_relpath} \
            --ExhaustiveMatching.block_size {self.config.block_size} \
            --SiftMatching.use_gpu 1 \
        """
        run_cmd_with_log(
            cmd_matching,
            "posing_matching",
            log_dir=workspace / log_reldir,
            timeout=36000,
            failed_callback=failed_callback,
        )

        database_filename_filtered = str(self.config.database_relpath) + ".filtered.db"

        cmd_filtering = f"""{sys.executable} {COLMAP_DIR}/scripts/python/matching/filtering.py --data-dir {workspace} --input-database-name {self.config.exp_reldir /self.config.database_relpath} --output-database-name {self.config.exp_reldir /database_filename_filtered} --filtering-by-track --filtering-long-track"""
        run_cmd_with_log(
            cmd_filtering,
            "posing_filtering_matches",
            log_dir=workspace / log_reldir,
            timeout=36000,
            failed_callback=failed_callback,
        )

        os.system(
            f"mv {workspace}/{self.config.exp_reldir /database_filename_filtered} {workspace}/{self.config.exp_reldir /self.config.database_relpath}"
        )

        (workspace / self.config.exp_reldir / self.config.sparse_reldir).mkdir(exist_ok=True, parents=True)

        cmd_mapper = f"""{self.config.colmap_bin} sequential_keyframe_mapper \
            --database_path {workspace/self.config.exp_reldir /self.config.database_relpath} \
            --image_path {workspace / image_reldir} \
            --output_path {workspace /self.config.exp_reldir / self.config.sparse_reldir} \
            --Mapper.multiple_models 0 \
        """

        run_cmd_with_log(
            cmd_mapper, "posing_mapper", log_dir=workspace / log_reldir, timeout=36000, failed_callback=failed_callback
        )

        sparse_dir = workspace / self.config.exp_reldir / self.config.sparse_reldir / "0"

        cmd_ana = f"""\
        {self.config.colmap_bin} model_analyzer --path {sparse_dir} \
        """
        run_cmd_with_log(
            cmd_ana, "posing_report", log_dir=workspace / log_reldir, timeout=36000, failed_callback=failed_callback
        )

        cmd_undistort = f"""\
        {self.config.colmap_bin} image_undistorter --image_path {workspace / image_reldir} \
        --input_path {sparse_dir} \
        --output_path {workspace /undistorted_colmap_reldir} \
        """
        run_cmd_with_log(
            cmd_undistort,
            "image_undistorter",
            log_dir=workspace / log_reldir,
            timeout=36000,
            failed_callback=failed_callback,
        )

    @timer
    def run_fisheye(
        self, workspace: Path, image_reldir: Path, mask_reldir: Path, undistorted_colmap_reldir: Path, log_reldir: Path
    ):

        self.config.colmap_bin = COLMAP_DIR / "build/src/colmap/exe/colmap"

        if not self.config.colmap_bin.exists():
            self.config.colmap_bin = "colmap"

        if (workspace / self.config.exp_reldir / self.config.database_relpath).exists():
            (workspace / self.config.exp_reldir / self.config.database_relpath).unlink()

        cmd_feature = f"""{self.config.colmap_bin} feature_extractor \
            --image_path {workspace / image_reldir} \
            --database_path {workspace/self.config.exp_reldir /self.config.database_relpath} \
            --ImageReader.single_camera_per_folder 1 \
            --ImageReader.camera_model {self.config.camera_model} \
            --SiftExtraction.max_num_features {self.config.max_num_features} \
            """

        if mask_reldir is not None and (workspace / mask_reldir).exists():
            cmd_feature += f" --ImageReader.mask_path {workspace / mask_reldir}"

        if self.config.camera_params is not None:
            cmd_feature += f" --ImageReader.camera_params {self.config.camera_params}"

        run_cmd_with_log(cmd_feature, "posing_feature", log_dir=workspace / log_reldir)

        cmd_matching = f"""{self.config.colmap_bin} exhaustive_matcher \
            --database_path {workspace/self.config.exp_reldir /self.config.database_relpath} \
            --ExhaustiveMatching.block_size {self.config.block_size} \
            --SiftMatching.use_gpu 1 \
        """
        run_cmd_with_log(cmd_matching, "posing_matching", log_dir=workspace / log_reldir, timeout=36000)

        (workspace / self.config.exp_reldir / self.config.sparse_reldir).mkdir(exist_ok=True, parents=True)

        cmd_mapper = f"""{self.config.colmap_bin} sequential_keyframe_mapper \
            --database_path {workspace/self.config.exp_reldir /self.config.database_relpath} \
            --image_path {workspace / image_reldir} \
            --output_path {workspace /self.config.exp_reldir / self.config.sparse_reldir} \
            --Mapper.multiple_models 0 \
        """

        run_cmd_with_log(cmd_mapper, "posing_mapper", log_dir=workspace / log_reldir, timeout=36000)

        sparse_dir = workspace / self.config.exp_reldir / self.config.sparse_reldir / "0"

        cmd_ana = f"""\
        {self.config.colmap_bin} model_analyzer --path {sparse_dir} \
        """
        run_cmd_with_log(cmd_ana, "posing_report", log_dir=workspace / log_reldir, timeout=36000)

        cmd_undistort = f"""\
        {self.config.colmap_bin} image_undistorter --image_path {workspace / image_reldir} \
        --input_path {sparse_dir} \
        --output_path {workspace /undistorted_colmap_reldir} \
        """
        run_cmd_with_log(cmd_undistort, "image_undistorter", log_dir=workspace / log_reldir, timeout=36000)

        # cmd_fisheye2pinhole = f"""\
        # {sys.executable} submodules/colmap/scripts/python/camera/fisheye2pinhole.py \
        # --fisheye-dir {workspace} \
        # --pinhole-dir {workspace /undistorted_colmap_reldir} \
        # --fisheye-mask-reldir {mask_reldir} \
        # """
        # run_cmd_with_log(cmd_fisheye2pinhole, "image_fisheye2pinhole", log_dir=workspace / log_reldir, timeout=36000)


def main(
    data_dir: Path,
    exp_name: Path = Path("baseline"),
    version: Literal["baseline", "keyframe"] = "baseline",
    num_features: int = 8192,
):

    from logger import init_global_logger

    init_global_logger(data_dir / exp_name / Path("logs") / "pipeline.log")
    runner = ColmapRunner(exp_name)
    runner.config.max_num_features = num_features
    if version == "baseline":
        runner.run(data_dir, Path("images"), None, exp_name / Path("undistorted_colmap"), exp_name / Path("logs"))
    else:
        runner.run_v2(data_dir, Path("images"), None, exp_name / Path("undistorted_colmap"), exp_name / Path("logs"))


if __name__ == "__main__":

    import tyro

    tyro.cli(main)
