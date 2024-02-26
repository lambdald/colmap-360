"""
Description: file content
"""

from typing import Union
import os
import random
import importlib

import torch
import numpy as np
import cv2
import time
import datetime


def get_datatime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def seed_everything(seed):
    """Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def find_object_by_name(arch_path: str):
    module_name, obj_name = arch_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, obj_name)
    return obj


def build_from_name(arch_path: str, kwargs: dict):
    """根据字符串名字调用方法或构建对象
    Args:
        arch_path (str): [description]
        kwargs (dict): [description]
    Returns:
        [type]: [description]
    """
    obj = find_object_by_name(arch_path)
    print(arch_path, obj)
    return obj(**kwargs)


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", 0))


def float_to_colormap(
    src_data: Union[np.ndarray, torch.Tensor], min_val=None, max_val=None, out_type="numpy"
) -> Union[np.ndarray, torch.FloatTensor]:
    """将2d数据转换为colormap, 主要用于可视化

    Args:
        src_data (Union[np.ndarray, torch.Tensor]): 输入数据可以是numpy或者torch类型
        min_val (_type_, optional): 如果不为None, 会clip数据. Defaults to None.
        max_val (_type_, optional): 如果不为None, 会clip数据. Defaults to None.
        out_type (str, optional): numpy or torch. Defaults to 'numpy'.

    Raises:
        NotImplementedError: 只支持numpy和torch类型数据

    Returns:
        Union[np.ndarray, torch.FloatTensor]: 返回numpy.uint8[h, w, 3]或者torch.FloatTensor[3, h, w]
    """

    if isinstance(src_data, torch.Tensor):
        data_np = src_data.detach().cpu().numpy()
    elif isinstance(src_data, np.ndarray):
        data_np = src_data
    else:
        raise NotImplementedError

    assert data_np.ndim == 2, "only support 2d data"
    data_np = np.nan_to_num(data_np)

    if min_val is not None or max_val is not None:
        data_np = np.clip(data_np, a_min=min_val, a_max=max_val)

    data = cv2.normalize(data_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    colormap = cv2.applyColorMap(data, cv2.COLORMAP_HSV)  # [h, w, 3]

    if out_type == "numpy":
        # [h, w, 3] np.uint8
        return colormap
    elif out_type == "torch":
        # [3, h, w]
        colormap = torch.from_numpy(colormap).permute(2, 0, 1).float() / 255.0
    return colormap


def get_current_process_memory_gb() -> float:
    # 获取当前进程内存占用。(GB)
    import psutil

    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024.0**3


def get_current_torch_cuda_memory(device=None):
    return torch.cuda.memory_allocated(device=device) / 1024**3


from contextlib import nullcontext
from typing import Optional, List
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
import rich

from rich.traceback import install

CONSOLE = Console(width=120)
install(show_locals=False, console=CONSOLE)


class ItersPerSecColumn(ProgressColumn):
    """Renders the iterations per second for a progress bar."""

    def __init__(self, suffix="it/s") -> None:
        super().__init__()
        self.suffix = suffix

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:.2f} {self.suffix}", style="progress.data.speed")


def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager that does nothing is verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=spinner)


def get_progress(description: str, suffix: Optional[str] = None):
    """Helper function to return a rich Progress object."""
    progress_list = [TextColumn(description), BarColumn(), TaskProgressColumn(show_speed=True)]
    progress_list += [ItersPerSecColumn(suffix=suffix)] if suffix else []
    progress_list += [TimeRemainingColumn(elapsed_when_finished=True, compact=True)]
    progress = Progress(*progress_list)
    return progress


def get_all_files(root: Union[str, Path], pattern: str) -> List[Path]:
    """get files in root dir, including files in subdirs.

    Args:
        root (str): _description_
        pattern (str): _description_

    Returns:
        List[Path]: _description_
    """
    root = Path(root)
    files = []
    for img_path in root.glob(pattern):
        files.append(img_path)

    for subdir in root.iterdir():
        if subdir.is_dir():
            files += get_all_files(subdir, pattern)
    return sorted(files)



def glob_imgs(path) -> List[Path]:
    imgs = []
    for ext in ["*.png", "*.PNG", "*.jpeg", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(get_all_files(path, ext))
    return imgs


def create_dir_if_not_exists(dir_path):
    lpath = Path(dir_path)
    if not lpath.exists():
        print("create dir:", dir_path)
        lpath.mkdir(parents=True)


def delete_if_exists(inpath):
    lpath = Path(inpath)
    if not lpath.exists():
        return

    if lpath.is_dir():
        shutil.rmtree(lpath)
    else:
        lpath.unlink()


def save_data_by_zip(file_path, data, data_type):
    file_path = Path(file_path)
    with ZipFile(str(file_path), compression=zipfile.ZIP_DEFLATED, mode="w") as zf:
        with zf.open(file_path.name, "w") as f:
            if data_type == "np":
                np.save(f, data)
            elif data_type == "pickle":
                pickle.dump(data, f)
            elif data_type == "torch":
                torch.save(data, f)
            else:
                raise NotImplementedError


def load_data_by_zip(file_path, data_type):
    file_path = Path(file_path)

    with ZipFile(str(file_path)) as zf:
        with zf.open(file_path.name) as f:
            buffer = io.BytesIO(f.read())
            if data_type == "np":
                data = np.load(buffer)
            elif data_type == "pickle":
                data = pickle.load(buffer)
            elif data_type == "torch":
                data = torch.load(f)
            else:
                raise NotImplementedError
    return data


def is_empty_dir(path):
    return len(os.listdir(path)) == 0
