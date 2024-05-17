import logging
from pathlib import Path
from rich.logging import RichHandler


def init_logger(filename, verbosity=1, name=None, to_stdout=True):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter("[%(asctime)s]%(message)s", datefmt="%Y/%m/%d-%H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # 禁止log向上传播
    # https://docs.python.org/3/library/logging.html#logging.Logger.propagate
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()
    # 写入文件
    fh = logging.FileHandler(filename, "a", encoding="utf-8", delay=True)
    fh.setFormatter(formatter)
    fh.setLevel(level_dict[verbosity])
    logger.addHandler(fh)
    # 终端显示
    if to_stdout:
        # sh = logging.StreamHandler(sys.stdout)
        # sh.setFormatter(formatter)
        # sh.setLevel(level_dict[verbosity])
        # logger.addHandler(sh)
        rh = RichHandler()
        logger.addHandler(rh)

    return logger


def init_global_logger(logfile=None) -> logging.Logger:
    if logfile is None:
        logfile = Path("/tmp/3dgs_pipeline.txt")

    logfile.parent.mkdir(parents=True, exist_ok=True)

    print("The global logger is not initialized. \nInitialize the global logger.")
    print("Global log path:", logfile)
    global_logger = init_logger(logfile)
    return global_logger
