import yaml
from src.params import Config
from pathlib import Path
from src.paths import MODULE_DIR
from contextlib import contextmanager
from loguru import logger
from functools import wraps
import time


def read_config(path: Path = MODULE_DIR.parent / "params.yaml") -> Config:
    with open(path, "r") as f:
        conf = yaml.safe_load(f)
    return Config(**conf)


@contextmanager
def log_block(message: str, depth=2):
    logger_ = logger.opt(depth=depth)
    logger_.info(message)
    yield
    logger_.info(f"(DONE) {message}")


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
