"""
helper function
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
)

import numpy as np
import open3d as o3d
import open3d.core as o3c
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from datasets.tum import TUMDataset
from models.checker import Checker
from models.model import Model, split_data, to_numpy
from utils.convert import downsample, merge_points, transform
from utils.storage import CacheSE, DatasetCache


class Resources(TypedDict):
    dataset: TUMDataset
    checker: Checker


def init(clean: bool = True):
    """
    初始化项目：
    - 设置日志格式，增加输出到本地文件
    - 设置临时工作目录 run/
    - 添加临时缓存 run/cache.h5

    clean: 清理旧数据，否则继续上次的缓存
    """
    os.makedirs("run", exist_ok=True)

    logger.remove(0)
    logger.add(
        sys.stderr,
        format="<level>[{level}]</> {message} <light-yellow>({function}:{line})</> <light-blue>({elapsed})</>",
    )

    log_file = "run/log.txt"
    if clean and os.path.exists(log_file):
        os.remove(log_file)
    logger.add(log_file)

    cache_file = "run/cache.h5"
    if clean and os.path.exists(cache_file):
        os.remove(cache_file)
        cache = CacheSE("run/cache.h5", mode="w")
    else:
        cache = CacheSE("run/cache.h5", mode="a")

    # teddy related
    # 数据集
    dataset = TUMDataset(
        "/mnt/e/3d-datasets/TUM/6.ObjectReconstruction/rgbd_dataset_freiburg1_teddy",
        "fr1",
    )
    # 目标信息（mask points）
    teddy_cache = CacheSE("data/teddy1_dataset/teddy_target_cache.h5", mode="r")

    cache_file = "run/dataset_cache.h5"
    if clean and os.path.exists(cache_file):
        os.remove(cache_file)
        dataset_cache = DatasetCache(cache_file, mode="w")
    else:
        dataset_cache = DatasetCache(cache_file, mode="a")

    ## 规定了数据加载的方式
    def load_points(self, frame_id: int) -> np.ndarray:
        return self.target_cache[f"points/{frame_id}"]

    def load_depth(self, frame_id: int) -> np.ndarray:
        mask = self.target_cache[f"mask/{frame_id}"]
        depth = self.dataset.frames[frame_id].depth
        return (depth * mask).astype(np.uint16)

    def load_color(self, frame_id: int) -> np.ndarray:
        mask = self.target_cache[f"mask/{frame_id}"]
        color = self.dataset.frames[frame_id].color
        masked_color = color.copy()
        masked_color[mask == 0] = 0
        return masked_color

    teddy = dict(
        dataset=dataset,
        target_cache=teddy_cache,
        dataset_cache=dataset_cache,
        checker=Checker("run/rf_reg_cls_teddy.pkl"),
        model=Model(),
        load_points=load_points,
        load_depth=load_depth,
        load_color=load_color,
    )

    logger.info("init completed.")
    return teddy


@dataclass
class TeddyResources:
    dataset: TUMDataset
    target_cache: CacheSE
    dataset_cache: DatasetCache

    def load_points(self, frame_id: int) -> np.ndarray:
        return self.target_cache[f"points/{frame_id}"]

    def load_depth(self, frame_id: int) -> np.ndarray:
        mask = self.target_cache[f"mask/{frame_id}"]
        depth = self.dataset.frames[frame_id].depth
        return (depth * mask).astype(np.uint16)

    def load_color(self, frame_id: int) -> np.ndarray:
        mask = self.target_cache[f"mask/{frame_id}"]
        color = self.dataset.frames[frame_id].color
        masked_color = color.copy()
        masked_color[mask == 0] = 0
        return masked_color

    def load_pose(self, frame_id: int) -> np.ndarray:
        return self.dataset.frames[frame_id].pose
