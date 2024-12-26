"""
Azure Kinect Dataset
"""

from pathlib import Path
from typing import List, NamedTuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from utils.convert import downsample
from utils.storage import DatasetCache


class Frame(NamedTuple):
    timestamp: int  # in microsecond (1e-6 second)
    depth: np.ndarray
    color: np.ndarray
    pcd_array: np.ndarray
    type: str = ""


class KinectDataset:
    """
    读取 HDF5 缓存的 Kinect 数据集
    """

    def __init__(self, cache_path: str):
        assert Path(cache_path).is_file()
        self.cache = DatasetCache(cache_path, "r")
        self.meta = attrs = self.cache.hdf5.attrs
        self.dataset_id = attrs["dataset_id"]
        self.depth_scale = attrs["depth_scale"]
        self.width = attrs["width"]
        self.height = attrs["height"]
        self.K = attrs["K"]

        hdf5 = self.cache.hdf5.get(self.dataset_id)
        assert isinstance(hdf5, h5py.Group)
        self.hdf5 = hdf5
        self.timestamps = list(map(int, hdf5.keys()))

    def __getitem__(self, idx: Union[int, str]) -> Frame:
        if isinstance(idx, str):
            timestamp = int(idx)
        elif isinstance(idx, int):
            if idx < 0 or idx >= len(self.timestamps):
                raise IndexError(f"Index {idx} out of range")
            timestamp = self.timestamps[idx]
        else:
            raise TypeError(f"Index must be int or str, not {type(idx)}")

        g = self.hdf5.get(str(timestamp))
        assert isinstance(g, h5py.Group)
        type_ = g.attrs.get("type", "")
        depth = np.asarray(g["frame_depth"])
        color = np.asarray(g["frame_color"])
        points = np.asarray(g["frame_points"]) / self.depth_scale
        points = downsample(points, 0.01)

        return Frame(timestamp, depth, color, points, type_)

    def __iter__(self):
        for i in range(len(self.timestamps)):
            yield self[i]

    def __len__(self):
        return len(self.timestamps)
