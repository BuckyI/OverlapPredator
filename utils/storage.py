from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
from loguru import logger

from models.model import merge_data

h5py.get_config().track_order = True  # 保证读取顺序按照插入顺序
DATA_TYPE = Dict[str, Union[np.ndarray, List[np.ndarray]]]


def peek_structure(f):
    """
    get structure of whole dataset
    if key is int or float, consider it as a list, only show first 2
    >>> f = h5py.File('dataset.h5', 'a')
    >>> peek_structure(f)
    """

    def _peek(t):
        if isinstance(t, h5py.Dataset):
            return t.shape
        elif isinstance(t, h5py.Group):
            if all(k.replace(".", "").isdigit() for k in t.keys()):  # list, keys are numbers (int, float)
                return [{kk: _peek(vv)} for _, (kk, vv) in zip(range(2), t.items())]
            return {k: _peek(v) for k, v in t.items()}
        else:
            raise Exception("Unknown type")

    return {k: _peek(v) for k, v in f.items()}


class Storage:
    def __init__(self, path: str, mode="r"):
        """
        path: h5 file path
        mode: 'r' for read, 'w' for write, 'a' for Read/write if exists, create otherwise
        """
        # 检验
        assert path.endswith(".h5"), "not a valid path"
        p = Path(path)
        if p.exists() and mode == "w":
            raise FileExistsError(f"storage in {path} already exists")
        if not p.exists() and mode == "r":
            raise FileNotFoundError(f"storage in {path} not found")
        if p.exists() and mode == "a":
            logger.warning(f"storage in {path} found, will append data to it.")

        h5py.get_config().track_order = True  # 保证读取顺序按照插入顺序
        self.mode = mode
        self.path = path
        self.hdf5 = h5py.File(path, mode)

    def get_group(self, name: str) -> h5py.Group:
        "获取 name 指定的 group, 如果不存在则创建一个新的"
        group = self.hdf5.get(name)
        if group is None:
            logger.info("creating empty group: {name}", name=name)
            group = self.hdf5.create_group(name)
        assert isinstance(group, h5py.Group), f"get {name} but it's not a group"
        return group

    def __del__(self):
        self.hdf5.close()


def save_to_group(group: h5py.Group, data: DATA_TYPE):
    """保存 data 到 group
    value type of data should be:
        - numpy.ndarray
        - list[numpy.ndarray]
    key 不可以和已有的 key 冲突
    """
    assert all(k not in group.keys() for k in data.keys()), "key conflict"

    def save_list(group, name, data):
        list_group = group.create_group(name)
        # 添加 attrs 以方便指示读取方法 ！！！！！
        list_group.attrs["_type"] = "list_group"
        for i, d in enumerate(data):
            group.create_dataset(f"{name}/{i}", data=d)

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            group.create_dataset(k, data=v)
        elif isinstance(v, list):
            assert all(isinstance(d, np.ndarray) for d in v), "unexpected data type"
            save_list(group, k, v)
        else:
            raise ValueError(f"unexpected data type {type(v)}")


def load_from_group(group: h5py.Group, keys: Optional[List[str]] = None, recursive=False) -> DATA_TYPE:
    """
    读取指定的数据
    group: h5py.Group | h5py.File, 需要加载数据的来源
    keys: list[str], 需要读取的 key，如果不指定则读取 group 内的所有数据
    recursive: bool, 是否递归读取 group 内嵌套的数据，
        默认为 False，即只读取第一层的 dataset 类型；
        如果为 True，则递归读取 group 下嵌套的所有数据，使用时需要注意内存使用！
    """
    if keys is None:
        keys = list(group.keys())
    else:
        assert all(k in group.keys() for k in keys), f"missing key: {keys}"

    data = {}
    for k in keys:
        v = group[k]
        if isinstance(v, h5py.Dataset):
            data[k] = np.array(v)
        elif isinstance(v, h5py.Group):
            if v.attrs.get("_type") == "list_group":
                # 针对 list group 作特殊适配
                # 注，这里返回列表元素的顺序应该和插入顺序一致
                # 为保障这一点，需要确保 h5py.get_config().track_order = True
                data[k] = [np.array(d) for d in v.values()]
            elif recursive:
                data[k] = load_from_group(v, recursive=recursive)
        else:
            raise ValueError(f"unexpected data type {type(v)}")
    return data


class DatasetCache(Storage):
    """
    存储数据集图像的预处理数据

    [结构]
    dataset_name:
        timestamp:
            各字段
    """

    # 模型推理所需要的数据字段
    required_fields = {
        "lengths",
        "points",
        "neighbors",
        "pools",
        "upsamples",
        "features",
    }

    def __init__(self, path: str, mode="r"):
        super().__init__(path, mode)
        logger.info(f"{self.__class__.__name__} loaded")

    @property
    def datasets(self) -> List[str]:
        "所有存在缓存的数据集"
        return list(map(str, self.hdf5.keys()))

    def get_frame_group(self, name: str, timestamp: float) -> h5py.Group:
        """
        name: str, 数据集
        timestamp: float, 帧的 key
        """
        if name not in self.datasets:
            logger.warning(f"{name} not found in cache, will create a new group")

        return self.get_group(f"{name}/{timestamp}")

    def get_frame_timestamps(self, name: str) -> List[float]:
        "获取数据集 name 中的所有帧，注意帧的 key 为 float"
        if name not in self.datasets:
            logger.error(f"{name} not found in cache")
            return []
        group = self.hdf5.get(name)
        assert isinstance(group, h5py.Group), f"get {name} but it's not a group"
        return list(map(float, group.keys()))

    def save_frame_data(self, dataset: str, timestamp: float, data: DATA_TYPE):
        "将帧预处理的数据保存至本地数据库"
        logger.info(f"caching frame {timestamp}")
        group = self.get_frame_group(dataset, timestamp)
        save_to_group(group, data)

    def load_model_inputs(self, dataset: str, source_timestamp: float, target_timestamp: float):
        "加载缓存的指定数据，供后续合并以及模型推理所需"
        times = self.get_frame_timestamps(dataset)
        assert source_timestamp in times and target_timestamp in times, "missing frame"

        required_fields = [
            "lengths",
            "points",
            "neighbors",
            "pools",
            "upsamples",
            "features",
        ]
        source = load_from_group(self.get_frame_group(dataset, source_timestamp), required_fields)
        target = load_from_group(self.get_frame_group(dataset, target_timestamp), required_fields)

        return merge_data(source, target)


class StepCache(Storage):
    """
    存储程序运行过程中的中间数据

    [结构]
    step 0
        attrs 存储元数据
        datasets 存储大的矩阵数据
    step 1
    ...
    """

    def __init__(self, path: str, mode: str = "w"):
        if (mode == "w" or mode == "a") and Path(path).exists():
            raise Exception("程序运行缓存不应复用")
        super().__init__(path, mode)

    def log_step(self, meta: Optional[dict] = None, **data: dict):
        """
        在 'steps' group 中保存每次执行配准的数据
        meta 存储描述性的元数据等，其中的 value 不可以为 dict 类型
        data 存储数组类型数据，其中的 value 必须是可以被转化为 ndarray 的
        """
        step_group = self.hdf5
        next_i = max([int(i) for i in step_group.keys()] + [-1]) + 1
        group = step_group.create_group(f"{next_i}/")
        if meta is not None:
            for k, v in meta.items():
                group.attrs[k] = v

        for key, value in data.items():
            group.create_dataset(key, data=np.array(value))

        step_group.file.flush()
