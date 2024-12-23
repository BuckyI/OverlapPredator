from datetime import datetime
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

    def close(self):
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

        # 对于数据库的覆盖，双重验证一下
        if Path(path).exists() and mode == "w":
            check = input("你将要覆盖已有的缓存，是否继续？(y/n)")
            assert check.strip().lower() == "y", "abort"

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

    def exists(self, dataset: str, timestamp: float) -> bool:
        "快速判断数据集 dataset 中时间戳为 timestamp 的帧是否在数据库中有数据"
        return f"{dataset}/{timestamp}" in self.hdf5

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

    def load_frame_data(self, dataset: str, timestamp: float) -> DATA_TYPE:
        "加载指定数据集指定时间戳的帧数据，用于模型 pair_decode 操作"
        required_fields = [
            "lengths",
            "points",
            "neighbors",
            "pools",
            "upsamples",
            "features",
        ]
        return load_from_group(self.get_frame_group(dataset, timestamp), required_fields)


class RunCache(Storage):
    """
    存储程序运行过程中的中间数据

    [结构]
    step 0
        attrs 存储元数据
        datasets 存储大的矩阵数据
    step 1
    ...
    """

    def __init__(self, path: str, mode: str = "w", *, description: str = ""):
        """
        mode: make sure you know what you are doing
            - r: 读取已有的缓存
            - a: 向已有的缓存追加数据
            - w: 创建新的缓存，如果已经存在，会自动增加时间后缀避免重复
        """

        # 避免程序不同 run 覆盖缓存，自动添加时间后缀
        p = Path(path)
        if mode == "w" and p.exists():
            p = p.with_name(p.stem + datetime.now().strftime("%Y%m%d%H%M%S") + p.suffix)
            logger.warning(f"创建缓存至 {p.name}")
            path = p.as_posix()

        super().__init__(path, mode)
        # 初次创建时添加描述信息
        if self.hdf5.attrs.get("description") is None:
            self.hdf5.attrs["description"] = description

        self.index_ = None  # 用于搜索的索引，需要确保记录的数据结构一致

    @property
    def index(self) -> Dict[tuple, str]:
        "用于快速查找指定记录的索引"
        if self.index_ is None:
            logger.info("为快速查找，开始建立索引")
            self.index_ = {
                (g.attrs["dataset_id"], g.attrs["source_timestamp"], g.attrs["target_timestamp"]): k
                for k, g in self.hdf5.items()
            }
            logger.info("索引建立完毕")
        return self.index_

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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.hdf5.attrs.get('description')}"

    def update_step(self, key: Union[str, tuple], meta: dict, overwrite: bool = False):
        """
        更新指定 step 的 attrs
        key: step key
        meta: 需要更新的元数据
        overwrite: 当 meta 中的 key 在 attrs 中已存在时，是否覆盖已有数据，默认不覆盖。
        """
        if isinstance(key, tuple):
            key = self.index[key]
        step = self.hdf5.get(key)
        assert step is not None, f"step {key} does not exist"
        assert isinstance(step, h5py.Group), f"step {key} is not a group"
        for k, v in meta.items():
            if k not in step.attrs:
                step.attrs[k] = v
            elif overwrite:
                logger.warning("Overwriting existing attribute: {k} in step {key}.")
                step.attrs[k] = v
