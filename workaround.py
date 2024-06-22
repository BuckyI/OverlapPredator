import pickle
from typing import Protocol

import numpy as np
import open3d as o3d


class Transformer(Protocol):

    @staticmethod
    def encode(data) -> dict: ...
    @staticmethod
    def decode(data): ...


class PointCloud(Transformer):
    @staticmethod
    def encode(data):
        return {
            "np_points": np.asarray(data.points),
            "np_colors": np.asarray(data.colors),
            "np_normals": np.asarray(data.normals),
        }

    @staticmethod
    def decode(data):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data["np_points"])
        pcd.colors = o3d.utility.Vector3dVector(data["np_colors"])
        pcd.normals = o3d.utility.Vector3dVector(data["np_normals"])
        return pcd


registered = {
    o3d.geometry.PointCloud: PointCloud,
}


def save(path="data.pickle", **kwargs):
    data = {}
    for k, v in kwargs.items():
        _type = type(v)
        if _type in registered:
            data[k] = (_type, registered[_type].encode(v))
        else:  # general object
            data[k] = (object, v)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load(path="data.pickle"):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    data = {}
    for _key, (_type, _data) in raw.items():
        if _type in registered:
            data[_key] = registered[_type].decode(_data)
        else:
            data[_key] = _data

    return data
