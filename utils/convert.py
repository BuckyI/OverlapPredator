from typing import List, Union

import numpy as np
import small_gicp
import torch
from scipy.spatial.transform import Rotation as R


def to_numpy(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    elif isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    raise TypeError(f"not support type {type(t)}")


def euler2matrix(pitch, yaw, roll, x, y, z):
    trans = np.eye(4)
    trans[:3, 3] = [x, y, z]
    r = R.from_euler("xyz", [pitch, yaw, roll], degrees=True)
    trans[:3, :3] = r.as_matrix()
    return trans


def matrix2euler(trans):
    r = R.from_matrix(trans[:3, :3]).as_euler("xyz", degrees=True)
    t = trans[:3, 3]
    return np.array((r, t)).flatten()


def transform(source: np.ndarray, trans: np.ndarray):
    """
    对点云进行坐标变换
    source: pcd Nx3
    trans: 4x4 transform matrix from source to target
    return: transformed pcd Nx3
    """
    source_homo = np.concatenate((source, np.ones((source.shape[0], 1))), axis=1)
    return (source_homo @ trans.transpose())[..., :3]  # N, 3


def downsample(points: np.ndarray, resolution: float) -> np.ndarray:
    "对点云进行网格下采样"
    return small_gicp.voxelgrid_sampling(points, downsampling_resolution=resolution).points()[:, :3]


def merge_points(points_list: List[np.ndarray]) -> np.ndarray:
    "合并多个点云"
    return np.concatenate(points_list, axis=0)


def compute_vertex(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    将深度图转化为点云坐标
    depth: HxW
    K: 3x3
    return: HxWx3
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    X, Y = np.meshgrid(np.arange(0, W), np.arange(0, H))  # [H, W]
    vertex = np.stack([(X - cx) / fx, (Y - cy) / fy, np.ones_like(X)], -1) * depth[..., None]  # [H, W, 3]
    return vertex
