from typing import List, Union

import cv2
import numpy as np
import small_gicp
import torch
from loguru import logger
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


def average_poses(poses: List[np.ndarray]) -> np.ndarray:
    "计算多个变换矩阵的平均值"
    rot = R.from_matrix([p[:3, :3] for p in poses]).mean().as_matrix()
    trans = np.mean([p[:3, 3] for p in poses], axis=0)

    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


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
    return small_gicp.voxelgrid_sampling(
        points, downsampling_resolution=resolution
    ).points()[:, :3]


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
    vertex = (
        np.stack([(X - cx) / fx, (Y - cy) / fy, np.ones_like(X)], -1) * depth[..., None]
    )  # [H, W, 3]
    return vertex


def resize_image_like(
    image: np.ndarray, target: np.ndarray, interpolation: int = cv2.INTER_NEAREST
) -> np.ndarray:
    """
    将 mask resize 到与 target 相同大小
    image: H1xW1 深度图或二进制掩膜
    target: H2xW2 或 H2xW2xC 要对齐大小的目标
    interpolation: cv2 interpolation method
        cv2.INTER_NEAREST 0：最近邻插值，适用于二进制掩膜
        cv2.INTER_LINEAR 1：双线性插值，适用于深度图
        cv2.INTER_CUBIC 2：双三次插值，适用于深度图
    return: resized image H2xW1
    """
    if target.ndim == 2:  # depth
        h, w = target.shape
    elif target.ndim == 3:  # color
        h, w, _ = target.shape
    else:
        raise ValueError(f"target shape {target.shape} is unexpected")

    # warning
    if interpolation != cv2.INTER_NEAREST and np.issubdtype(image.dtype, np.integer):
        logger.warning(
            "mask is integer type, but not using INTER_NEAREST,"
            "hope you know what you are doing"
        )

    resized = cv2.resize(image, (w, h), interpolation=interpolation)
    return resized


def rgb2luminance(colors: np.ndarray) -> np.ndarray:
    return 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
