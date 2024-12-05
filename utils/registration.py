from typing import Iterable, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import small_gicp


class RegistrationResult(NamedTuple):
    source: np.ndarray
    target: np.ndarray
    T_target_source: np.ndarray


def GICP_registration(
    source: np.ndarray,
    target: np.ndarray,
    init_T: Optional[np.ndarray] = None,
    resolution: float = 0.02,
):
    """
    执行单次 GICP 配准
    inputs:
        source: np.ndarray (N, 3)
        target: np.ndarray (N, 3)
        init_T: np.ndarray (4, 4) 初始位姿变换矩阵，默认为单位矩阵
        resolution: 配准的降采样分辨率，默认为 (0.25, 0.1, 0.02)，可以传入单个 float 执行单次配准。
    return:
        transformation: np.ndarray (4, 4)
        registration_result: small_gicp.RegistrationResult
    """
    init_T = np.eye(4) if init_T is None else init_T
    result = small_gicp.align(
        target,
        source,
        init_T_target_source=init_T,
        registration_type="GICP",
        downsampling_resolution=resolution,
        max_correspondence_distance=2 * resolution,
        num_threads=2,
    )
    return result.T_target_source, result


def GICP_registrations(source: np.ndarray, target: np.ndarray, init_T: Optional[np.ndarray] = None):
    """以 (0.25, 0.1, 0.02) 的分辨率执行分层 GICP 配准，稳定性更高"""
    T = init_T or np.eye(4)
    for r in (0.25, 0.1, 0.02):
        T, result = GICP_registration(source, target, init_T=T, resolution=r)
    return T, result


def GICP_registration_with_evaluation(
    source: np.ndarray,
    target: np.ndarray,
    init_T: np.ndarray = np.eye(4),
    resolution: float = 0.02,
):
    """
    执行 GICP 配准，同时计算配准相关的评价指标
    [hang 先不手动实现评价指标的计算了]
    """
    raise NotImplementedError("还未完工")
    target, target_tree = small_gicp.preprocess_points(target, downsampling_resolution=resolution)
    source, source_tree = small_gicp.preprocess_points(source, downsampling_resolution=resolution)
    # `target` and `source` are small_gicp.PointCloud with the following methods
    # target.size()           # Number of points
    # target.points()         # Nx4 numpy array   [x, y, z, 1] x N
    # target.normals()        # Nx4 numpy array   [nx, ny, nz, 0] x N
    # target.covs()           # Array of 4x4 covariance matrices

    result = small_gicp.align(
        target,
        source,
        target_tree,
        init_T_target_source=init_T,
        registration_type="GICP",
        max_correspondence_distance=2 * resolution,
        num_threads=2,
    )

    return {
        "T": result.T_target_source,
        "source": source,
        "target": target,
        "source_tree": source_tree,
        "target_tree": target_tree,
        "result": result,
    }
