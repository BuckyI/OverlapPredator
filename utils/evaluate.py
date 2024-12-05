import numpy as np
import open3d as o3d
import small_gicp
import torch

from lib.benchmark_utils import to_o3d_pcd

from .convert import transform


def get_trans_rot(t):
    "get translation and rotation from transformation matrix"
    trans = np.linalg.norm(t[:3, 3])
    rot = np.arccos((np.trace(t[:3, :3]) - 1) / 2)
    return trans, rot


def pose_difference(t1: np.ndarray, t2: np.ndarray):
    """
    计算两个位姿之间的差异
    计算方法为旋转量和平移量的模长（本质上也为李代数表示的位姿差异的模长）
    """
    return np.linalg.norm(get_trans_rot(np.linalg.inv(t1) @ t2))


def chamfer_distance(a: np.ndarray, b: np.ndarray):
    "计算两个点云之间的 chamfer 距离"
    assert a.shape[0] and b.shape[0], "点数量不能为0"

    tree1 = small_gicp.KdTree(a)
    tree2 = small_gicp.KdTree(b)
    _, dist1 = tree1.batch_nearest_neighbor_search(b)
    _, dist2 = tree2.batch_nearest_neighbor_search(a)
    return float(np.mean(dist1) + np.mean(dist2))


def chamfer_distance_feat(sp, tp, sf, tf, trans: np.ndarray = np.eye(4)) -> float:
    """
    sp: source points
    tp: target points
    sf: source features
    tf: target features
    trans: transformation matrix from sp to tp
    return float
    """
    sp = transform(sp, trans)  # 位姿变换对齐

    tree1 = small_gicp.KdTree(sp)
    tree2 = small_gicp.KdTree(tp)
    idx1, dist1 = tree1.batch_nearest_neighbor_search(tp)
    idx2, dist2 = tree2.batch_nearest_neighbor_search(sp)

    return np.linalg.norm((tf - sf[idx1]), axis=1).mean() + np.linalg.norm((sf - tf[idx2]), axis=1).mean()


@torch.jit.script
def get_similarity(feat1: torch.Tensor, feat2: torch.Tensor):
    return (torch.dot(feat1, feat2) / feat1.norm() / feat2.norm()).item()


def evaluate_registration(
    sp: np.ndarray,
    tp: np.ndarray,
    trans: np.ndarray = np.eye(4),
    resolution: float = 0.02,
):
    """
    评估配准结果
    sp: source points
    tp: target points
    trans: transformation matrix from sp to tp
    resolution: voxel size, default 0.02 （评估前先对点云进行降采样）
    """
    _sp, _tp = to_o3d_pcd(sp), to_o3d_pcd(tp)
    _sp, _tp = _sp.voxel_down_sample(resolution), _tp.voxel_down_sample(resolution)
    result = o3d.pipelines.registration.evaluate_registration(
        source=_sp,
        target=_tp,
        max_correspondence_distance=2 * resolution,  # 固定为这个比例
        transformation=trans,
    )
    return result
