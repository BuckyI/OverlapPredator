from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import small_gicp
import torch
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from lib.benchmark_utils import to_o3d_pcd

from .convert import transform


def get_trans_rot(t):
    "get translation and rotation from transformation matrix"
    trans = np.linalg.norm(t[:3, 3])
    rot = np.arccos(np.clip((np.trace(t[:3, :3]) - 1) / 2, -1, 1))
    return trans, rot


def pose_difference(t1: np.ndarray, t2: np.ndarray):
    """
    计算两个位姿之间的差异
    计算方法为旋转量和平移量的模长（本质上也为李代数表示的位姿差异的模长）
    """
    return np.linalg.norm(get_trans_rot(np.linalg.inv(t1) @ t2))


def pose_difference2(t1: np.ndarray, t2: np.ndarray):
    """
    计算位姿差异，但是旋转量 * 2 以更加重视旋转差异。
    参考 Bundle Fusion (Dai 2018) 的位姿差异评估方法
    """
    t = np.linalg.inv(t1) @ t2
    trans = np.linalg.norm(t[:3, 3])
    rot = np.arccos(np.clip((np.trace(t[:3, :3]) - 1) / 2, -1, 1))
    return np.sqrt(trans**2 + 4 * rot**2)


def absolute_trajectory_error(
    gt: List[np.ndarray],
    pred: List[np.ndarray],
    *,
    align: int = 0,
):
    """
    gt: ground truth trajectory
    pred: predicted trajectory
    align: align gt and pred, default 0
        0: align first frame
        1: use umeyama_alignment
    """

    assert len(gt) == len(pred), "not the same length"

    if align == 0:
        trans = gt[0] @ np.linalg.inv(pred[0])
    elif align == 1:
        # umeyama_alignment
        X = np.array([i[:3, 3] for i in pred])
        Y = np.array([i[:3, 3] for i in gt])
        centroid_X = np.mean(X, axis=0)
        centroid_Y = np.mean(Y, axis=0)
        X_centered = X - centroid_X
        Y_centered = Y - centroid_Y
        H = X_centered.T @ Y_centered
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T  # Vt是V的转置，因此 V = Vt.T
        R = V @ U.T
        # 确保旋转矩阵是正交且行列式为+1（防止反射）
        if np.linalg.det(R) < 0:
            V[:, -1] *= -1
            R = V @ U.T
        # 计算缩放因子s
        # numerator = np.trace(Y_centered.T @ (X_centered @ R.T))
        # denominator = np.trace(X_centered.T @ X_centered)
        # s = numerator / denominator
        s = 1.0  # 不缩放

        trans = np.eye(4)
        trans[:3, :3] = s * R
        trans[:3, 3] = centroid_Y - s * R @ centroid_X
    else:
        raise ValueError()

    aligned_pred = [trans @ pose for pose in pred]
    error = [pose_difference(gt[i], aligned_pred[i]) for i in range(len(gt))]
    return error


def chamfer_distance(a: np.ndarray, b: np.ndarray, trans: np.ndarray = np.eye(4)):
    """
    计算两个点云之间的 chamfer 距离
    a: source points
    b: target points
    trans: transformation matrix from a to b
    return float
    """
    assert a.shape[0] and b.shape[0], "点数量不能为0"
    a = transform(a, trans)  # 位姿变换对齐

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

    return (
        np.linalg.norm((tf - sf[idx1]), axis=1).mean()
        + np.linalg.norm((sf - tf[idx2]), axis=1).mean()
    )


@torch.jit.script
def get_similarity(feat1: torch.Tensor, feat2: torch.Tensor):
    return (torch.dot(feat1, feat2) / feat1.norm() / feat2.norm()).item()


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


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
    return {
        "source_point_size": len(_sp.points),
        "target_point_size": len(_tp.points),
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "inlier_num": len(result.correspondence_set),
    }


def check_data_consistency(data1: dict, data2: dict, verbose: bool = True):
    """检验两个 dict 内的数据是否一致"""

    def _all_close(a, b, name=""):
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and not torch.allclose(i, j)
        ):
            e = torch.abs(i - j).mean()
            msg = f"{name}[Tensor] not equal, mean error: {e}"
            return msg
        if (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and not np.allclose(i, j)
        ):
            e = np.abs(i - j).mean()
            msg = f"{name}[ndarray] not equal, mean error: {e}"
            return msg
        return ""

    assert data1.keys() == data2.keys(), "keys not equal"
    errors = []
    for k in data1.keys():
        if isinstance(data1[k], list):
            for idx, (i, j) in enumerate(zip(data1[k], data2[k])):
                msg = _all_close(i, j, f"{k}[{idx=}]")
                if msg:
                    errors.append(msg)
        elif isinstance(data1[k], torch.Tensor) or isinstance(data1[k], np.ndarray):
            msg = _all_close(data1[k], data2[k], k)
            if msg:
                errors.append(msg)
        else:
            errors.append(f"unknown type {type(data1[k])}")
    if verbose:
        print("\n".join(errors) or "All equal.")
    return errors


def show_pr_curve(y_true, probas_pred):
    "评估分类任务性能"
    precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
    auc_score = auc(recall, precision)
    print("auc score:", auc_score)

    plt.figure()
    plt.plot(recall, precision, color="darkorange", label="Precision-Recall curve")
    # plt.plot([0, 1], [1, 0], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()


def show_roc_curve(y_true, probas_pred):
    "评估分类任务性能"
    fpr, tpr, thresholds = roc_curve(y_true, probas_pred)
    auc_score = auc(fpr, tpr)
    print("auc score:", auc_score)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc_score:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def evaluate_pose_graph(pose_graph, dataset, frame_ids: List[int]):
    """
    评估位姿图
    pose_graph: o3d.pipelines.registration.PoseGraph
    dataset: dataset.frames contains all the frames
    frame_ids: 用于根据 node id 查询 dataset frame id，长度应该和 pose_graph.nodes 一致
    """
    edge_data = []
    for e in pose_graph.edges:
        sf = dataset.frames[frame_ids[e.source_node_id]]
        tf = dataset.frames[frame_ids[e.target_node_id]]
        gt_T = np.linalg.inv(tf.pose) @ sf.pose
        data = {
            "source_timestamp": sf.timestamp,
            "target_timestamp": tf.timestamp,
            "source_node_id": e.source_node_id,
            "target_node_id": e.target_node_id,
            "confidence": e.confidence,
            "uncertain": e.uncertain,
            "transformation": e.transformation,
            "gt_transformation": gt_T,
            "error": pose_difference2(e.transformation, gt_T),
            "is_loop": (abs(sf.timestamp - tf.timestamp) > 5)
            and (pose_difference2(tf.pose, sf.pose) < 1),
        }
        edge_data.append(data)
    edge_data = pd.DataFrame(edge_data)

    node_data = []
    for i, n in enumerate(pose_graph.nodes):
        frame = dataset.frames[frame_ids[i]]
        data = {
            "timestamp": frame.timestamp,
            "error": pose_difference2(n.pose, frame.pose),
        }
        node_data.append(data)
    node_data = pd.DataFrame(node_data)

    eval_result = {}
    eval_result["outlier_count"] = (edge_data["error"] > 0.3).sum()
    eval_result["outlier_percent"] = (edge_data["error"] > 0.3).sum() / len(edge_data)
    eval_result["edge_error_mean"] = edge_data["error"].mean()
    eval_result["loop_edges_count"] = edge_data["uncertain"].sum()
    eval_result["node_error_mean"] = node_data["error"].mean()

    print(eval_result)

    eval_result["edge_data"] = edge_data
    eval_result["node_data"] = node_data
    return eval_result


def abosolute_pose_error(poses: List[np.ndarray], poses_gt: List[np.ndarray]):
    assert len(poses) == len(poses_gt)
    # 转换到同一个 gt 坐标系下
    T = poses_gt[0] @ np.linalg.inv(poses[0])
    poses = [T @ pose for pose in poses]
    return [pose_difference2(i, j) for i, j in zip(poses_gt, poses)]


def binary_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def disturb_transform(
    T: np.ndarray, trans_mag: float, rot_mag: float, n_samples: int = 1
):
    """
    给4x4位姿变换矩阵添加指定幅度的噪声

    Args:
        T (np.ndarray): 4x4位姿变换矩阵
        trans_mag (float): 平移噪声的幅度(m)
        rot_mag (float): 旋转噪声的幅度(degree)
        n_samples (int, optional): 生成噪声的样本数. Defaults to 1.

    Returns:
        disturbed_Ts (List[np.ndarray]): 带噪声的位姿变换矩阵列表
        errors (List[float]): 噪声位姿变换矩阵与原始位姿变换矩阵的差异
    """
    # 分解原始位姿矩阵
    R_original = T[:3, :3]
    t_original = T[:3, 3]

    # 生成平移噪声
    t_noises = np.random.uniform(-trans_mag, trans_mag, (n_samples, 3))
    # 生成旋转噪声
    # 随机生成旋转轴（单位向量）
    random_axis = np.random.randn(n_samples, 3)
    random_axis /= np.linalg.norm(random_axis, axis=1, keepdims=True)
    # 生成旋转角度
    angles = np.random.uniform(-rot_mag, rot_mag, n_samples)
    R_noises = [
        R.from_rotvec(np.deg2rad(angles[i]) * random_axis[i]).as_matrix()
        for i in range(n_samples)
    ]

    # 应用噪声
    disturbed_Ts = []
    errors = []
    for i in range(n_samples):
        disturbed_T = np.eye(4)
        disturbed_T[:3, :3] = R_original @ R_noises[i]  # 右乘局部坐标系扰动
        disturbed_T[:3, 3] = t_original + t_noises[i]
        error = pose_difference2(T, disturbed_T)
        disturbed_Ts.append(disturbed_T)
        errors.append(error)
    return disturbed_Ts, errors
