from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import small_gicp
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

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


def pose_difference2(t1: np.ndarray, t2: np.ndarray):
    """
    计算位姿差异，但是旋转量 * 2 以更加重视旋转差异。
    参考 Bundle Fusion (Dai 2018) 的位姿差异评估方法
    """
    t = np.linalg.inv(t1) @ t2
    trans = np.linalg.norm(t[:3, 3])
    rot = np.arccos((np.trace(t[:3, :3]) - 1) / 2)
    return np.sqrt(trans**2 + 4 * rot**2)


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
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and not torch.allclose(i, j):
            e = torch.abs(i - j).mean()
            msg = f"{name}[Tensor] not equal, mean error: {e}"
            return msg
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and not np.allclose(i, j):
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
    plt.title(f"Precision-Recall Curve")
    plt.legend()
    plt.show()


def show_roc_curve(y_true, probas_pred):
    "评估分类任务性能"
    fpr, tpr, thresholds = roc_curve(y_true, probas_pred)
    auc_score = auc(fpr, tpr)
    print("auc score:", auc_score)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc_score:.2f})")
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
            "is_loop": (abs(sf.timestamp - tf.timestamp) > 5) and (pose_difference2(tf.pose, sf.pose) < 1),
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
