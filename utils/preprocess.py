import cv2
import numpy as np
import open3d as o3d
import small_gicp
from sklearn.cluster import DBSCAN

from utils.convert import compute_vertex, downsample, resize_image_like


def cluster_dbscan(points: np.ndarray, eps=0.02, min_points=10):
    """
    eps: 聚类距离阈值
    min_points: 最小点数阈值
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    )
    return labels


def dbscan_extract(points: np.ndarray, eps=0.02, min_points=10, percentage=0.9):
    """
    return : index, points[index]
    """
    clustering = DBSCAN(eps=eps, min_samples=min_points)
    labels = clustering.fit_predict(points)

    # labels >= 0 -> remove noise points
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    cluster_counts = dict(zip(unique_labels, counts))
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: -x[1])

    selected_labels = []
    cumulative_sum = 0
    threshold = percentage * len(labels)  # 假定至少 percentage 的点属于目标物体
    for label, count in sorted_clusters:
        selected_labels.append(label)
        cumulative_sum += count
        if cumulative_sum >= threshold:
            break  # 达到阈值后停止

    mask = np.isin(labels, selected_labels)
    index = np.where(mask)[0]
    return index, points[index]


def refine_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    eps=0.02,
    min_points=10,
    percentage=0.9,
):
    """
    通过深度信息优化目标掩码，并提取精简的点云数据

    Args:
        mask (np.ndarray): 初始二值掩码（0/1），形状为(H, W)
        depth (np.ndarray): 深度图，形状为(H, W)
        K (np.ndarray): 相机内参矩阵，形状为(3, 3)
        eps (float, optional): DBSCAN聚类邻域半径（米），默认0.02
        min_points (int, optional): DBSCAN最小样本数，默认10
        percentage (float, optional): 保留点云的比例阈值，默认0.9，即假设噪声点不超过 10%

    Returns:
        np.ndarray: 优化后的二值掩码，形状与输入深度图一致
        np.ndarray: 精简后的点云坐标，形状为(N, 3)
    """
    if mask.shape != depth.shape:
        mask = resize_image_like(mask.astype(np.int8), depth)
    assert mask.shape == depth.shape

    # 提取原始目标
    masked_depth = depth * mask
    points = compute_vertex(masked_depth, K).reshape(-1, 3)  # width * height, 3

    # 稀疏点聚类
    sparse_points = downsample(points, 0.02)  # 降采样
    sparse_labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(
        sparse_points
    )  # 聚类

    # 恢复稠密点云标签
    tree = small_gicp.KdTree(sparse_points)  # 降采样的点云
    indices, dist = tree.batch_nearest_neighbor_search(
        points
    )  # 深度图直接转换得到的点云
    indices = np.array(indices, dtype=np.int32)
    dist = np.array(dist, dtype=np.float32)
    labels = sparse_labels[indices]  # 原始点云的 label
    labels[dist > 0.05] = -1  # 距离大于 0.05 的对应点认为不存在对应点

    # 像素点距离目标边界的距离
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    distance = distance.flatten()

    # 根据距离确定 label 保留优先级，距离越大说明越靠近物体中心，优先级越高
    label_distance = {}
    for _label in np.unique(labels):
        label_distance[_label] = np.max(distance[labels == _label])  # 取最大值
    sorted_labels = sorted(label_distance.items(), key=lambda x: -x[1])  # 从大到小

    # 筛选
    kept_labels = []
    cumulative_sum = 0
    threshold = percentage * len(labels[labels != -1])  # 去除无效点后的点数目
    for _label, _ in sorted_labels:
        if _label == -1:  # 无效标签，即离群点 / 非目标点
            continue

        kept_labels.append(_label)
        cumulative_sum += len(labels[labels == _label])
        if cumulative_sum >= threshold:
            break  # 达到阈值后停止

    kept_index = np.isin(labels, kept_labels)
    fixed_mask = np.bitwise_and(mask, kept_index.reshape(mask.shape))
    cleaned_pcd = points[kept_index]
    return fixed_mask, cleaned_pcd
