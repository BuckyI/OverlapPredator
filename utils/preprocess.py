import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


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
