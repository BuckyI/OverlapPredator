import numpy as np
import open3d as o3d


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
