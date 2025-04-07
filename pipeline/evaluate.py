"""
评估 pipline 中获得数据的质量
"""

import numpy as np
import pandas as pd
import small_gicp

from utils.convert import transform


def evaluate_(data: dict):
    """
    [DEPRECATED]
    根据配准结果，计算 fitness, cd, cdf 三个指标
    data: model registration result
    >>> data = model.registration(source, target, debug=True)
    """
    source = data["source_raw"].cpu().numpy()  # N1, 3
    target = data["target_raw"].cpu().numpy()  # N2, 3
    source_feat = data["source_raw_feats"].cpu().numpy()  # N1, 32
    target_feat = data["target_raw_feats"].cpu().numpy()  # N2, 32
    trans = data["T"]  # 4, 4
    source_trans = transform(source, trans)  # N1, 3

    target_tree = small_gicp.KdTree(target)
    indices, dists = target_tree.batch_nearest_neighbor_search(source_trans, 1)
    indices, dists = np.array(indices), np.array(dists)

    max_dist = 0.05  # inlier threshold
    inliers = dists < max_dist  # N, 1 inliers

    fitness = inliers.sum() / len(source)
    cd = dists[inliers].mean()
    cdf = np.linalg.norm(
        source_feat[inliers] - target_feat[indices[inliers]], axis=1
    ).mean()
    return fitness, cd, cdf


def voting_evaluate(
    source: np.ndarray,
    target: np.ndarray,
    source_feat: np.ndarray,
    target_feat: np.ndarray,
    source_luminance: np.ndarray,
    target_luminance: np.ndarray,
    trans: np.ndarray,
):
    """
    source: N1, 3 source points
    target: N2, 3 target points
    source_feat: N1, 32 source features
    target_feat: N2, 32 target features
    source_luminance: N1, source luminance
    target_luminance: N2, target luminance
    trans: 4, 4 transformation matrix from source to target

    return
    space_dist: N1, euclidean distance between corresponding points
    feature_dist: N1, feature cosine similarity between corresponding points
    color_dist: N1, luminance difference between corresponding points
    """
    source_trans = transform(source, trans)  # N1, 3
    target_tree = small_gicp.KdTree(target)
    indices, dists = target_tree.batch_nearest_neighbor_search(source_trans, 1)
    indices, dists = np.asarray(indices), np.asarray(dists)

    # 计算评价指标
    space_dist = dists
    # 模型编码时，已经将特征归一化到模长为 1，所以这里直接计算向量内积即可
    feature_dist = np.sum(source_feat * target_feat[indices], axis=1)
    # feature_dist = np.linalg.norm(source_feat - target_feat[indices], axis=1)
    color_dist = np.abs(source_luminance - target_luminance[indices])
    return space_dist, feature_dist, color_dist


def calculate_luminance_gradient(points: np.ndarray, colors: np.ndarray):
    """
    计算彩色点云的亮度和亮度梯度
    """
    luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    tree = small_gicp.KdTree(points)
    indices, dists = tree.batch_knn_search(points, 6)  # 20
    indices, dists = np.array(indices), np.array(dists)
    neighbor_luminance = luminance[indices[:, 1:]]  # 跳过每个点的第一个邻居（它自己）

    # 方法 1
    gradient = np.linalg.norm(neighbor_luminance - luminance[:, np.newaxis], axis=1)

    # 方法 2
    # def f(x):  # gaussian function (x=0.1, f=0.1)
    #     return np.exp(-np.square(x) / 0.01 * np.log(10))

    # weight = f(dists[:, 1:])  # 跳过每个点的第一个邻居（它自己）
    # gradient = np.sum(
    #     np.abs(neighbor_luminance - luminance[:, np.newaxis]) * weight, axis=1
    # ) / np.sum(weight, axis=1)

    # 方法 3
    # gradients_3d = np.zeros_like(points)  # N, 3
    # X = points[indices]
    # y = luminance[indices]

    # for i in range(len(points)):
    #     # 拟合平面 y = a*x + b*y + c*z + d
    #     a = X[i]
    #     b = y[i]
    #     a = np.c_[a, np.ones(len(a))]  # 构造矩阵 [x, y, z, 1]
    #     coeff, _, _, _ = np.linalg.lstsq(a, b)

    #     # 梯度方向为平面法向量 (a, b, c)，并归一化
    #     grad = coeff[:3]
    #     grad_norm = grad / (np.linalg.norm(grad) + 1e-6)  # 防止除零
    #     gradients_3d[i] = grad_norm * np.std(b)  # 梯度强度与亮度变化相关

    return luminance, gradient
