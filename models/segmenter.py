"""
用于实例分割猪目标的 YOLO 模型
"""

from typing import List

import cv2
import numpy as np
import open3d as o3d
import torch
from loguru import logger
from ultralytics import YOLO

from utils.convert import compute_vertex


def extract_target_points(depth: np.ndarray, mask: np.ndarray, K: np.ndarray):
    """
    depth: 深度图
    mask: 预测的目标的 mask
    K: 相机内参
    return: 预处理过后的目标点云，经过 0.01 分辨率降采样和去除统计外点。
    由于 YOLO 处理图像会修改尺寸，mask 可以和 depth 的 shape 不一样
    """
    depth = depth.copy()  # 防止修改原数据
    if depth.shape != mask.shape:
        h, w = depth.shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    depth[mask == 0] = 0
    points = compute_vertex(depth, K)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    # pcd = pcd.remove_duplicated_points()
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
    pcd = pcd.select_by_index(ind)
    return np.asarray(pcd.points)


class Segmenter:

    def __init__(self, model_path: str = "weights/pig_segment.pt") -> None:
        self.model = YOLO(model_path)

        self.track_id = None  # 跟踪目标的 id
        self.result = None  # 保留最近一次的预测结果

    def segment(self, image: np.ndarray, *, rgb=True) -> List[np.ndarray]:
        if rgb:
            image = image[:, :, ::-1]  # 转换为 bgr 顺序

        result = self.model(image)[0]
        self.result = result
        if result.masks is None:
            return []

        masks = []
        h, w = result.masks.orig_shape
        for m in result.masks.data:
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(m)
        return masks

    def track(self, image: np.ndarray, *, rgb=True):
        """
        时间连续地跟踪单个物体，可以手动设置 self.track_id = None 以重新开始跟踪
        rgb: image 是否为 rgb 顺序，否则为 bgr 顺序
        return:
            None: 没有跟踪到物体
            mask: ndarray 跟踪到的物体的 mask
        """
        if rgb:
            image = image[:, :, ::-1]  # 转换为 bgr 顺序

        result = self.model.track(image, persist=True)[0]  # 只有一帧，所以取第0个
        self.result = result
        assert result.boxes is not None
        if not result.boxes.is_track:
            logger.info("没有检测到物体")
            return None

        assert result.boxes.id is not None
        assert result.masks is not None
        _id = self.track_id
        if _id is None:
            _id = self.track_id = int(result.boxes.id[0])  # 正常情况下应该为 1
            logger.info(f"没有正在跟踪的物体，开始跟踪目标 {_id}")

        pos = np.where(result.boxes.id == _id)[0]
        # 当检测出目标时，应为包含 1 个元素的向量
        if pos.size == 0:
            logger.error(f"没有检测到目标 {_id}")
            return None

        logger.debug(f"检测到目标 {_id}")
        mask = result.masks.data[pos[0]]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # YOLO 会输出 480, 640 的 mask，需要 resize 到和原图一样大
        h, w, _ = image.shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask
