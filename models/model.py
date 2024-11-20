"""
用于加速和简化模型推理，封装的类
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger

from datasets.dataloader import batch_grid_subsampling_kpconv, batch_neighbors_kpconv


@dataclass
class Model:

    # settings from `configs/test/indoor.yaml`, removed unused fields
    first_subsampling_dl = 0.025
    conv_radius = 2.5
    deform_radius = 5.0
    # indoor architecture
    architecture = [
        "simple",
        "resnetb",
        "resnetb_strided",
        "resnetb",
        "resnetb",
        "resnetb_strided",
        "resnetb",
        "resnetb",
        "resnetb_strided",
        "resnetb",
        "resnetb",
        "nearest_upsample",
        "unary",
        "nearest_upsample",
        "unary",
        "nearest_upsample",
        "last_unary",
    ]
    # ref: original model settings
    neighborhood_limits = (38, 36, 36, 38)  # (Encoder)第 i 层最大邻域数

    model_path: str = "weights/traced_model.pth"
    min_point_size: int = 100  # 点云最小点数，小于该值则不进行预处理

    def __post_init__(self):
        assert Path(self.model_path).is_file(), f"{self.model_path} does not exists"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = torch.jit.load(self.model_path).to(self.device)
        logger.trace(f"model loaded from {self.model_path} to {self.device}")

    @torch.inference_mode()
    def preporcess(self, raw_points: np.ndarray) -> Tuple[bool, dict]:
        """
        预处理点云对，生成模型推理所需的数据结构 (original: collate_fn_descriptor)

        1. 点云预处理，获取 points, neighbors, pools, upsamples, stack_lengths
        2. 使用模型编码，获取 features

        由于使用模型推理，返回数据位于设定的设备（GPU）
        return: result, data
        因特定原因失败时，result 为 False
        - 降采样后点数目过少
        """
        if len(raw_points) < self.min_point_size:
            logger.error(f"点云数太少 {len(raw_points)}，这个应该在其他地方处理的")
            return False, {}

        neighborhood_limits = self.neighborhood_limits

        # 受限于调用的函数，这里需要：
        # 生成假的 empty 点云，与原始点云共同处理
        # 转换为 Tensor
        empty = np.empty((2, 3), dtype=np.float64)
        points = torch.tensor(np.concatenate([raw_points, empty], axis=0))
        lengths = torch.tensor([len(raw_points), len(empty)], dtype=torch.int32)

        # Starting radius of convolutions
        r_normal = self.first_subsampling_dl * self.conv_radius

        # Starting layer
        layer_blocks = []
        layer = 0

        # Lists of inputs
        input_points = []  # 第 i 层的 pcd
        input_neighbors = []  # 第 i 层查询 pcd 邻居的索引
        input_pools = []  # (Encoder)第 i 层对 pcd 进行 pooling / subsampling / 降采样的索引 (只有为 pool/strided 层才有效)
        input_upsamples = []  # (Decoder)第 i 层对 pcd 进行 upsampling / 上采样的索引
        input_batches_len = []

        for block_i, block in enumerate(self.architecture):
            # Stop when meeting a global pooling or upsampling
            if "global" in block or "upsample" in block:
                break

            # Get all blocks of the layer
            if not ("pool" in block or "strided" in block):
                layer_blocks += [block]
                if block_i < len(self.architecture) - 1 and not ("upsample" in self.architecture[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************

            if layer_blocks:  # NOTE: not ("pool" in block or "strided" in block)
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(["deformable" in blck for blck in layer_blocks[:-1]]):
                    r = r_normal * self.deform_radius / self.conv_radius
                else:
                    r = r_normal
                # NOTE: 计算了点的 neighbors
                conv_i = batch_neighbors_kpconv(
                    points,
                    points,
                    lengths,
                    lengths,
                    r,
                    neighborhood_limits[layer],
                )

            else:  # NOTE: ("pool" in block or "strided" in block)
                # This layer only perform pooling, no neighbors required
                conv_i = torch.zeros((0, 1), dtype=torch.int64)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if "pool" in block or "strided" in block:

                # New subsampling length
                dl = 2 * r_normal / self.conv_radius

                # Subsampled points
                # NOTE: 对 batched_points 进行降采样, 获得 pool_p 点, pool_b 长度
                pool_p, pool_b = batch_grid_subsampling_kpconv(points, lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if "deformable" in block:
                    r = r_normal * self.deform_radius / self.conv_radius
                else:
                    r = r_normal

                # Subsample indices
                # NOTE: 计算了将 batched_points 降采样到 pool_p 对应的 neighbors index
                pool_i = batch_neighbors_kpconv(pool_p, points, pool_b, lengths, r, neighborhood_limits[layer])

                # Upsample indices (with the radius of the next layer to keep wanted density)
                # NOTE: 计算了将 pool_p 上采样恢复为 batched_points 对应的 neighbors index
                up_i = batch_neighbors_kpconv(points, pool_p, lengths, pool_b, 2 * r, neighborhood_limits[layer])

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = torch.zeros((0, 1), dtype=torch.int64)
                pool_p = torch.zeros((0, 3), dtype=torch.float32)
                pool_b = torch.zeros((0,), dtype=torch.int64)
                up_i = torch.zeros((0, 1), dtype=torch.int64)

            # Updating input lists
            input_points += [points.float()]
            input_neighbors += [conv_i.long()]
            input_pools += [pool_i.long()]
            input_upsamples += [up_i.long()]
            input_batches_len += [lengths]

            # New points for next layer
            points = pool_p
            lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer += 1
            layer_blocks = []

        # 过滤掉数目太少的点
        if (pts := input_batches_len[-1][0]) < self.min_point_size:
            logger.warning(f"Too few points ({pts}) in the input cloud, skipping")
            return False, {}

        # 模型编码
        to_device = lambda ls: [item.to(self.device) for item in ls]
        data = {
            "points": to_device(input_points),
            "neighbors": to_device(input_neighbors),
            "pools": to_device(input_pools),
            "upsamples": to_device(input_upsamples),
            "stack_lengths": to_device(input_batches_len),
        }
        data["features"] = self.model.encode(data)

        point_size = torch.tensor([len(p) for p in data["points"]])

        # 预处理，将无效索引值替换为 inf
        def invalid2inf(data, max_id):
            # 原本输出中，无效索引值为点云数目（刚好超出索引）
            return [torch.where(i >= j, torch.inf, i) for i, j in zip(data, max_id)]

        data["neighbors"] = invalid2inf(data["neighbors"], point_size)
        data["pools"] = invalid2inf(data["pools"], point_size)
        data["upsamples"] = invalid2inf(data["upsamples"], torch.roll(point_size, -1))

        data["lengths"] = offsets = torch.tensor([l[0] for l in data["stack_lengths"]])

        # 截取只属于原始输入的那部分
        def clip(data, offsets):
            return [p[:l] for p, l in zip(data, offsets)]

        data["points"] = clip(data["points"], offsets)
        data["neighbors"] = clip(data["neighbors"], offsets)
        data["pools"] = clip(data["pools"], torch.roll(offsets, -1))
        data["upsamples"] = clip(data["upsamples"], offsets)
        data["features"] = clip(data["features"], offsets)

        return True, data
