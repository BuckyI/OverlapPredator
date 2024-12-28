"""
用于加速和简化模型推理，封装的类
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import small_gicp
import torch
from loguru import logger

from datasets.dataloader import batch_grid_subsampling_kpconv, batch_neighbors_kpconv
from lib.benchmark_utils import to_o3d_feats, to_o3d_pcd

# 原始代码预处理、模型推理需要 tensor 类型的数据
DATA_TENSOR = Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
# 使用 numpy 类型的数据存储，注意：可能只包括单个点云的数据，此时无效位置设为了 inf
DATA_NP = Dict[str, Union[np.ndarray, List[np.ndarray]]]
# 模型数据流必须的字段
DATA_FIELDS = ["points", "neighbors", "pools", "upsamples", "stack_lengths", "features"]


def to_tensor(data: DATA_NP, device: Union[str, torch.device] = "cpu") -> DATA_TENSOR:
    """类型转换：获得 tensor，保证在合适的设备和为合适的数据类型"""
    assert all(field in data for field in DATA_FIELDS), f"data validation failed"

    def change_dtype(data, dtype=torch.float32):
        "turn (list of) numpy to tensor on specific device"
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=device, dtype=dtype)
        elif isinstance(data, list):
            return [torch.from_numpy(i).to(device=device, dtype=dtype) for i in data]
        raise Exception("unexpected data type", type(data))

    points = change_dtype(data["points"], torch.float32)
    neighbors = change_dtype(data["neighbors"], torch.int64)
    pools = change_dtype(data["pools"], torch.int64)
    upsamples = change_dtype(data["upsamples"], torch.int64)
    stack_lengths = change_dtype(data["stack_lengths"], torch.int32)
    features = change_dtype(data["features"], torch.float32)

    return {
        "points": points,
        "neighbors": neighbors,
        "pools": pools,
        "upsamples": upsamples,
        "stack_lengths": stack_lengths,
        "features": features,
    }


def to_numpy(data: DATA_TENSOR) -> DATA_NP:
    """将 tensor 格式的数据转换为 numpy 格式，便于本地存储"""
    result = {}
    for k, v in data.items():
        if isinstance(v, list):
            result[k] = [d.cpu().numpy() for d in v]
        else:
            result[k] = v.cpu().numpy()
    return result


def merge_data(source: DATA_NP, target: DATA_NP, device: Union[str, torch.device] = "cuda:0") -> DATA_TENSOR:
    """用于从本地取出 source 和 target 数据后，合并为模型推理需要的格式和数据类型"""
    # 避免修改原始数据（比如传入同一个点云的数据进行合成，会因内存共享出错）
    s1, s2 = source.copy(), target.copy()

    # fix id: 两个点云合并，则第二个点云的索引值需要添加第一个点云的长度
    # 注意存在本身就属于无效点的索引，即 inf，但是这里不需要担心，因为它们在相加后仍然为 inf
    offsets = np.array(s1["lengths"])
    s2["neighbors"] = [i + j for i, j in zip(s2["neighbors"], s1["lengths"])]
    s2["pools"] = [i + j for i, j in zip(s2["pools"], s1["lengths"])]
    s2["upsamples"] = [i + j for i, j in zip(s2["upsamples"], np.roll(s1["lengths"], -1))]

    def concatenate(l1, l2):
        """
        l1, l2: list[np.array] of same length
        """
        result = []
        for i, j in zip(l1, l2):
            # 有时候会出现 axis=1 维度不一致的问题，起因是 batch_neighbors_kpconv 获得的矩阵列数无法保证都为最大邻域数
            if i.shape[1] < j.shape[1]:  # pad i
                i = np.pad(i, ((0, 0), (0, j.shape[1] - i.shape[1])), mode="constant", constant_values=np.inf)
            elif j.shape[1] < i.shape[1]:  # pad j
                j = np.pad(j, ((0, 0), (0, i.shape[1] - j.shape[1])), mode="constant", constant_values=np.inf)
            result.append(np.concatenate([i, j], axis=0))
        return result

    points = concatenate(s1["points"], s2["points"])
    neighbors = concatenate(s1["neighbors"], s2["neighbors"])
    pools = concatenate(s1["pools"], s2["pools"])
    upsamples = concatenate(s1["upsamples"], s2["upsamples"])
    stack_lengths = [np.array([i, j]) for i, j in zip(s1["lengths"], s2["lengths"])]
    features = concatenate(s1["features"], s2["features"])

    # 将无效索引（inf）替换为最大索引值 +1, 并将数据类型转换为整型
    def fix_array(i, j):
        return np.where(i < np.inf, i, j).astype(np.int64)

    point_size = np.array([len(p) for p in points])
    neighbors = [fix_array(i, j) for i, j in zip(neighbors, point_size)]
    pools = [fix_array(i, j) for i, j in zip(pools, point_size)]
    upsamples = [fix_array(i, j) for i, j in zip(upsamples, np.roll(point_size, -1))]

    # 获得 tensor，保证在合适的设备和为合适的数据类型
    def change_dtype(data, dtype=torch.float32):
        "turn (list of) numpy to tensor on specific device"
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=device, dtype=dtype)
        elif isinstance(data, list):
            return [torch.from_numpy(i).to(device=device, dtype=dtype) for i in data]
        raise Exception("unexpected data type", type(data))

    points = change_dtype(points, torch.float32)
    neighbors = change_dtype(neighbors, torch.int64)
    pools = change_dtype(pools, torch.int64)
    upsamples = change_dtype(upsamples, torch.int64)
    stack_lengths = change_dtype(stack_lengths, torch.int32)
    features = change_dtype(features, torch.float32)

    result = {
        "points": points,
        "neighbors": neighbors,
        "pools": pools,
        "upsamples": upsamples,
        "stack_lengths": stack_lengths,
        "features": features,
    }
    return result


def split_data(array, length: int, func: Optional[Callable] = None):
    """方便从模型推理数据中分割出 source 和 target 的数据，通过 func 预处理"""
    if func is None:
        return array[:length], array[length:]
    else:
        return func(array[:length]), func(array[length:])


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
    def _preprocess(self, source: np.ndarray, target: np.ndarray) -> DATA_TENSOR:
        neighborhood_limits = self.neighborhood_limits

        points = torch.tensor(np.concatenate([source, target], axis=0))
        lengths = torch.tensor([len(source), len(target)], dtype=torch.int32)

        # Starting radius of convolutions
        r_normal = self.first_subsampling_dl * self.conv_radius

        # Starting layer
        layer_blocks = []
        layer = 0

        # Lists of inputs
        input_points: List[torch.Tensor] = []  # 第 i 层的 pcd
        input_neighbors: List[torch.Tensor] = []  # 第 i 层查询 pcd 邻居的索引
        # (Encoder)第 i 层对 pcd 进行 pooling / subsampling / 降采样的索引 (只有为 pool/strided 层才有效)
        input_pools: List[torch.Tensor] = []
        input_upsamples: List[torch.Tensor] = []  # (Decoder)第 i 层对 pcd 进行 upsampling / 上采样的索引
        input_batches_len: List[torch.Tensor] = []

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
        return {
            "points": input_points,
            "neighbors": input_neighbors,
            "pools": input_pools,
            "upsamples": input_upsamples,
            "stack_lengths": input_batches_len,
        }

    @torch.inference_mode()
    def preporcess(self, raw_points: np.ndarray) -> Tuple[bool, DATA_TENSOR]:
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

        # 受限于调用的函数，这里需要：
        # 生成假的 empty 点云，与原始点云共同处理
        # 转换为 Tensor
        empty = np.empty((2, 3), dtype=np.float64)
        data = self._preprocess(raw_points, empty)

        # 过滤掉数目太少的点
        if (pts := data["stack_lengths"][-1][0]) < self.min_point_size:
            logger.warning(f"Too few points ({pts}) in the input cloud, skipping")
            return False, {}

        # encode 获取特征
        for k, v in data.items():
            data[k] = [vv.to(self.device) for vv in v]
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

    def encode(self, points: np.ndarray) -> Tuple[bool, DATA_TENSOR]:
        return self.preporcess(points)

    @torch.inference_mode()
    def pair_decode(self, source: DATA_NP, target: DATA_NP) -> Tuple[bool, DATA_TENSOR]:
        """
        点云对共同 Decoder 编码操作
        source, target 是 encode 的结果，二者进行信息交互并通过 Decoder 获得
        - final_feature
        - scores_overlap
        - scores_saliency
        """
        required_fields = [
            "lengths",
            "points",
            "neighbors",
            "pools",
            "upsamples",
            "features",
        ]
        assert all([f in source for f in required_fields]), f"invalid input {required_fields}"
        assert all([f in target for f in required_fields]), f"invalid input {required_fields}"

        data = merge_data(source, target)
        feats, scores_overlap, scores_saliency = self.model.decode(data)
        data["final_feature"] = feats
        data["scores_overlap"] = scores_overlap
        data["scores_saliency"] = scores_saliency
        return True, data

    @torch.inference_mode()
    def encode_decode(self, source: np.ndarray, target: np.ndarray) -> DATA_TENSOR:
        """
        对点云对进行完整的模型推理操作，Encoder-Decoder
        可以传入同一个点云，即 source == target
        """
        data = self._preprocess(source, target)
        for k, v in data.items():
            data[k] = [vv.to(self.device) for vv in v]

        # 由于 pytorch 限制，model.encode 获得的为定长 tuple
        # 为了与其他数据保持类型一致，这里仍然转化为 list
        data["features"] = list(self.model.encode(data))
        feats, scores_overlap, scores_saliency = self.model.decode(data)
        data["final_feature"] = feats
        data["scores_overlap"] = scores_overlap
        data["scores_saliency"] = scores_saliency
        return data

    def registration(self, source: np.ndarray, target: np.ndarray, debug: bool = False):
        """
        source: source points
        target: target points
        return: transformation from source to target
        """
        data = self.encode_decode(source, target)
        result = self.registration_(data, debug)
        return result if debug else result["T"]  # type: ignore

    def registration_(self, data: DATA_TENSOR, debug: bool = False) -> DATA_NP:
        """根据模型输出结果进行点云配准，
        返回 dict 包括详细中间信息(如果 debug 为 True)和最终结果，方便实验
        """
        len_src = int(data["stack_lengths"][0][0])
        src_pcd, tgt_pcd = split_data(data["points"][0], len_src)
        src_feats, tgt_feats = split_data(data["final_feature"], len_src)
        src_overlap, tgt_overlap = split_data(data["scores_overlap"], len_src)
        src_saliency, tgt_saliency = split_data(data["scores_saliency"], len_src)

        def _downsample_improved(pcd, feats, scores, max_points=1000):
            # NOTE: 论文中采样数为 1000 时效果最好
            max_points = min(max_points, pcd.size(0))
            # assert max_points > 0
            sample_id: torch.Tensor = torch.multinomial(scores, max_points, replacement=False)
            sampled_pcd = pcd[sample_id]
            sampled_feats = feats[sample_id]
            sampled_scores = scores[sample_id]
            return (
                sample_id.cpu().numpy(),
                sampled_pcd.cpu().numpy(),
                sampled_feats.cpu().numpy(),
                sampled_scores,
            )  #  衡量采样结果的 confidence

        src_scores = src_overlap * src_saliency  # source length
        tgt_scores = tgt_overlap * tgt_saliency  # target length

        src_idx, src_pcd_down, src_feats_down, src_score_down = _downsample_improved(src_pcd, src_feats, src_scores)
        tgt_idx, tgt_pcd_down, tgt_feats_down, tgt_score_down = _downsample_improved(tgt_pcd, tgt_feats, tgt_scores)

        # feature based ransac registration
        # NOTE: registration_ransac_based_on_feature_matching 本身就带 mutual 参数, 可能这个函数写的时候还不带, 所以自己实现了一个 mutual
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            to_o3d_pcd(src_pcd_down),
            to_o3d_pcd(tgt_pcd_down),
            to_o3d_feats(src_feats_down),
            to_o3d_feats(tgt_feats_down),
            True,  # Enables mutual filter such that the correspondence of the source point’s correspondence is itself.
            0.05,  # Maximum correspondence points-pair distance.
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,  # ransac_n
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
        )

        # small gicp refine
        T = result_ransac.transformation
        for s in (0.25, 0.1, 0.02):
            reg_res = small_gicp.align(
                tgt_pcd_down,
                src_pcd_down,
                init_T_target_source=T,
                registration_type="GICP",
                downsampling_resolution=s,
                max_correspondence_distance=2 * s,
                max_iterations=50,
            )
            T = reg_res.T_target_source

        return (
            dict(
                T=T,
                T0=result_ransac.transformation,
                source_raw=src_pcd.cpu().numpy(),
                target_raw=tgt_pcd.cpu().numpy(),
                source_raw_scores=src_scores.cpu().numpy(),
                target_raw_scores=tgt_scores.cpu().numpy(),
                source=src_pcd_down,
                target=tgt_pcd_down,
                source_id=src_idx,
                target_id=tgt_idx,
                source_feats=src_feats_down,
                target_feats=tgt_feats_down,
                source_scores=src_score_down.cpu().numpy(),
                target_scores=tgt_score_down.cpu().numpy(),
            )
            if debug
            else dict(T=T)
        )
