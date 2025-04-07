"""
采用 chunk2model 的思路进行目标三维重建
主要分成两个步骤：
1. 根据原始帧提取目标生成 chunk（提取目标部分预先缓存了），利用 tsdf 融合获得 chunk pcd
2. chunk pcd 对全局 tsdf model 进行配准，配准成功后融合到全局 tsdf model 中
3. 配准成功性判断：采用 voting strategy
"""

# %%
import os
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from loguru import logger
from tqdm import tqdm

from configs.settings import params
from datasets.tum import TUMDataset
from models.checker import Checker, check_registration
from models.model import Model
from pipeline.data import Chunk
from pipeline.evaluate import voting_evaluate
from pipeline.recon import optimize_chunk, tsdf2, tsdf3
from utils.convert import rgb2luminance
from utils.evaluate import (
    absolute_trajectory_error,
    binary_classification_metrics,
    pose_difference2,
)
from utils.log import log_formats, simplify
from utils.preprocess import refine_mask
from utils.reconstruction import save_scene
from utils.registration import GICP_registration
from utils.storage import CacheSE
from utils.visualize import show_colored_points, show_transformation

# %% 初始化
simplify(log_formats[2])
logger.add("run/log.txt")

param = params[5]
dataset = TUMDataset(param[0], param[2])
teddy_cache = CacheSE("data/teddy1_dataset/teddy_target_cache.h5", mode="r")

model = Model()

# remove old cache
os.makedirs("run", exist_ok=True)
cache_path = "run/cache.h5"
if os.path.exists(cache_path):
    os.remove(cache_path)
cache = CacheSE(cache_path, mode="w", compress=True)  # overwrite


# %% 函数定义
# 这部分需要用到全局变量，不太好独立出来
# 加载数据集相关函数的定义 🌟🌟🌟
def load_points(frame_id: int):
    "dataset -> cache -> load points"
    return teddy_cache[f"points/{frame_id}"]


def load_depth(frame_id):
    mask = teddy_cache[f"mask/{frame_id}"]
    depth = dataset.frames[frame_id].depth
    masked_depth = depth.copy() * mask
    return masked_depth.astype(np.uint16)


def load_color(frame_id):
    mask = teddy_cache[f"mask/{frame_id}"]
    color = dataset.frames[frame_id].color
    masked_color = color.copy()
    masked_color[mask == 0] = 0
    return masked_color


# 配准的函数定义 🌟🌟🌟
def registration(sid: int, tid: int, init_T=np.eye(4)):
    """
    sid: source id in dataset
    tid: target id in dataset
    return:
        flag: bool, True if valid
        trans: np.ndarray, transformation matrix
    """
    sp = load_points(sid)
    tp = load_points(tid)

    trans, _ = GICP_registration(sp, tp, init_T)
    flag = check_registration(sp, tp, trans)
    logger.debug(f"REG {sid} -> {tid} valid: {flag}")
    return flag, trans


# # chunk2model 思路不需要频繁编码重复的点云，因此无需缓存
# # dataset_cache = DatasetCache("run/dataset_cache.h5", mode=mode)
# def load_chunk_data(chunk_id: int, points: Optional[np.ndarray] = None):
#     "load chunk data from cache or compute it. points are need if cache miss."
#     global dataset_cache, model
#     if not dataset_cache.exists("chunk", chunk_id):  # 缓存不存在
#         assert points is not None, "points is needed if cache miss"
#         res, data = model.encode(points)
#         if not res:  # 模型编码失败
#             logger.error(f"encode chunk {chunk_id} failed")
#             # raise Exception("encode chunk failed, really unexpected")
#             return None
#         else:  # 模型编码成功
#             data = to_numpy(data)
#             dataset_cache.save_frame_data("chunk", chunk_id, data)
#             logger.info(f"encode chunk {chunk_id} done")
#     else:  # 缓存存在
#         data = dataset_cache.load_frame_data("chunk", chunk_id)
#         logger.info(f"load chunk {chunk_id} from cache")
#     return data


# 三维重建 🌟🌟🌟
# def construction_chunk(chunk: Chunk):
#     """
#     点云直接拼接
#     """
#     "get chunk pcd"
#     pcds = []
#     for i in range(len(chunk.frame_ids)):
#         frame_id = chunk.frame_ids[i]
#         frame_pose = chunk.frame_poses[i]
#         points = load_points(frame_id)

#         # near_points = points[points[:, 2] < 2]
#         pcds.append(transform(points, frame_pose))

#     merged_pcd = downsample(merge_points(pcds), 0.01)
#     return merged_pcd


def construction_chunk_tsdf(chunk: Chunk):
    """
    tsdf construction
    """
    depths = (load_depth(fid) for fid in chunk.frame_ids)
    colors = (load_color(fid) for fid in chunk.frame_ids)
    poses = chunk.frame_poses

    try:
        vbg = tsdf2(
            depths, colors, poses, dataset.K, dataset.depth_scale, vol_size=0.01
        )

        pcd = vbg.extract_point_cloud()
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        points = pcd.point.positions.numpy()
        colors = pcd.point.colors.numpy()
        luminance = rgb2luminance(colors)
        return True, {
            "points": points,
            "colors": colors,
            "luminance": luminance,
        }
    except Exception as e:
        logger.error(f"TSDF construction failed: {e}")
        return False, {}


# def teddy_reconstruction_tsdf(
#     chunks: List[Chunk], pose_label="pose_optimized", path="run/test.ply"
# ):
#     """
#     chunks: list[Chunk] 参与重建的 chunk
#     pose_label: str, chunk.meta 中的 pose 标签，如 "pose_optimized", "pose", "gt_pose"
#     """
#     global teddy_cache, dataset
#     frame_id_pose = defaultdict(list)  # frame_id -> world poses
#     for i, c in enumerate(chunks):
#         assert isinstance(c, Chunk)
#         # if c.meta["valid_state"] != "valid":
#         #     continue
#         world_poses = c.transform(c.meta[pose_label], inplace=False)
#         for i, fid in enumerate(c.frame_ids):
#             frame_id_pose[fid].append(world_poses[i])

#     frame_ids = []
#     frame_poses = []
#     for fid, poses in frame_id_pose.items():
#         frame_ids.append(fid)
#         frame_poses.append(average_poses(poses))

#     depths = (load_depth(fid) for fid in frame_ids)
#     colors = (load_color(fid) for fid in frame_ids)
#     vbg = tsdf2(
#         depths, colors, frame_poses, dataset.K, dataset.depth_scale, vol_size=0.01
#     )
#     save_scene(vbg, path, "mesh")
#     return vbg


# def teddy_reconstruction_pcd(
#     chunks: List[Chunk], pose_label="pose_optimized", path="run/merge.ply"
# ):
#     assert all("points" in c.meta for c in chunks)
#     assert all(pose_label in c.meta for c in chunks)
#     pcds = []
#     for i, c in enumerate(chunks):
#         # if c.meta["valid_state"] != "valid":  # not valid
#         #     continue
#         pcds.append(transform(c.meta["points"], c.meta[pose_label]))
#     pcd = merge_points(pcds)
#     pcd = downsample(pcd, 0.01)

#     # show_pcd(global_scene)
#     save_pcd(pcd, path=path)
#     return pcd


# evaluation 🌟🌟🌟
# def evaluate_chunk_edge(e: Edge):
#     global dataset
#     sc = chunks[e.source_id]  # source chunk
#     tc = chunks[e.target_id]  # target chunk
#     scp = dataset.frames[sc.frame_ids[0]].pose  # source chunk pose
#     tcp = dataset.frames[tc.frame_ids[0]].pose  # target chunk pose
#     gt_trans = np.linalg.inv(tcp) @ scp
#     return pose_difference2(e.T_ts, gt_trans)


# %% [configuration] 🌟🌟🌟
CHUNK_SIZE = 90  # 30fps  1s -> 30 frames  90
TRANS_THR = 0.5
ROT_THR = 0.3
OVERLAP_RATIO = 0.4
MIN_POINTS = 2000  # 每一帧的最小点云数目
MIN_CHUNK_POINTS = 2000  # 每个 chunk 的最小点云数目 8000
THR_SPACE = 0.01
THR_FEATURE = 0.4
THR_COLOR = 0.15


# %% [chunk generate] 🌟🌟🌟
class ChunkState(Enum):
    begin = auto()  # no chunk before, initialize
    progress = auto()  # chunk accumulating
    full = auto()  # chunk success before
    icp_failed = auto()  # chunk break before
    trans_thr = auto()  # accumulated trans or rot reach threshold
    no_target = auto()  # no target area in current frame, thus complete chunk


chunks: List[Chunk] = []
chunk = None
state = ChunkState.begin


def create_chunk():
    global chunks
    chunk = Chunk(registration)
    chunk.meta["id"] = len(chunks)
    return chunk


def post_process_chunk(c: Chunk):
    if len(c.frame_ids) == 0:
        c.meta["valid_state"] = "zero frames"
        return

    if len(c.frame_ids) > 5:
        c.enhance_parallel(skip_every=(3, 5, 7, 9, 10))
        optimize_chunk(c)
    # points = construction_chunk(c)
    # c.meta["points"] = points
    res, metas = construction_chunk_tsdf(c)
    if not res:
        logger.warning(f"chunk {c.meta['id']} construction failed")
        c.meta["valid_state"] = "construction failed"
        return
    c.meta.update(metas)

    if len(c.meta["points"]) < MIN_CHUNK_POINTS:
        logger.warning(f"chunk {c.meta['id']} has too few points")
        c.meta["valid_state"] = "not enough points"
        return

    c.meta["valid_state"] = "valid"


for i in range(0, len(dataset), 1):
    # init
    if state == ChunkState.begin:
        chunk = create_chunk()
        chunk.meta["pose"] = np.eye(4)
    elif state == ChunkState.full or state == ChunkState.trans_thr:
        # 处理最后一个 chunk
        assert chunk is not None
        post_process_chunk(chunk)
        chunk.meta["state"] = str(state)  # 记录 chunk 终止时的状态
        if chunk.meta["valid_state"] == "valid":
            chunks.append(chunk)

        # 重新开始一个新的 chunk
        chunk = create_chunk()
        # 由于上一个 chunk 成功完结，所以建立新的 chunk 添加一些重叠，从而建立可靠的联系
        # 当然如果上一个 chunk 被丢弃，那么这个补充就不会发挥作用
        pose = chunk.append_overlap(chunks[-1], ratio=OVERLAP_RATIO)
        chunk.meta["pose"] = chunks[-1].meta["pose"] @ pose

    elif state == ChunkState.icp_failed:
        assert chunk is not None and chunk.frame_poses
        post_process_chunk(chunk)
        chunk.meta["state"] = str(state)  # 记录 chunk 终止时的状态
        if chunk.meta["valid_state"] == "valid":
            chunks.append(chunk)

        pose = chunk.frame_poses[-1]  # 使用 last chunk 最后一次配准成功的位姿为参考
        chunk = create_chunk()
        chunk.meta["pose"] = chunks[-1].meta["pose"] @ pose
    elif state == ChunkState.no_target:
        # 跟踪丢失，只能希望模型配准阶段可以连接成功
        assert chunk is not None
        post_process_chunk(chunk)
        chunk.meta["state"] = str(state)  # 记录 chunk 终止时的状态
        if chunk.meta["valid_state"] == "valid":
            chunks.append(chunk)

        logger.warning(f"Lost Tracking at {i}")
        chunk = create_chunk()
        chunk.meta["pose"] = chunks[-1].meta["pose"]  # 猜测停留在上一个 chunk 附近
    else:
        assert state == ChunkState.progress
        # TODO 其他分割 chunk 的思路，如以 chunk 合成点云的点云数目到达10万为限制

    assert chunk is not None

    # 保证当前帧需要有目标物体
    points = load_points(i)
    if len(points) < MIN_POINTS:
        state = ChunkState.no_target
        continue

    code = chunk.append(i)
    state = ChunkState.progress
    if code == "icp-failed":
        state = ChunkState.icp_failed
        logger.error("icp failed, lose tracking")
    elif len(chunk.frame_ids) == CHUNK_SIZE:  # chunk full
        state = ChunkState.full
        logger.info("chunk full")
    else:
        # trans, rot = get_trans_rot(chunk.frame_poses[-1])
        # if trans > TRANS_THR or rot > ROT_THR:  # 0.5m, 0.2rad
        #     state = ChunkState.trans_thr
        #     logger.info(f"translation threshold: {trans=} {rot=}")
        ...
else:
    if state == ChunkState.progress:  # finish loop but chunk not full
        logger.info("chunk not full")
        # 检查一下要不要保留这一帧


# add some useful info to chunk
for i, c in enumerate(chunks):
    assert c.meta["id"] == i  # NOTE: chunk id == index in `chunks`
    c.meta["gt_pose"] = dataset.frames[c.frame_ids[0]].pose
    c.meta["valid_state"] = c.meta.get("valid_state", "valid")  # default valid

cache.dump(chunks, "run/chunks.joblib")

for i, c in enumerate(chunks):
    show_colored_points(
        c.meta["points"],
        c.meta["colors"],
        export=f"run/vis/chunk_{i}_{len(c.meta['points'])}.html",
    )
# %% [chunk2model 3d reconstruction] 🌟🌟🌟
chunks = cache.load("run/chunks.joblib")

history = []
vbg = None
world_points = None
world_luminance = None
for i, chunk in enumerate(chunks):
    debug: Dict[str, Any] = dict(
        chunk_id=i,
        gt_pose=chunk.meta["gt_pose"],
        state="",
        point_size=len(chunk.meta["points"]),
    )
    history.append(debug)

    if vbg is None:
        chunk_pose = chunk.meta["gt_pose"]
        debug["state"] = "first chunk"
    else:
        assert world_points is not None
        assert world_luminance is not None
        try:
            source = chunk.meta["points"]
            target = world_points  # tsdf 提取的点云
            reg_result = model.registration(source, target, debug=True)
            # check_result = checker.check_model_registration_debug(reg_result)

            chunk_pose: np.ndarray = reg_result["T"]

            # evaluate
            source = reg_result["source_raw"].cpu().numpy()  # N1, 3
            target = reg_result["target_raw"].cpu().numpy()  # N2, 3
            source_feat = reg_result["source_raw_feats"].cpu().numpy()  # N1, 32
            target_feat = reg_result["target_raw_feats"].cpu().numpy()  # N2, 32
            source_luminance = chunk.meta["luminance"]
            target_luminance = world_luminance

            space_dist, feature_dist, color_dist = voting_evaluate(
                source,
                target,
                source_feat,
                target_feat,
                source_luminance,
                target_luminance,
                chunk_pose,
            )
            votes = (
                (space_dist < THR_SPACE)
                & (feature_dist > THR_FEATURE)
                & (color_dist < THR_COLOR)
            )
            flag = np.sum(votes) / len(votes) > 0.4
            debug["state"] = "model registration"
            debug["chunk_pose"] = chunk_pose
            debug["space_dist"] = space_dist
            debug["feature_dist"] = feature_dist
            debug["color_dist"] = color_dist
            debug["flag"] = flag
            debug["votes"] = votes
            debug["vote_rate"] = np.sum(votes) / len(votes)
            debug["error"] = pose_difference2(chunk.meta["gt_pose"], chunk_pose)

        except Exception as e:
            logger.error(f"chunk {chunk.meta['id']} registration failed: {e}")
            debug["state"] = "registration failed"
            continue

        show_transformation(
            source, target, chunk_pose, export=f"run/vis/reg_{i}_{flag}.html"
        )
        if not flag:  # not valid
            logger.warning(f"chunk {chunk.meta['id']} registration filtered")
            debug["state"] = "registration excluded"
            continue

    debug["chunk_pose"] = chunk_pose
    frame_ids = chunk.frame_ids
    frame_poses = chunk.transform(chunk_pose, inplace=False)
    depths = (load_depth(i) for i in frame_ids)
    colors = (load_color(i) for i in frame_ids)

    vbg = tsdf3(
        depths=depths,
        colors=colors,
        poses=frame_poses,
        K=dataset.K,
        depth_scale=dataset.depth_scale,
        vol_size=0.01,
        vbg=vbg,
    )

    # 提取模型
    world_pcd = vbg.extract_point_cloud()
    world_pcd = world_pcd.voxel_down_sample(voxel_size=0.01)
    world_points = world_pcd.point.positions.numpy()
    world_colors = world_pcd.point.colors.numpy()
    world_luminance = rgb2luminance(world_colors)
    save_scene(vbg, f"run/vis/tsdf_{i}.ply", type="mesh")
    show_colored_points(world_points, world_colors, export=f"run/vis/{i}.html")

cache.dump(history, "run/history.joblib")

# %% [评估过滤规则的效果]
history = cache.load("run/history.joblib")
for h in history:
    # 计算 fitness
    if "space_dist" in h:
        vote = h["space_dist"] < THR_SPACE
        h["fitness"] = vote.sum() / len(vote)
df = pd.DataFrame(history, columns=["chunk_id", "error", "vote_rate", "fitness"])
fig = px.scatter(
    df,
    x="error",
    y="vote_rate",
    color="fitness",
    hover_data=df.keys(),
)
fig.show()

# %% [matplot]
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    x=df["error"],
    y=df["vote_rate"],
    c=df["fitness"],
    cmap="coolwarm",  # 可以选择其他颜色映射
    alpha=0.7,
    s=100,  # 点的大小
)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label("Fitness")


# %%
gt_lable = df["error"] < 1
label = df["fitness"] > 0.95
print(label.value_counts())
print(binary_classification_metrics(gt_lable, label))
label = df["vote_rate"] > 0.4
print(label.value_counts())
print(binary_classification_metrics(gt_lable, label))

# %% 实验
# 利用 frame 真值 pose 进行目标重建 ⚙️⚙️
# 进行目标重建
# 场景重建 dataset.frames[fid].depth, dataset.frames[fid].color
