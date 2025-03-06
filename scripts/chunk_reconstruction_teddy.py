"""
snippet to reconstruction teddy
针对 teddy 的场景做了很多改进和调整
"""

# %%
import os
from collections import defaultdict
from enum import Enum, auto
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from loguru import logger

from configs.settings import params
from datasets.tum import TUMDataset
from models.checker import Checker
from models.model import Model, split_data, to_numpy
from utils.convert import average_poses, downsample, merge_points, transform
from utils.evaluate import get_trans_rot, pose_difference2
from utils.log import log_formats, simplify
from utils.reconstruction import (
    Chunk,
    Edge,
    construct_pose_graph,
    extract_pcd,
    optimize_pose_graph,
    save_scene,
    tsdf2,
)
from utils.registration import GICP_registration
from utils.storage import CacheSE, DatasetCache
from utils.visualize import (
    pick_point,
    show_colored_points,
    show_pcd,
    show_pcd_with_keypoints,
    show_pcds,
    show_rgbd_image,
    show_transformation,
)

simplify(log_formats[2])
logger.add("run/log.txt")

param = params[5]
dataset = TUMDataset(param[0], param[2])
dataset_points = CacheSE("data/teddy1_dataset/teddy_points.hdf5", mode="r")
dataset_masks = CacheSE("data/teddy1_dataset/teddy_mask.hdf5", mode="r")

model = Model()
checker = Checker()

# remove old cache
os.makedirs("run", exist_ok=True)
_ps = ["run/cache.h5", "run/dataset_cache.h5"]
for p in _ps:
    if os.path.exists(p):
        os.remove(p)

cache = CacheSE("run/cache.h5", mode="w")  # overwrite
dataset_cache = DatasetCache("run/dataset_cache.h5", mode="w")


def load_points(frame_id: int):
    "dataset -> cache -> load points"
    return dataset_points[f"{frame_id:05d}"]


# def load_points(frame_id: int):
#     "dataset -> cache -> load points"
#     global dataset, cache, dataset_cache, dataset_points

#     key = f"points/{frame_id}"
#     if key in cache.hdf5:
#         pcd = cache[key]
#     else:
#         pcd = dataset.frames[frame_id].pcd_array
#         cache[key] = pcd
#     return pcd


def registration(sid: int, tid: int, init_T=np.eye(4)):
    """
    sid: source id in dataset
    tid: target id in dataset
    return:
        flag: bool, True if valid
        trans: np.ndarray, transformation matrix
    """
    global checker

    sp = load_points(sid)
    tp = load_points(tid)

    trans, _ = GICP_registration(sp, tp, init_T)
    flag = checker.check_registration(sp, tp, trans)
    logger.debug(f"REG {sid} -> {tid} valid: {flag}")
    return flag, trans


# %% configuration
CHUNK_SIZE = 90  # 30fps  1s -> 30 frames 5s 单个 chunk 的最大帧数
TRANS_THR = 0.5
ROT_THR = 0.3
OVERLAP_RATIO = 0.4
n = np.ceil(len(dataset) / CHUNK_SIZE)  # n chunks
total_reg = n * (n - 1) // 2
logger.info(f"{total_reg=}, {len(dataset)=}, {CHUNK_SIZE=}, {n=}")


# %% chunk generate
class ChunkState(Enum):
    begin = auto()  # no chunk before, initialize
    progress = auto()  # chunk accumulating
    full = auto()  # chunk success before
    icp_failed = auto()  # chunk break before
    trans_thr = auto()  # accumulated trans or rot reach threshold
    no_target = auto()  # no target area in current frame, thus complete chunk


chunks = []
# 用于后续全局优化，记录 chunk 之间因时序连接而产生的变换关系
chunk_poses = []
chunk_edges = []

chunk = None
# CHUNK_SIZE = CHUNK_SIZE / 2  # 因为跳帧了
state = ChunkState.begin


def remove_bad_chunks():
    global chunks, chunk_poses, chunk_edges
    assert len(chunks)
    c = chunks[-1]
    # 如果当前 chunk 的帧数小于 3，则认为该 chunk 不完整，需要重新开始
    if len(c.frame_ids) < 3:
        chunks.pop()
        chunk_poses.pop()
        if chunk_edges and chunk_edges[-1].source_id == c.id:
            chunk_edges.pop()


for i in range(0, len(dataset), 1):
    # init
    if state == ChunkState.begin:
        chunk = Chunk(registration, id=len(chunks))
        chunks.append(chunk)
        chunk_poses.append(np.eye(4))
    elif state == ChunkState.full or state == ChunkState.trans_thr:
        remove_bad_chunks()
        chunk = Chunk(registration, id=len(chunks))
        pose = chunk.append_overlap(chunks[-1], ratio=OVERLAP_RATIO)
        chunks.append(chunk)
        assert len(chunks) >= 2
        chunk_edges.append(Edge(chunks[-1].id, chunks[-2].id, pose, "odometry"))
        chunk_poses.append(chunk_poses[-1] @ pose)
    elif state == ChunkState.icp_failed:
        assert chunk is not None and chunk.frame_poses
        remove_bad_chunks()
        pose = chunk.frame_poses[-1]  # 使用 last chunk 最后一次配准成功的位姿为参考
        chunk = Chunk(registration, id=len(chunks))
        chunks.append(chunk)
        assert len(chunks) >= 2
        chunk_edges.append(Edge(chunks[-1].id, chunks[-2].id, pose, "guess"))
        chunk_poses.append(chunk_poses[-1] @ pose)
    elif state == ChunkState.no_target:
        # 跟踪丢失，只能希望模型配准阶段可以连接成功
        remove_bad_chunks()
        logger.warning(f"Lost Tracking at {i}")
        chunk = Chunk(registration, id=len(chunks))
        chunks.append(chunk)
        chunk_poses.append(chunk_poses[-1])  # 猜测停留在上一个 chunk 附近
    else:
        assert state == ChunkState.progress
        # TODO 其他分割 chunk 的思路，如以 chunk 合成点云的点云数目到达10万为限制

    assert chunk is not None

    # 保证当前帧需要有目标物体
    points = load_points(i)
    if len(points) < 2000:
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
        trans, rot = get_trans_rot(chunk.frame_poses[-1])
        if trans > TRANS_THR or rot > ROT_THR:  # 0.5m, 0.2rad
            state = ChunkState.trans_thr
            logger.info(f"translation threshold: {trans=} {rot=}")
else:
    if state == ChunkState.progress:  # finish loop but chunk not full
        logger.info("chunk not full")
        # 检查一下要不要保留这一帧
    remove_bad_chunks()

cache.dump(chunks, "run/chunks.joblib")
cache.dump(chunk_poses, "run/chunk_poses.joblib")
cache.dump(chunk_edges, "run/chunk_edges.joblib")


# %% optimize chunks inner frames
# def construction(chunk: Chunk, dataset):
#     logger.info(f"chunk {chunk.frame_ids[0]}-{chunk.frame_ids[-1]} construct")
#     timestamps = [dataset.timestamps[i] for i in chunk.frame_ids]
#     depths, colors = dataset.batch_load_rgbd(timestamps)
#     K = dataset.K
#     vbg = tsdf2(depths, colors, chunk.frame_poses, K, dataset.depth_scale, vol_size=0.01)
#     points = vbg.extract_point_cloud().point.positions.numpy()
#     return points
def construction(chunk: Chunk):
    global dataset
    pcds = []
    for i in range(len(chunk.frame_ids)):
        frame_id = chunk.frame_ids[i]
        frame_pose = chunk.frame_poses[i]
        points = load_points(frame_id)

        near_pcd = points[points[:, 2] <= 2.5]  # 只取近处的点
        pcds.append(transform(near_pcd, frame_pose))

    merged_pcd = downsample(merge_points(pcds), 0.01)
    return merged_pcd


pcds = []
for c in chunks:
    assert isinstance(c, Chunk)
    c.enhance_parallel(skip_every=(3, 5, 7, 9, 10))
    c.optimize()
    points = construction(c)
    pcds.append(points)
    logger.info(f"chunk {c.id} done")

cache.dump(chunks, "run/chunks.joblib")
cache.dump(pcds, "run/pcds.joblib")

# %% TEST PCD
# for i, p in enumerate(pcds):
#     show_pcd(p, export=f"run/vis/chunk_{i}.html")


# %% TEST MODEL reg
# model = Model()
# sid, tid = 5, 1
# result = model.registration(pcds[sid], pcds[tid], debug=True)
# flag = checker.check_model_registration(result, 0.9)
# print(flag)
# show_transformation(pcds[sid], pcds[tid], result["T"])


# %% model registration
def load_chunk_data(chunk_id: int, points: Optional[np.ndarray] = None):
    "load chunk data from cache or compute it. points are need if cache miss."
    global dataset_cache, model
    if not dataset_cache.exists("chunk", chunk_id):  # 缓存不存在
        assert points is not None, "points is needed if cache miss"
        res, data = model.encode(points)
        if not res:  # 模型编码失败
            logger.error(f"encode chunk {chunk_id} failed")
            # raise Exception("encode chunk failed, really unexpected")
            return None
        else:  # 模型编码成功
            data = to_numpy(data)
            dataset_cache.save_frame_data("chunk", chunk_id, data)
            logger.info(f"encode chunk {chunk_id} done")
    else:  # 缓存存在
        data = dataset_cache.load_frame_data("chunk", chunk_id)
        logger.info(f"load chunk {chunk_id} from cache")
    return data


chunks = cache.load("run/chunks.joblib")
pcds = cache.load("run/pcds.joblib")
chunk_edges = cache.load("run/chunk_edges.joblib")
model = Model()

# %% 预先编码，及时找到失败的 chunk
model.min_point_size = 10
invalid_chunk_ids = []
for i, pcd in enumerate(pcds):
    data = load_chunk_data(i, pcd)
    if data is None:
        logger.error(f"chunk {i} encode failed")
        invalid_chunk_ids.append(i)
logger.warning(f"{len(invalid_chunk_ids)} invalid chunks found")
cache.dump(invalid_chunk_ids, "run/invalid_chunk_ids.joblib")

# %% 正式配准
for tid in range(len(chunks)):
    for sid in range(tid + 1, len(chunks)):
        if sid in invalid_chunk_ids or tid in invalid_chunk_ids:
            continue

        sp, tp = pcds[sid], pcds[tid]
        sd = load_chunk_data(sid, sp)
        td = load_chunk_data(tid, tp)
        _, inputs = model.pair_decode(sd, td)
        reg_result = model.registration_(inputs, debug=True)
        flag = checker.check_model_registration(reg_result, 0.9)
        if not flag:
            _logger = logger.error if sid == tid + 1 else logger.warning
            _logger(f"failed to register chunk {sid}-{tid}")
            continue

        trans = reg_result["T"]
        assert isinstance(trans, np.ndarray)
        chunk_edges.append(Edge(sid, tid, trans, "model"))
        logger.debug(f"add edge {sid}-{tid}")

cache.dump(chunk_edges, "run/chunk_edges_model.joblib")

# %% 位姿图优化
chunks = cache.load("run/chunks.joblib")
chunk_poses = cache.load("run/chunk_poses.joblib")
chunk_edges = cache.load("run/chunk_edges_model.joblib")
invalid_chunk_ids = cache.load("run/invalid_chunk_ids.joblib")
pcds = cache.load("run/pcds.joblib")

pose_graph = o3d.pipelines.registration.PoseGraph()
chunk2node = dict()  # map from chunk id to node id
assert len(chunk_poses) == len(chunks)
for i in range(len(chunks)):
    chunk2node[i] = i  # len(pose_graph.nodes)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(chunk_poses[i]))

for e in chunk_edges:
    edge = o3d.pipelines.registration.PoseGraphEdge(
        chunk2node[e.source_id],
        chunk2node[e.target_id],
        e.T_ts,
        uncertain=(e.edge_type != "odomerty"),
    )
    pose_graph.edges.append(edge)

pose_graph = optimize_pose_graph(pose_graph, verbose=True)
logger.info("finished pose graph optimization")

poses = [n.pose for n in pose_graph.nodes]
cache.dump(poses, "run/chunk_poses_optimized.joblib")

# %% 点云融合进行三维重建
from utils.reconstruction import save_pcd, simple_merge_points

poses = cache.load("run/chunk_poses_optimized.joblib")
invalid_chunk_ids = cache.load("run/invalid_chunk_ids.joblib")
pcds = cache.load("run/pcds.joblib")

poses_ = [p for i, p in enumerate(poses) if i not in invalid_chunk_ids]
pcds_ = [p for i, p in enumerate(pcds) if i not in invalid_chunk_ids]

global_scene = simple_merge_points(pcds_, poses_)
# show_pcd(global_scene)
save_pcd(global_scene, path="run/scene_simpmerged.ply")
cache.dump(global_scene, "run/scene_simpmerged_array.joblib")

# %% TSDF: get frame poses
chunk_poses = cache.load("run/chunk_poses_optimized.joblib")
chunks = cache.load("run/chunks.joblib")
invalid_chunk_ids = cache.load("run/invalid_chunk_ids.joblib")

frame_id_pose = defaultdict(list)
for i in range(len(chunks)):
    if i in invalid_chunk_ids:
        continue

    c = chunks[i]
    assert isinstance(c, Chunk)
    world_poses = c.transform(chunk_poses[i], inplace=False)
    for i, fid in enumerate(c.frame_ids):
        frame_id_pose[fid].append(world_poses[i])

frame_ids = []
frame_poses = []
for fid, poses in frame_id_pose.items():
    frame_ids.append(fid)
    frame_poses.append(average_poses(poses))

final_frame_poses = {fid: p for fid, p in zip(frame_ids, frame_poses)}
cache.dump(final_frame_poses, "run/frame_poses.joblib")

# %% TSDF: integrate frames
depths = (dataset.frames[fid].depth for fid in frame_ids)
colors = (dataset.frames[fid].color for fid in frame_ids)

vbg = tsdf2(depths, colors, frame_poses, dataset.K, dataset.depth_scale, vol_size=0.02)
# pcd = vbg.extract_point_cloud()
# points = pcd.point.positions.numpy()
save_scene(vbg, "run/scene_tsdf.ply", "mesh")
logger.info("finished tsdf reconstruction")
# %% 尝试使用真值 chunk 调整位姿
from utils.evaluate import pose_difference2

chunks = cache.load("run/chunks.joblib")
invalid_chunk_ids = cache.load("run/invalid_chunk_ids.joblib")

frame_id_pose = defaultdict(list)
for i in range(len(chunks)):
    if i in invalid_chunk_ids:
        continue

    c = chunks[i]
    assert isinstance(c, Chunk)

    gt_chunk_pose = dataset.frames[c.frame_ids[0]].pose
    world_poses = c.transform(gt_chunk_pose, inplace=False)
    for i, fid in enumerate(c.frame_ids):
        frame_id_pose[fid].append(world_poses[i])

frame_ids = []
frame_poses = []
for fid, poses in frame_id_pose.items():
    frame_ids.append(fid)
    frame_poses.append(average_poses(poses))

final_frame_poses = {fid: p for fid, p in zip(frame_ids, frame_poses)}
cache.dump(final_frame_poses, "run/frame_poses_improved.joblib")

# 过滤掉位子误差大的帧，实际上所有的帧都小于 0.3
frame_ids_ = []
frame_poses_ = []
for fid, p in final_frame_poses.items():
    diff = pose_difference2(p, dataset.frames[fid].pose)
    if diff < 0.3:
        frame_ids_.append(fid)
        frame_poses_.append(p)
    else:
        logger.warning(f"frame {fid} pose difference too large: {diff}")

# 最终适合参与重建的帧
cache.dump(
    {"frame_ids": frame_ids_, "frame_poses": frame_poses_},
    "run/frame_id_pose_improved.joblib",
)

# %% TSDF: integrate frames with improved poses
depths = (dataset.frames[fid].depth for fid in frame_ids_)
colors = (dataset.frames[fid].color for fid in frame_ids_)

vbg = tsdf2(depths, colors, frame_poses_, dataset.K, dataset.depth_scale, vol_size=0.01)
save_scene(vbg, "run/scene_tsdf_improved.ply", "mesh")
logger.info("finished tsdf reconstruction")


# %% object reconstruction with improved poese
frame_id_pose = cache.load("run/frame_id_pose_improved.joblib")
mask_cache = CacheSE("data/teddy1_dataset/teddy_mask_fixed.hdf5", "r")
frame_ids = frame_id_pose["frame_ids"]
frame_poses = frame_id_pose["frame_poses"]


def load_depth(idx):
    mask = mask_cache[f"{idx:05d}"]
    depth = dataset.frames[idx].depth
    masked_depth = depth.copy()
    masked_depth[mask == 0] = 0
    return masked_depth


def load_color(idx):
    mask = mask_cache[f"{idx:05d}"]
    color = dataset.frames[idx].color
    masked_color = color.copy()
    masked_color[mask == 0] = 0
    return masked_color


depths = (load_depth(fid) for fid in frame_ids)
colors = (load_color(fid) for fid in frame_ids)
vbg = tsdf2(depths, colors, frame_poses, dataset.K, dataset.depth_scale, vol_size=0.01)
save_scene(vbg, "run/teddy_tsdf_improved.ply", "mesh")
