"""
snippet to reconstruction
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

from datasets.kinect import KinectDataset
from models.checker import Checker
from models.model import Model, split_data, to_numpy
from utils.convert import average_poses, downsample, merge_points, transform
from utils.evaluate import pose_difference2
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

model = Model()
dataset = KinectDataset("data/pig_kinect_0419.h5")
checker = Checker()

# remove old cache
if not os.path.exists("run"):
    os.makedirs("run")
_ps = ["run/cache.h5", "run/dataset_cache.h5"]
for p in _ps:
    if os.path.exists(p):
        os.remove(p)

cache = CacheSE("run/cache.h5", mode="w")  # overwrite
dataset_cache = DatasetCache("run/dataset_cache.h5", mode="w")

# configure
k = 210  # 30fps  1s -> 30 frames 7s
n = np.ceil(len(dataset) / k)  # n chunks
total_reg = n * (n - 1) // 2
logger.info(f"{total_reg=}, {len(dataset)=}, {k=}, {n=}")


def load_points(frame_id: int):
    "dataset -> cache -> load points"
    global dataset, cache, dataset_cache

    key = f"points/{frame_id}"
    if key in cache.hdf5:
        pcd = cache[key]
    else:
        pcd = dataset[frame_id].pcd_array
        cache[key] = pcd
    return pcd


# %% chunk generate
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


class ChunkState(Enum):
    begin = auto()  # no chunk before, initialize
    progress = auto()  # chunk accumulating
    full = auto()  # chunk success before
    icp_failed = auto()  # chunk break before


chunks = []
# 用于后续全局优化，记录 chunk 之间因时序连接而产生的变换关系
chunk_poses = []
chunk_edges = []

chunk = None
k = k / 2  # 因为跳帧了
state = ChunkState.begin
for i in range(0, 3000, 2):
    # init
    if state == ChunkState.begin:
        chunk = Chunk(registration, id=len(chunks))
        chunks.append(chunk)
        chunk_poses.append(np.eye(4))
    elif state == ChunkState.full:
        chunk = Chunk(registration, id=len(chunks))
        pose = chunk.append_overlap(chunks[-1], ratio=0.4)
        chunks.append(chunk)
        chunk_edges.append(Edge(chunks[-1].id, chunks[-2].id, pose, "odometry"))
        chunk_poses.append(chunk_poses[-1] @ pose)
    elif state == ChunkState.icp_failed:
        assert chunk is not None and chunk.frame_poses
        pose = chunk.frame_poses[-1]  # 使用 last chunk 最后一次配准成功的位姿为参考
        chunk = Chunk(registration, id=len(chunks))
        chunk_edges.append(Edge(chunks[-1].id, chunks[-2].id, pose, "guess"))
        chunks.append(chunk)
        chunk_poses.append(chunk_poses[-1] @ pose)
    else:
        assert state == ChunkState.progress
        # TODO 其他分割 chunk 的思路，如以 chunk 合成点云的点云数目到达10万为限制

    assert chunk is not None
    code = chunk.append(i)
    state = ChunkState.progress
    if code == "icp-failed":
        state = ChunkState.icp_failed
        logger.error("icp failed, lose tracking")
    elif len(chunk.frame_ids) == k:  # chunk full
        state = ChunkState.full
        logger.info("chunk full")

else:
    if state == ChunkState.progress:  # finish loop but chunk not full
        logger.info("chunk not full")

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

    merged_pcd = downsample(merge_points(pcds), 0.02)
    return merged_pcd


pcds = []
for c in chunks:
    c.enhance_parallel()
    c.optimize()
    points = construction(c)
    pcds.append(points)
    logger.info(f"chunk {c.id} done")

cache.dump(chunks, "run/chunks.joblib")
cache.dump(pcds, "run/pcds.joblib")


# %% model registration
def load_chunk_data(chunk_id: int, points: Optional[np.ndarray] = None):
    "load chunk data from cache or compute it. points are need if cache miss."
    global dataset_cache
    if not dataset_cache.exists("chunk", chunk_id):  # 缓存不存在
        assert points is not None, "points is needed if cache miss"
        res, data = model.encode(points)
        if not res:  # 模型编码失败
            logger.error(f"encode chunk {chunk_id} failed")
            raise Exception("encode chunk failed, really unexpected")
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

for tid in range(len(chunks)):
    for sid in range(tid + 1, len(chunks)):
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

# %%
chunks = cache.load("run/chunks.joblib")
chunk_poses = cache.load("run/chunk_poses.joblib")
chunk_edges = cache.load("run/chunk_edges_model.joblib")
pcds = cache.load("run/pcds.joblib")

pose_graph = o3d.pipelines.registration.PoseGraph()
chunk2node = dict()  # map from chunk id to node id
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

global_scene = simple_merge_points(pcds, poses)
show_pcd(global_scene)
save_pcd(global_scene, path="run/scene_simpmerged.ply")

# %% TSDF: get frame poses
chunk_poses = cache.load("run/chunk_poses_optimized.joblib")
chunks = cache.load("run/chunks.joblib")

frame_id_pose = defaultdict(list)
for i in range(len(chunks)):
    c = chunks[i]
    assert isinstance(c, Chunk)
    world_poses = c.transform(chunk_poses[i], inplace=False)
    for i, fid in enumerate(c.frame_ids):
        # if i % 2 != 0:
        #     continue  # reduce the number of frames to integrate

        frame_id_pose[fid].append(world_poses[i])

frame_ids = []
frame_poses = []
for fid, poses in frame_id_pose.items():
    frame_ids.append(fid)
    frame_poses.append(average_poses(poses))

# %% TSDF: integrate frames
timestamps = [dataset.timestamps[fid] for fid in frame_ids]

depths, colors = dataset.batch_load_rgbd_iter(timestamps)
K = dataset.K
vbg = tsdf2(depths, colors, frame_poses, K, dataset.depth_scale, vol_size=0.02)
# pcd = vbg.extract_point_cloud()
# points = pcd.point.positions.numpy()
save_scene(vbg, "run/scene_tsdf.ply", "mesh")
logger.info("finished tsdf reconstruction")
# %%
