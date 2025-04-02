"""
三维重建
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
)

import numpy as np
import open3d as o3d
import open3d.core as o3c  # type: ignore
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from utils.convert import average_poses, downsample, merge_points, transform

from .data import Chunk, Edge


def construct_pose_graph(edges: List[Edge]):
    """
    edges: (source frame id, source target id) -> (relative pose, edge type)
    edge type: ['loop', 'odometry']
    注意：相邻顶点必须要有 odometry edge
    注意: frame id 大小代表了时间顺序，id 增加，时间往后。
    return:
        open3d.pipelines.registration.PoseGraph: pose graph
        node_ids(list): node id -> frame id 用于查找数据集中的帧
    """
    node_ids = np.unique(
        [[e.source_id, e.target_id] for e in edges]
    ).tolist()  # node id -> frame id
    # Note: np.unique returns the *sorted* unique elements of an array.
    frame2node = {node_ids[i]: i for i in range(len(node_ids))}

    pose_graph = o3d.pipelines.registration.PoseGraph()

    # 添加顶点
    odometry_edges = dict(
        ((e.source_id, e.target_id), e.T_ts) for e in edges if e.edge_type == "odometry"
    )
    # 尝试构建初始顶点位置
    init_poses = [np.eye(4)]
    for i in range(1, len(node_ids)):
        if (node_ids[i], node_ids[i - 1]) in odometry_edges:
            T_ts = odometry_edges[(node_ids[i], node_ids[i - 1])]
        elif (node_ids[i - 1], node_ids[i]) in odometry_edges:
            T_st = odometry_edges[(node_ids[i - 1], node_ids[i])]
            T_ts = np.linalg.inv(T_st)
        else:
            raise ValueError(
                f"no odometry edge between {node_ids[i - 1]} and {node_ids[i]}"
            )

        init_poses.append(init_poses[-1] @ T_ts)
    assert len(init_poses) == len(node_ids)
    for i in range(len(node_ids)):
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(init_poses[i]))

    # 添加边
    for e in edges:
        edge = o3d.pipelines.registration.PoseGraphEdge(
            frame2node[e.source_id],
            frame2node[e.target_id],
            e.T_ts,
            uncertain=(e.edge_type != "odomerty"),
        )
        pose_graph.edges.append(edge)

    return pose_graph, node_ids


def optimize_pose_graph(pose_graph, verbose: bool = False, thr=0.25):
    """
    thr: edge_prune_threshold, default 0.25
    """
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        # max_correspondence_distance=0.07,
        edge_prune_threshold=thr,
        # preference_loop_closure=0.1,
        reference_node=0,
    )

    # In-place optimization
    if verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    return pose_graph


def save_pose_graph(path, pose_graph, node_ids=None):
    """
    path: folder path to save
    pose_graph: open3d.pipelines.registration.PoseGraph
    node_ids: node id -> frame id
    """
    path = Path(path)
    if not path.exists():
        path.mkdir()
    o3d.io.write_pose_graph(Path(path, "pose_graph.json").as_posix(), pose_graph)
    if node_ids is not None:
        json.dump(node_ids, open(Path(path, "pose_graph_node_ids.json"), "w"))


def load_pose_graph(path):
    pg_path = Path(path, "pose_graph.json")
    node_ids_path = Path(path, "pose_graph_node_ids.json")
    pose_graph = node_ids = None
    if pg_path.exists():
        pose_graph = o3d.io.read_pose_graph(pg_path.as_posix())
    if node_ids_path.exists():
        node_ids = json.load(open(node_ids_path, "r"))
    return pose_graph, node_ids


def tsdf2(
    depths: Iterable[np.ndarray],
    colors: Iterable[np.ndarray],
    poses: Iterable[np.ndarray],
    K: np.ndarray,
    depth_scale: float,
    depth_max: float = 5.0,
    vol_size: float = 3.0 / 512,
) -> o3d.t.geometry.VoxelBlockGrid:
    """
    tsdf from nparray
    PLEASE CHECK DATATYPE:
        depth: uint16
        color: uint8
        K: float64
        depth_scale: float

    return o3d.t.geometry.VoxelBlockGrid
    """
    # 类型检查
    depth_scale = float(depth_scale)
    depth_max = float(depth_max)
    vol_size = float(vol_size)

    device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=vol_size,
        block_resolution=16,
        block_count=5000,
        device=device,
    )

    for i, (depth, color, pose) in tqdm(enumerate(zip(depths, colors, poses))):
        depth = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(depth))
        color = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(color))
        intrinsic = o3d.core.Tensor(K, o3d.core.Dtype.Float64)
        extrinsic = o3d.core.Tensor(np.linalg.inv(pose), o3d.core.Dtype.Float64)

        try:
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, extrinsic, depth_scale, depth_max
            )  # Nx3 tensor
            vbg.integrate(
                frustum_block_coords,
                depth,
                color,
                intrinsic,
                intrinsic,
                extrinsic,
                depth_scale=depth_scale,
                depth_max=depth_max,
            )
        except Exception as e:
            logger.error(f"encounter error: {e}, skip frame {i}")
    return vbg


def tsdf3(
    depths: Iterable[np.ndarray],
    colors: Iterable[np.ndarray],
    poses: Iterable[np.ndarray],
    K: np.ndarray,
    depth_scale: float,
    depth_max: float = 5.0,
    vol_size: float = 3.0 / 512,
    vbg: Optional[o3d.t.geometry.VoxelBlockGrid] = None,
) -> o3d.t.geometry.VoxelBlockGrid:
    """
    continuous tsdf from array
    PLEASE CHECK DATATYPE:
        depth: uint16
        color: uint8
        K: float64
        depth_scale: float

    return o3d.t.geometry.VoxelBlockGrid
    """
    # 类型检查
    depth_scale = float(depth_scale)
    depth_max = float(depth_max)
    vol_size = float(vol_size)

    device = o3d.core.Device("CPU:0")
    if vbg is None:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=vol_size,
            block_resolution=16,
            block_count=5000,
            device=device,
        )

    assert isinstance(vbg, o3d.t.geometry.VoxelBlockGrid)

    for i, (depth, color, pose) in tqdm(enumerate(zip(depths, colors, poses))):
        depth = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(depth))
        color = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(color))
        intrinsic = o3d.core.Tensor(K, o3d.core.Dtype.Float64)
        extrinsic = o3d.core.Tensor(np.linalg.inv(pose), o3d.core.Dtype.Float64)

        try:
            frustum_block_coords = vbg.compute_unique_block_coordinates(  # type: ignore
                depth, intrinsic, extrinsic, depth_scale, depth_max
            )  # Nx3 tensor
            vbg.integrate(  # type: ignore
                frustum_block_coords,
                depth,
                color,
                intrinsic,
                intrinsic,
                extrinsic,
                depth_scale=depth_scale,
                depth_max=depth_max,
            )
        except Exception as e:
            logger.error(f"encounter error: {e}, skip frame {i}")
    return vbg


def save_scene(
    vbg: o3d.t.geometry.VoxelBlockGrid, path: str = "scene.ply", type: str = "pcd"
):
    """
    path: xxx.ply
    type: pcd or mesh
    """
    if type == "pcd":
        pcd = vbg.extract_point_cloud()
        # This is how you can get the points:
        # points = pcd.point.positions.numpy()
        # colors = pcd.point.colors.numpy()
        return o3d.t.io.write_point_cloud(path, pcd, print_progress=True)
    if type == "mesh":
        mesh = vbg.extract_triangle_mesh()
        return o3d.t.io.write_triangle_mesh(path, mesh, print_progress=True)
    raise ValueError("Invalid type")


def save_pcd(
    points: np.ndarray, colors: Optional[np.ndarray] = None, path: str = "pcd.ply"
):
    """
    points: Nx3
    colors: Nx3
    path: xxx.ply
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return o3d.io.write_point_cloud(path, pcd, print_progress=True)


def optimize_chunk(chunk: Chunk):
    "利用 chunk 收集的信息构建 pose graph 并优化"
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for i in chunk.frame_poses:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(i))

    frame2node = {chunk.frame_ids[i]: i for i in range(len(chunk.frame_ids))}
    for e in chunk.edges:
        edge = o3d.pipelines.registration.PoseGraphEdge(
            frame2node[e.source_id],
            frame2node[e.target_id],
            e.T_ts,
            uncertain=(e.edge_type != "odomerty"),
        )
        pose_graph.edges.append(edge)

    pose_graph = optimize_pose_graph(pose_graph)
    chunk.frame_poses = [n.pose for n in pose_graph.nodes]
    logger.info("completed pose graph optimization")


def recon_merge_points(
    chunks: List[Chunk], pose_label: str = "pose_optimized", path="run/merge.ply"
):
    assert all("points" in c.meta for c in chunks)
    assert all(pose_label in c.meta for c in chunks)

    pcds = []
    for i, c in enumerate(chunks):
        # if c.meta["valid_state"] != "valid":  # not valid
        #     continue
        pcds.append(transform(c.meta["points"], c.meta[pose_label]))
    pcd = merge_points(pcds)
    pcd = downsample(pcd, 0.01)

    save_pcd(pcd, path=path)
    return pcd


def teddy_recon_tsdf(
    chunks: List[Chunk],
    pose_label: str,
    path: str,
    tool: dict,
):
    """
    chunks: list[Chunk] 参与重建的 chunk
    pose_label: str, chunk.meta 中的 pose 标签，如 "pose_optimized", "pose", "gt_pose"
    path: str, 保存重建结果的路径
    tool: dict, 重建工具的配置，需要包含读取深度图之类的函数实现
    """
    required_keys = ["load_depth", "load_color"]
    assert all(k in tool for k in required_keys)
    load_depth: Callable = tool["load_depth"]
    load_color: Callable = tool["load_color"]
    dataset = tool["dataset"]

    frame_id_pose = defaultdict(list)  # frame_id -> world poses
    for i, c in enumerate(chunks):
        assert isinstance(c, Chunk)
        # if c.meta["valid_state"] != "valid":
        #     continue
        world_poses = c.transform(c.meta[pose_label], inplace=False)
        for i, fid in enumerate(c.frame_ids):
            frame_id_pose[fid].append(world_poses[i])

    frame_ids = []
    frame_poses = []
    for fid, poses in frame_id_pose.items():
        frame_ids.append(fid)
        frame_poses.append(average_poses(poses))

    depths = (load_depth(fid) for fid in frame_ids)
    colors = (load_color(fid) for fid in frame_ids)
    vbg = tsdf2(
        depths, colors, frame_poses, dataset.K, dataset.depth_scale, vol_size=0.01
    )
    save_scene(vbg, path, "mesh")
    return vbg
