from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, TypedDict

import numpy as np
import open3d as o3d
import open3d.core as o3c

from datasets.tum import Frame

from .convert import downsample, merge_points, transform


class Edge(NamedTuple):
    source_id: int
    target_id: int
    T_ts: np.ndarray
    edge_type: str


def construct_pose_graph(edges: List[Edge]):
    """
    edges: (source frame id, source target id) -> (relative pose, edge type)
    edge type: ['loop', 'odometry']
    注意：相邻顶点必须要有 odometry edge
    return:
        open3d.pipelines.registration.PoseGraph: pose graph
        node_ids(list): node id -> frame id 用于查找数据集中的帧
    """
    node_ids = np.unique([[e.source_id, e.target_id] for e in edges]).tolist()  # node id -> frame id
    # Note: np.unique returns the *sorted* unique elements of an array.
    frame2node = {node_ids[i]: i for i in range(len(node_ids))}

    pose_graph = o3d.pipelines.registration.PoseGraph()

    # 添加顶点
    odometry_edges = dict(((e.source_id, e.target_id), e.T_ts) for e in edges if e.edge_type == "odometry")
    # 尝试构建初始顶点位置
    init_poses = [np.eye(4)]
    for i in range(1, len(node_ids)):
        T_ts = odometry_edges[(node_ids[i], node_ids[i - 1])]
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
            uncertain=(e.edge_type == "odomerty"),
        )
        pose_graph.edges.append(edge)

    return pose_graph, node_ids


def optimize_pose_graph(pose_graph, verbose: bool = False):
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        # max_correspondence_distance=0.07,
        edge_prune_threshold=0.25,
        # preference_loop_closure=0.1,
        reference_node=0,
    )

    # In-place optimization
    if verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    return pose_graph


def tsdf(
    frames: List[Frame],
    poses: List[np.ndarray],
    vol_size: float = 3.0 / 512,
    depth_scale: float = 5000.0,
    depth_max: float = 5.0,
):
    """
    return o3d.t.geometry.VoxelBlockGrid
    """
    device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=vol_size,
        block_resolution=16,
        block_count=50000,
        device=device,
    )
    for frame, pose in zip(frames, poses):
        depth = o3d.t.io.read_image(frame.depth_path).to(device)
        color = o3d.t.io.read_image(frame.color_path).to(device)
        intrinsic = o3d.core.Tensor(frame.K, o3d.core.Dtype.Float64)
        extrinsic = o3d.core.Tensor(np.linalg.inv(pose), o3d.core.Dtype.Float64)

        frustum_block_coords = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic, depth_scale, depth_max)
        # Nx3 tensor
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

    return vbg


def tsdf2(
    depths: List[np.ndarray],
    colors: List[np.ndarray],
    poses: List[np.ndarray],
    K: np.ndarray,
    depth_scale: float,
    depth_max: float = 5.0,
    vol_size: float = 3.0 / 512,
):
    """
    tsdf from nparray
    PLEASE CHECK DATATYPE:
        depth: uint16
        color: uint8
        K: float64
        depth_scale: float

    return o3d.t.geometry.VoxelBlockGrid
    """
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
    for depth, color, pose in zip(depths, colors, poses):
        depth = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(depth))
        color = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(color))
        intrinsic = o3d.core.Tensor(K, o3d.core.Dtype.Float64)
        extrinsic = o3d.core.Tensor(np.linalg.inv(pose), o3d.core.Dtype.Float64)

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

    return vbg


def save_scene(vbg: o3d.t.geometry.VoxelBlockGrid, path: str = "scene.ply", type: str = "pcd"):
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


def simple_merge_points(points: List[np.ndarray], poses: List[np.ndarray]):
    """
    根据点和对应的位姿进行合并并降采样
    适用于快速验证位姿是否准确
    """
    pcds = []
    for pcd, pose in zip(points, poses):
        pcd = transform(pcd, pose)
        pcds.append(pcd)
    pcd = merge_points(pcds)
    pcd = downsample(pcd, 0.01)
    return pcd
