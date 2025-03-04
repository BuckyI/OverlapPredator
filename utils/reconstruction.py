import json
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
import open3d.core as o3c
from joblib import Parallel, delayed
from loguru import logger

from datasets.tum import Frame
from models.checker import Checker
from utils.registration import GICP_registration

from .convert import downsample, merge_points, transform


class Edge(NamedTuple):
    source_id: int
    target_id: int
    T_ts: np.ndarray
    edge_type: str  # ['loop', 'odometry']


class RegFuncType(Protocol):
    def __call__(
        self, sid: int, tid: int, init_T: np.ndarray = np.eye(4)
    ) -> Tuple[bool, np.ndarray]: ...


class Chunk:
    "时序相邻的帧融合"

    def __init__(self, reg_func: RegFuncType, id=None) -> None:
        """
        reg_func: (source_id, target_id) -> (bool, trans) 用于配准原始视频帧
        """
        self.id = id  # chunk identifier
        self.frame_ids: List = []
        self.frame_poses: List[np.ndarray] = []  # frame2world
        self.edges: List[Edge] = []
        self.register = reg_func

    def append_overlap(self, other_chunk: "Chunk", ratio: float = 0.3):
        """
        将 other_chunk 的数据追加到 self 中，用以提升配准效果
        return : np.ndarray, Chunk pose relative to other_chunk
        """
        assert not self.frame_ids  # must be empty
        k = int(len(other_chunk.frame_ids) * ratio)
        self.frame_ids.extend(other_chunk.frame_ids[-k:])
        self.frame_poses.extend(other_chunk.frame_poses[-k:])
        for e in other_chunk.edges:
            if e.source_id in self.frame_ids and e.target_id in self.frame_ids:
                self.edges.append(e)
        pose = self.frame_poses[0].copy()
        self.transform(
            np.linalg.inv(pose)
        )  # turn to eye, make frames relative to first frame
        return pose

    def append(self, idx: int):
        "append next frame, return status"
        if not self.frame_ids:
            self.frame_ids.append(idx)  # or timestamps
            self.frame_poses.append(np.eye(4))
            return "success"

        sid, tid = idx, self.frame_ids[-1]
        flag, trans = self.register(sid, tid)
        if not flag:
            return "icp-failed"

        self.frame_ids.append(idx)
        self.frame_poses.append(self.frame_poses[-1] @ trans)
        self.edges.append(Edge(sid, tid, trans, "odometry"))

        # TODO: 位姿平移阈值、时间阈值
        return "success"

    def optimize(self):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        for i in self.frame_poses:
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(i))

        frame2node = {self.frame_ids[i]: i for i in range(len(self.frame_ids))}
        for e in self.edges:
            edge = o3d.pipelines.registration.PoseGraphEdge(
                frame2node[e.source_id],
                frame2node[e.target_id],
                e.T_ts,
                uncertain=(e.edge_type != "odomerty"),
            )
            pose_graph.edges.append(edge)

        pose_graph = optimize_pose_graph(pose_graph)
        self.frame_poses = [n.pose for n in pose_graph.nodes]
        logger.info("completed pose graph optimization")

    def enhance(self):
        "enhance the chunk by adding keyframe edges"
        for k in [5, 10, 20, 30, 40, 50, 60]:  # TODO: keyframe selection
            key_frames = self.frame_ids[::k]
            key_poses = self.frame_poses[::k]
            for i in range(1, len(key_frames)):
                sid, tid = key_frames[i], key_frames[i - 1]
                init_pose = np.linalg.inv(key_poses[i - 1]) @ key_poses[i]  # sid -> tid
                flag, trans = self.register(sid, tid, init_pose)
                if flag:
                    self.edges.append(Edge(sid, tid, trans, "skipframe"))

    def _register_skipframe(self, sid: int, tid: int, init_pose: np.ndarray):
        "如果放到 enhance_parallel 内部会导致无法 pickle"
        flag, trans = self.register(sid, tid, init_pose)
        edge_type = "skipframe" if flag else "skipframe-failed"
        return Edge(sid, tid, trans, edge_type)

    def enhance_parallel(self):
        "enhance the chunk by adding keyframe edges"
        multi_work = Parallel(n_jobs=-1, backend="multiprocessing")
        tasks = []
        for k in [5, 10, 20, 30, 40, 50, 60]:  # TODO: keyframe selection
            key_frames = self.frame_ids[::k]
            key_poses = self.frame_poses[::k]
            for i in range(1, len(key_frames)):
                sid, tid = key_frames[i], key_frames[i - 1]
                init_pose = np.linalg.inv(key_poses[i - 1]) @ key_poses[i]  # sid -> tid
                tasks.append(delayed(self._register_skipframe)(sid, tid, init_pose))
        edges = multi_work(tasks)
        for e in edges:
            assert isinstance(e, Edge)  # for type hint
            if e.edge_type != "skipframe-failed":
                self.edges.append(e)

    def transform(self, trans: np.ndarray, inplace: bool = True):
        "transform chunk by a transformation matrix"
        poses = [trans @ p for p in self.frame_poses]
        if inplace:
            self.frame_poses = poses
        return poses

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["register"]
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        def empty_func(*args, **kwargs):
            raise NotImplementedError("not available for a unpickled chunk")

        self.register = empty_func


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
                f"no odometry edge between {node_ids[i-1]} and {node_ids[i]}"
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

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, depth_scale, depth_max
        )
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


def extract_pcd(vbg: o3d.t.geometry.VoxelBlockGrid) -> Tuple[np.ndarray, np.ndarray]:
    """
    vbg: o3d.t.geometry.VoxelBlockGrid
    return points, colors
    """
    pcd = vbg.extract_point_cloud()
    points = pcd.point.positions.numpy()
    colors = pcd.point.colors.numpy()
    return points, colors


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
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return o3d.io.write_point_cloud(path, pcd, print_progress=True)


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
