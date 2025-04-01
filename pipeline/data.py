"""
参与重建的数据结构
"""

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
from tqdm import tqdm

from utils.convert import downsample, merge_points


class Edge(NamedTuple):
    source_id: int
    target_id: int
    T_ts: np.ndarray
    edge_type: str  # ['loop', 'odometry']

    @property
    def key(self):
        return f"{self.source_id}-{self.target_id}-{self.edge_type}"


class RegFuncType(Protocol):
    def __call__(
        self, sid: int, tid: int, init_T: np.ndarray = np.eye(4)
    ) -> Tuple[bool, np.ndarray]: ...


class Chunk:
    "时序相邻的帧融合"

    def __init__(self, reg_func: Optional[RegFuncType] = None, id=None) -> None:
        """
        reg_func: (source_id, target_id) -> (bool, trans) 用于配准原始视频帧
        """
        self.id = id  # chunk identifier
        self.frame_ids: List = []
        self.frame_poses: List[np.ndarray] = []  # frame2world
        self.edges: List[Edge] = []
        self.meta = {}  # chunk meta info
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
        assert self.register is not None, "register function not set"

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

        return "success"

    def enhance(self):
        "enhance the chunk by adding keyframe edges"
        assert self.register is not None, "register function not set"

        for k in [5, 10, 20, 30, 40, 50, 60]:  # TODO: keyframe selection
            key_frames = self.frame_ids[::k]
            key_poses = self.frame_poses[::k]
            for i in range(1, len(key_frames)):
                sid, tid = key_frames[i], key_frames[i - 1]
                init_pose = np.linalg.inv(key_poses[i - 1]) @ key_poses[i]  # sid -> tid
                flag, trans = self.register(sid, tid, init_pose)
                if flag:
                    self.edges.append(Edge(sid, tid, trans, "skipframe"))

    def _register_skipframe(
        self, sid: int, tid: int, init_pose: np.ndarray, edge_type: str = "skipframe"
    ):
        """
        如果放到 enhance_parallel 内部会导致无法 pickle
        edge_type: "skipframe" or "odomerty",
            如果你希望间隔边发挥更重要的作用消除误差，可以试一下 "odomerty"
        """
        assert self.register is not None, "register function not set"

        flag, trans = self.register(sid, tid, init_pose)
        edge_type = edge_type if flag else "failed"
        return Edge(sid, tid, trans, edge_type)

    def enhance_parallel(
        self, skip_every: Iterable[int] = (5, 10, 20, 30, 40, 50, 60), **kwargs
    ):
        "enhance the chunk by adding keyframe edges"
        assert self.register is not None, "register function not set"

        multi_work = Parallel(n_jobs=-1, backend="multiprocessing")
        tasks = []
        for k in skip_every:  # keyframe selection
            key_frames = self.frame_ids[::k]
            key_poses = self.frame_poses[::k]
            for i in range(1, len(key_frames)):
                sid, tid = key_frames[i], key_frames[i - 1]
                init_pose = np.linalg.inv(key_poses[i - 1]) @ key_poses[i]  # sid -> tid
                tasks.append(
                    delayed(self._register_skipframe)(sid, tid, init_pose, **kwargs)
                )
        edges = multi_work(tasks)
        for e in edges:
            assert isinstance(e, Edge)  # for type hint
            if e.edge_type != "failed":
                self.edges.append(e)

    def transform(self, trans: np.ndarray, inplace: bool = True):
        "transform chunk by a transformation matrix"
        poses = [trans @ p for p in self.frame_poses]
        if inplace:
            self.frame_poses = poses
        return poses

    def gather_data(self):
        "gather data from chunk"
        return {
            "chunk_id": self.id,
            "frame_ids": self.frame_ids,
            "frame_poses": self.frame_poses,
            "edges": self.edges,
            "meta": self.meta,
        }
