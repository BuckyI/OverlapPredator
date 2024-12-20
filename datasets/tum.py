from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import open3d as o3d
import small_gicp
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class Frame:
    timestamp: float
    color_path: str
    depth_path: str
    pose: np.ndarray  # Twc
    K: np.ndarray
    width: int
    height: int

    @cached_property
    def depth(self) -> np.ndarray:
        return np.asarray(o3d.io.read_image(self.depth_path))

    @cached_property
    def color(self) -> np.ndarray:
        return np.asarray(o3d.io.read_image(self.color_path))

    @property
    def o3d_point_cloud(self) -> o3d.geometry.PointCloud:
        depth_image = o3d.io.read_image(self.depth_path)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.K)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, intrinsic, depth_scale=5000.0, depth_trunc=5
        ).voxel_down_sample(voxel_size=0.01)
        return pcd

    @cached_property
    def pcd_array(self) -> np.ndarray:
        "normalized, unit length 1m"
        return np.asarray(self.o3d_point_cloud.points)

    @cached_property
    def pcd_array_without_plane(self) -> np.ndarray:
        # TODO 如果图片中没有地面怎么办？
        pcd = self.o3d_point_cloud
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return np.asarray(outlier_cloud.points)

    @cached_property
    def pcd_array_005(self) -> np.ndarray:
        "low resolution pcd, unit length 1m"
        pcd, tree = small_gicp.preprocess_points(self.pcd_array, 0.05, num_threads=5)
        return pcd.points()[:, :3]

    @cached_property
    def pcd_array_without_plane_005(self) -> np.ndarray:
        "low resolution pcd, unit length 1m"
        pcd, tree = small_gicp.preprocess_points(self.pcd_array_without_plane, 0.05, num_threads=5)
        return pcd.points()[:, :3]


class TUMDataset:

    calibration = {
        "fr1": [517.306408, 516.469215, 318.643040, 255.313989],
        "fr2": [520.908620, 521.007327, 325.141442, 249.701764],
        "fr3": [535.4, 539.2, 320.1, 247.6],
    }

    def __init__(self, path: str, calibration: str = "fr3"):
        self.id = path.split("/")[-1]  # 使用文件名作为数据集的 id
        self.path = Path(path)
        assert self.path.is_dir(), f"dataset {path} does not exist"

        cal = self.calibration[calibration]
        K = np.array([[cal[0], 0, cal[2]], [0, cal[1], cal[3]], [0, 0, 1]])
        width, height = 640, 480

        associate_file = self.path / "associate_with_groundtruth.txt"
        assert associate_file.is_file(), "require associate_with_groundtruth.txt"
        self.frames: list[Frame] = []
        self.index = associates = open(associate_file, "r").readlines()
        for line in associates:
            info = line.split()
            timestamp = float(info[0])
            color_path = Path(self.path, info[1]).as_posix()
            depth_path = Path(self.path, info[3]).as_posix()

            T = np.eye(4)
            T[:3, :3] = R.from_quat([float(v) for v in info[8:12]]).as_matrix()  # qx qy qz qw
            T[:3, 3] = np.array([float(v) for v in info[5:8]])  # tx ty tz

            frame = Frame(
                timestamp=timestamp,
                color_path=color_path,
                depth_path=depth_path,
                pose=T,
                K=K,
                width=width,
                height=height,
            )
            self.frames.append(frame)

        self.timestamp2frame = {f.timestamp: f for f in self.frames}
        self.timestamps = [f.timestamp for f in self.frames]


def load_point_cloud(frame: Frame):
    depth_image = o3d.io.read_image(frame.depth_path)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(frame.width, frame.height, frame.K)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic, depth_scale=5000.0)  # tum
    return pcd


def visualize_trajectory(dataset: TUMDataset):
    import matplotlib.pyplot as plt

    x = []
    y = []
    z = []
    for f in dataset.frames:
        trans = f.pose[:3, 3]
        x.append(trans[0])
        y.append(trans[1])
        z.append(trans[2])
    ax = plt.subplot(111, projection="3d")
    ax.plot(x, y, z, "b.", markersize=5)
    plt.show()
