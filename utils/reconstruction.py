from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
import open3d.core as o3c

from datasets.tum import Frame


def tsdf(frames: List[Frame], poses: List[np.ndarray], vol_size: float = 3.0 / 512):
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

        frustum_block_coords = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 5000.0, 5)  # Nx3 tensor
        vbg.integrate(
            frustum_block_coords,
            depth,
            color,
            intrinsic,
            intrinsic,
            extrinsic,
            depth_scale=5000.0,
            depth_max=3.0,
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
