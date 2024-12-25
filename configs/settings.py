"""
方便进行实验，固定下来的参数和设置
"""

_prefix = "/mnt/e/3d-datasets/TUM/"
params = [
    (f"{_prefix}1.TestingAndDebugging/rgbd_dataset_freiburg1_xyz", "fr1_xyz", "fr1"),
    (f"{_prefix}1.TestingAndDebugging/rgbd_dataset_freiburg1_rpy", "fr1_rpy", "fr1"),
    (f"{_prefix}2.HandheldSLAM/rgbd_dataset_freiburg1_desk", "fr1_desk", "fr1"),
    (f"{_prefix}2.HandheldSLAM/rgbd_dataset_freiburg1_desk2", "fr1_desk2", "fr1"),
    (f"{_prefix}2.HandheldSLAM/rgbd_dataset_freiburg1_room", "fr1_room", "fr1"),
    # (f"{_prefix}3.RobotSLAM/rgbd_dataset_freiburg2_pioneer_slam", "fr2_pioneer_slam", "fr2"),
    (f"{_prefix}6.ObjectReconstruction/rgbd_dataset_freiburg1_teddy", "fr1_teddy", "fr1"),
    (f"{_prefix}6.ObjectReconstruction/rgbd_dataset_freiburg1_plant", "fr1_plant", "fr1"),
    (f"{_prefix}6.ObjectReconstruction/rgbd_dataset_freiburg2_flowerbouquet", "fr2_flowerbouquet", "fr2"),
    (f"{_prefix}6.ObjectReconstruction/rgbd_dataset_freiburg3_teddy", "fr3_teddy", "fr3"),
]
dataset_cache_path = "data/database.h5"
