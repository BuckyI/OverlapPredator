"""
对数据库的图片间隔 k 帧提取关键帧并预处理
"""

# %%
from tqdm import tqdm

from datasets.tum import TUMDataset
from models.model import Model, to_numpy
from utils.storage import DatasetCache

# 数据集设定
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


# %%
model = Model()
# database = DatasetCache("data/database.h5", "w")
database = DatasetCache("data/database.h5", "a")

# %%
for param in tqdm(params):
    dataset = TUMDataset(param[0], param[2])

    for i in tqdm(range(len(dataset.frames)), desc=param[1]):
        if i % 5 != 0:
            continue

        frame = dataset.frames[i]
        if database.exists(dataset.id, frame.timestamp):  # 跳过已经处理过的
            continue

        res, data = model.encode(frame.pcd_array)
        if not res:  # 跳过不成功的
            continue

        data = to_numpy(data)
        database.save_frame_data(dataset.id, frame.timestamp, data)
