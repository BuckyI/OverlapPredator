"""
收集数据集 GICP 配准信息
"""

# %%
import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets.tum import Frame, TUMDataset
from utils.evaluate import evaluate_registration, get_trans_rot, pose_difference
from utils.registration import GICP_registration

# %%
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
all_data = []
for param in tqdm(params):
    dataset = TUMDataset(param[0], param[2])

    for i in tqdm(range(len(dataset.frames)), desc=param[1]):
        # 如果是第一帧，初始化，不需要配准
        if i == 0:
            continue

        # 如果不是第一帧，和上一帧进行配准
        for j in list(range(i - 1, -1, -1))[:10]:  # 和过去的 10 帧进行配准，扩充数据丰富度
            sf, tf = dataset.frames[i], dataset.frames[j]
            s, t = sf.pcd_array, tf.pcd_array
            gt_pose = np.linalg.inv(tf.pose) @ sf.pose
            # GICP registration
            T, _ = GICP_registration(s, t)
            eval_result = evaluate_registration(s, t, T, 0.02)
            trans, rot = get_trans_rot(np.linalg.inv(T) @ gt_pose)
            gt_trans, gt_rot = get_trans_rot(gt_pose)
            data = {
                "dataset": param[1],
                "s_id": i,
                "t_id": j,
                "gt_pose_error": pose_difference(T, gt_pose),
                "trans_error": trans,
                "rot_error": rot,
                "gt_trans": gt_trans,
                "gt_rot": gt_rot,
                "T": T,
            }
            # data.update(eval_result)
            all_data.append(data)

    # 每个数据集处理完毕后，保存一次数据
    df = pd.DataFrame(all_data)
    df.to_csv("output/gicp_reg_data_full.csv", index=False)
