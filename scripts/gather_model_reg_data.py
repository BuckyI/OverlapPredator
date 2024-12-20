"""
对关键帧配准并收集配准信息
"""

# %%
import numpy as np
from loguru import logger
from tqdm import tqdm

from datasets.tum import TUMDataset
from models.model import Model, to_numpy
from utils.convert import transform
from utils.evaluate import (
    chamfer_distance,
    chamfer_distance_feat,
    cosine_similarity,
    evaluate_registration,
    get_trans_rot,
    pose_difference2,
)
from utils.storage import DatasetCache, RunCache

# 数据集设定
_prefix = "/mnt/e/3d-datasets/TUM/"
params = [
    (f"{_prefix}1.TestingAndDebugging/rgbd_dataset_freiburg1_xyz", "fr1_xyz", "fr1"),
    (f"{_prefix}1.TestingAndDebugging/rgbd_dataset_freiburg1_rpy", "fr1_rpy", "fr1"),
    (f"{_prefix}2.HandheldSLAM/rgbd_dataset_freiburg1_desk", "fr1_desk", "fr1"),
    (f"{_prefix}2.HandheldSLAM/rgbd_dataset_freiburg1_desk2", "fr1_desk2", "fr1"),
    (f"{_prefix}2.HandheldSLAM/rgbd_dataset_freiburg1_room", "fr1_room", "fr1"),
    # (f"{_prefix}3.RobotSLAM/rgbd_dataset_freiburg2_pioneer_slam", "fr2_pioneer_slam", "fr2"),  # no!
    (f"{_prefix}6.ObjectReconstruction/rgbd_dataset_freiburg1_teddy", "fr1_teddy", "fr1"),
    (f"{_prefix}6.ObjectReconstruction/rgbd_dataset_freiburg1_plant", "fr1_plant", "fr1"),
    (f"{_prefix}6.ObjectReconstruction/rgbd_dataset_freiburg2_flowerbouquet", "fr2_flowerbouquet", "fr2"),
    (f"{_prefix}6.ObjectReconstruction/rgbd_dataset_freiburg3_teddy", "fr3_teddy", "fr3"),
]

model = Model()
database = DatasetCache("data/database.h5", "r")

# %%
logger.add("output/{time}.log")

for param in params:
    dataset = TUMDataset(param[0], param[2])
    logger.info(f"processing {dataset.id}")

    runcache = RunCache(f"output/dataset_run_cache/{dataset.id}.h5", "a", description="关键帧两两配准，收集配准信息")
    timestamps = database.get_frame_timestamps(dataset.id)

    total = len(timestamps) * (len(timestamps) - 1) // 2
    count = 0
    bar = tqdm(total=total, desc=dataset.id)
    for i, t1 in enumerate(timestamps):
        for t2 in timestamps[i + 1 :]:
            count += 1

            # assert runcache.index.get((dataset.id, t2, t1)) is None, "该程序从新创建，应该没有重复的"
            if runcache.index.get((dataset.id, t2, t1)) is not None:  # 跳过重复的，程序可以继续上次运行
                bar.update(1)
                continue

            logger.debug(f"processing {t2} -> {t1} ({count}/{total})")
            # t1: target, t2: source
            meta = {}
            meta["source_timestamp"] = t2
            meta["target_timestamp"] = t1
            meta["dataset_id"] = dataset.id

            f1 = dataset.timestamp2frame[t1]
            f2 = dataset.timestamp2frame[t2]
            meta["gt_transformation"] = gt_T = np.linalg.inv(f1.pose) @ f2.pose
            trans, rot = get_trans_rot(gt_T)
            meta["gt_pose_difference"] = pose_difference2(f1.pose, f2.pose)
            meta["gt_trans"] = trans
            meta["gt_rot"] = rot

            g1 = database.get_frame_group(dataset.id, t1)
            g2 = database.get_frame_group(dataset.id, t2)
            meta["similarity"] = cosine_similarity(
                np.asarray(g1["features"]["3"]).max(axis=0),
                np.asarray(g2["features"]["3"]).max(axis=0),
            )
            meta["similarity_final"] = cosine_similarity(
                np.asarray(g1["final_feature"]).max(axis=0),
                np.asarray(g2["final_feature"]).max(axis=0),
            )
            source_range = np.linalg.norm(f2.pcd_array, axis=1)
            target_range = np.linalg.norm(f1.pcd_array, axis=1)
            source_hist, _ = np.histogram(source_range, range=(0, 3), bins=100, density=True)
            target_hist, _ = np.histogram(target_range, range=(0, 3), bins=100, density=True)
            meta["range_similarity"] = cosine_similarity(source_hist, target_hist)

            # 判断要不要执行耗时的配准操作
            _pd = meta["gt_pose_difference"]
            _td = abs(t2 - t1)
            if _pd < 0.5:
                meta["edge_type"] = "位姿差异小于 0.5"
            elif _td > 10 and _pd < 1:
                meta["edge_type"] = "时间差大于 10 且位姿差异小于 1"
            else:
                meta["edge_type"] = "可能不是匹配的边"
                runcache.log_step(meta)
                bar.update(1)
                continue

            logger.debug(f"registering {t2} -> {t1}")
            d1 = database.load_frame_data(dataset.id, t1)
            d2 = database.load_frame_data(dataset.id, t2)
            _, inputs = model.pair_decode(d2, d1)
            reg_result = model.registration_(inputs, debug=True)

            T = reg_result["T"]
            meta["transformation"] = T
            meta["pose_difference"] = pose_difference2(T, meta["gt_transformation"])

            # source_raw, target_raw 实际上就是 frame2, frame1 的 pcd_array
            source_raw, target_raw = reg_result["source_raw"], reg_result["target_raw"]
            source_down, target_down = reg_result["source"], reg_result["target"]
            source_id, target_id = reg_result["source_id"], reg_result["target_id"]
            # 降采样的索引
            meta["source_id"] = source_id
            meta["target_id"] = target_id
            meta["source_down"] = source_down
            meta["target_down"] = target_down
            meta["source_feats_down"] = reg_result["source_feats"]
            meta["target_feats_down"] = reg_result["target_feats"]
            meta["source_scores_down"] = reg_result["source_scores"]
            meta["target_scores_down"] = reg_result["target_scores"]

            meta["chamfer_distance_before"] = chamfer_distance(source_raw, target_raw)
            meta["chamfer_distance_after"] = chamfer_distance(transform(source_raw, T), target_raw)
            meta["chamfer_distance_after_sampled"] = chamfer_distance(transform(source_down, T), target_down)

            # eval_ = evaluate_registration(source_raw, target_raw, T, resolution=0.02)
            eval_result = evaluate_registration(source_raw, target_raw, T, resolution=0.02)
            meta["eval_fitness"] = eval_result["fitness"]
            meta["eval_source_point_size"] = eval_result["source_point_size"]
            meta["eval_target_point_size"] = eval_result["target_point_size"]
            meta["eval_inlier_rmse"] = eval_result["inlier_rmse"]
            meta["eval_inlier_num"] = eval_result["inlier_num"]

            eval_result = evaluate_registration(source_down, target_down, T, resolution=0.02)
            meta["eval_fitness_down"] = eval_result["fitness"]
            meta["eval_source_point_size_down"] = eval_result["source_point_size"]
            meta["eval_target_point_size_down"] = eval_result["target_point_size"]
            meta["eval_inlier_rmse_down"] = eval_result["inlier_rmse"]
            meta["eval_inlier_num_down"] = eval_result["inlier_num"]

            meta["chamfer_distance_after_feat_down"] = chamfer_distance_feat(
                reg_result["source"],
                reg_result["target"],
                reg_result["source_feats"],
                reg_result["target_feats"],
                reg_result["T"],
            )
            _fs = np.asarray(g2["final_feature"])
            _ft = np.asarray(g1["final_feature"])
            # 之前 final feature 忘记取一半了，所以这里取出来
            _fs = _fs[: _fs.shape[0] // 2]
            _ft = _ft[: _ft.shape[0] // 2]
            meta["chamfer_distance_after_feat"] = chamfer_distance_feat(
                source_raw,
                target_raw,
                _fs,
                _ft,
                reg_result["T"],
            )
            runcache.log_step(meta)
            bar.update(1)
    bar.close()
