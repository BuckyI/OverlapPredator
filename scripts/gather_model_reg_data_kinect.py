"""
对关键帧配准并收集配准信息(kinect 数据，无 gt pose)
注意：耗时非常久（阈值比较宽松）
"""

# %%
import os

import numpy as np
from loguru import logger
from tqdm import tqdm

from datasets.kinect import KinectDataset
from models.model import Model, to_numpy
from utils.convert import transform
from utils.evaluate import (
    chamfer_distance,
    chamfer_distance_feat,
    cosine_similarity,
    evaluate_registration,
)
from utils.storage import DatasetCache, RunCache

model = Model()
dataset = KinectDataset("data/pig_kinect_0419.h5")
runcache = RunCache("output/pig_kinect_0419.h5", "a", description="kinect 关键帧两两配准")
timestamps = [t for t in dataset.timestamps if dataset.hdf5[str(t)].attrs.get("type", "") == "keyframe"]

# %%
logger.add("output/{time}.log")

total = len(timestamps) * (len(timestamps) - 1) // 2
count = 0
bar = tqdm(total=total)
for i, t1 in enumerate(timestamps):
    for t2 in timestamps[i + 1 :]:
        count += 1

        # assert runcache.index.get((dataset.id, t2, t1)) is None, "该程序从新创建，应该没有重复的"
        if runcache.index.get((dataset.dataset_id, t2, t1)) is not None:  # 跳过重复的，程序可以继续上次运行
            bar.update(1)
            continue

        logger.debug(f"processing {t2} -> {t1} ({count}/{total})")
        # t1: target, t2: source
        meta = {}
        meta["source_timestamp"] = t2
        meta["target_timestamp"] = t1
        meta["dataset_id"] = dataset.dataset_id

        g1 = dataset.hdf5[str(t1)]
        g2 = dataset.hdf5[str(t2)]

        pcd1 = g1["points"]["0"]
        pcd2 = g2["points"]["0"]

        meta["similarity"] = cosine_similarity(
            np.asarray(g1["features"]["3"]).max(axis=0),
            np.asarray(g2["features"]["3"]).max(axis=0),
        )
        meta["similarity_final"] = cosine_similarity(
            np.asarray(g1["final_feature"]).max(axis=0),
            np.asarray(g2["final_feature"]).max(axis=0),
        )
        source_range = np.linalg.norm(pcd2, axis=1)
        target_range = np.linalg.norm(pcd1, axis=1)
        source_hist, _ = np.histogram(source_range, range=(0, 3), bins=100, density=True)
        target_hist, _ = np.histogram(target_range, range=(0, 3), bins=100, density=True)
        meta["range_similarity"] = cosine_similarity(source_hist, target_hist)

        _sp1, _sf1 = np.asarray(g1["points"]["3"]), np.asarray(g1["features"]["3"])
        _sp2, _sf2 = np.asarray(g2["points"]["3"]), np.asarray(g2["features"]["3"])
        meta["chamfer_distance_before"] = chamfer_distance_feat(_sp1, _sp2, _sf1, _sf2)

        # 判断要不要执行耗时的配准操作
        _pd = meta["chamfer_distance_before"]
        _td = abs(t2 - t1) / 1e6  # 间隔时间秒数
        if _pd < 51:
            meta["edge_type"] = "CD 差异小于 51"
        else:
            meta["edge_type"] = "可能不是匹配的边"
            runcache.log_step(meta)
            bar.update(1)
            continue

        logger.debug(f"registering {t2} -> {t1}")
        d1 = dataset.cache.load_frame_data(dataset.dataset_id, t1)
        d2 = dataset.cache.load_frame_data(dataset.dataset_id, t2)
        _, inputs = model.pair_decode(d2, d1)
        reg_result = model.registration_(inputs, debug=True)

        T = reg_result["T"]
        meta["transformation"] = T

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
