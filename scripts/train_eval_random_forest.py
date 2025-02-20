"""
加载事先缓存的数据集配准数据，训练随机森林模型，测试随机森林表现。
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from configs.settings import dataset_cache_path, params
from datasets.tum import TUMDataset
from models.checker import train_random_forest
from utils.evaluate import show_pr_curve, show_roc_curve
from utils.storage import DatasetCache, RunCache

# 数据集设定
database = DatasetCache(dataset_cache_path, "r")
# %%
data = []
for param in params:
    # 因为一些原因，跳过 rgbd_dataset_freiburg3_teddy
    if param[1] == "fr3_teddy":
        continue

    dataset = TUMDataset(param[0], param[2])
    runcache = RunCache(f"data/dataset_run_cache/{dataset.id}.h5", "r", description="关键帧两两配准，收集配准信息")
    hdf5 = runcache.hdf5
    for k in tqdm(hdf5):
        g = hdf5[k]
        d = dict(g.attrs)
        if g.attrs["edge_type"] != "可能不是匹配的边":
            data.append(d)
df = pd.DataFrame(data)
print(df.keys())

# %%

# # 使用部分数据集训练，用于测试泛化性
# select_dataset = [
#     "rgbd_dataset_freiburg1_xyz",
#     "rgbd_dataset_freiburg1_rpy",
#     "rgbd_dataset_freiburg1_desk",
#     "rgbd_dataset_freiburg1_desk2",
#     "rgbd_dataset_freiburg1_room",
# ]
# cond = (df["edge_type"] != "可能不是匹配的边") & (df["dataset_id"].isin(select_dataset))

# cond = (df["edge_type"] != "可能不是匹配的边") & (df["dataset_id"] == "rgbd_dataset_freiburg3_teddy")

cond = df["edge_type"] != "可能不是匹配的边"
data = df[cond].copy()
data["label"] = data["pose_difference"] > 1
feat_columns = [
    # "eval_inlier_rmse_down",
    # "eval_source_point_size_down",
    # "eval_target_point_size_down",
    "eval_fitness_down",
    "chamfer_distance_after_sampled",
    "chamfer_distance_after_feat_down",
]
X = data[feat_columns].to_numpy()
Y = data["label"].to_numpy()
# %% 训练随机森林，并分析效果
rf = train_random_forest(X, Y)

# 打印特征的重要性
index = np.argsort(rf.feature_importances_)[::-1]
for i in index:
    print(feat_columns[i], rf.feature_importances_[i])

pred = rf.predict_proba(X)[:, 1]
show_pr_curve(Y, pred)
# show_roc_curve(Y, pred)

pred = rf.predict(X)
data["predict_label"] = pred
fig = px.scatter(data, x="gt_pose_difference", y="pose_difference", color=pred)
fig.update_traces(marker_size=2)
fig.show()

# %% 在未使用的数据集上测试泛化性
# cond = ~df["dataset_id"].isin(select_dataset)
test_data = df[cond].copy()
test_data["label"] = test_data["pose_difference"] > 1
testX = test_data[feat_columns].to_numpy()
testY = test_data["label"].to_numpy()
pred = rf.predict_proba(testX)[:, 1]
show_pr_curve(testY, pred)

predY = rf.predict(testX)
print("准确率(accuracy): {:.2f}%".format(accuracy_score(testY, predY) * 100))
print("精度(precision): {:.2f}%".format(precision_score(testY, predY) * 100))
print("召回率(recall): {:.2f}%".format(recall_score(testY, predY) * 100))
print("F1分数: {:.2f}%".format(f1_score(testY, predY) * 100))

# %% 使用全部数据训练模型
rf = RandomForestClassifier(n_estimators=200, random_state=42, oob_score=True)
rf.fit(X, Y)
y_test = Y
y_pred = rf.predict(X)

print("准确率(accuracy): {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("精度(precision): {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("召回率(recall): {:.2f}%".format(recall_score(y_test, y_pred) * 100))
print("F1分数: {:.2f}%".format(f1_score(y_test, y_pred) * 100))

# %%
import joblib

joblib.dump(rf, "output/rf_cls_reg_fail_v1.pkl")
