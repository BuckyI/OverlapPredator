"""
结合配准结果进行检查和预测的模型
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from utils.evaluate import evaluate_registration


def check_registration(sp: np.ndarray, tp: np.ndarray, trans: np.ndarray = np.eye(4)) -> bool:
    """
    根据 source, target 点云以及配准位姿变换结果，评估是否有可能是配准失败的错误边。
    使用场景：逐帧 GICP 配准评估（阈值在此场景下进行测试和验证）
    return bool indicate whether the registration is valid
    """
    result = evaluate_registration(sp, tp, trans)
    return (
        result["inliner_rmse"] < 0.02
        and result["source_point_size"] > 100
        and result["target_point_size"] > 100
        and result["fitness"] > 0.65
    )


def train_random_forest(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, oob_score=True)
    logger.info("开始训练随机森林分类器")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    logger.info("准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    logger.info("召回率: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
    logger.info("F1分数: {:.2f}%".format(f1_score(y_test, y_pred) * 100))
    return rf


@dataclass
class Checker:
    model_path: str = "weights/rf_cls_reg_fail.pkl"

    def __post_init__(self):
        assert Path(self.model_path).is_file(), f"{self.model_path} does not exists"

        self.model = joblib.load(self.model_path)
        logger.trace(f"model loaded from {self.model_path}: {self.model}")

    def predict(self, X: np.ndarray):
        """
        根据输入获得预测结果
        注意输入特征：
        "similarity",
        "range_similarity",
        "eval_fitness_down",
        "eval_source_point_size_down",
        "eval_target_point_size_down",
        "chamfer_distance_after_sampled",
        "chamfer_distance_after_feat_down",
        """
        if X.shape == (7,):  # 只有一个输入
            X = X.reshape(1, -1)
        return self.model.predict(X)
