"""
结合配准结果进行检查和预测的模型
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple

import joblib
import numpy as np
from loguru import logger

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

