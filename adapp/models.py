# -*- coding: utf-8 -*-
"""结构化模型（XGBoost + StandardScaler + 特征顺序）的加载。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import FEATURE_ORDER_PATH, SCALER_PATH, XGB_MODEL_PATH
from .features import ALL_FEATURES, transform_features


@dataclass
class StructuredModel:
    """XGBoost 分类器 + 与之配套的 scaler 和特征顺序。

    三者必须来自同一次训练：scaler 决定数值怎么缩放，feature_order 决定列怎么排，
    任何一个对不上，模型拿到的就是错位的输入。
    """

    model: Any
    scaler: Any
    feature_order: list[str]

    @classmethod
    def load(cls) -> "StructuredModel | None":
        """三个文件齐了才返回实例，缺任何一个都返回 None 并说明缺什么。"""
        missing = [
            str(p) for p in (XGB_MODEL_PATH, SCALER_PATH) if not p.exists()
        ]
        if missing:
            print(f"[结构化模型] 缺少文件，XGB 分支不可用：{missing}")
            print("             跑 `python scripts/train_xgb.py` 可以重新生成。")
            return None

        model = joblib.load(XGB_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        if FEATURE_ORDER_PATH.exists():
            feature_order = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
        else:
            feature_order = list(ALL_FEATURES)
            print(f"[结构化模型] 未找到 {FEATURE_ORDER_PATH.name}，改用默认特征顺序")

        print(f"[结构化模型] 已加载 {XGB_MODEL_PATH.name}（{len(feature_order)} 个特征）")
        return cls(model=model, scaler=scaler, feature_order=feature_order)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化并对齐列顺序，输出可直接喂给模型的 DataFrame。"""
        out = transform_features(df, self.scaler)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out, columns=self.feature_order)
        return out[self.feature_order]

    def predict_proba(self, df_prepared: pd.DataFrame) -> np.ndarray:
        """返回正类（AD）概率，一维数组。"""
        return self.model.predict_proba(df_prepared)[:, 1]

    def as_frame(self, X) -> pd.DataFrame:
        """把任意二维输入统一成带正确列名的 DataFrame（SHAP 需要列名）。"""
        if isinstance(X, pd.DataFrame):
            return X[self.feature_order]
        return pd.DataFrame(np.asarray(X), columns=self.feature_order)
