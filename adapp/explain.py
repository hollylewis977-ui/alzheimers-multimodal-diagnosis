# -*- coding: utf-8 -*-
"""SHAP 可解释性：计算特征贡献值并画图。

三层降级，任何一层不可用都不会让预测接口挂掉：
    1. shap.TreeExplainer（首选，XGBoost 上最快最准）
    2. shap.Explainer（通用回退）
    3. xgboost 自带的 pred_contribs（连 shap 都没装时）
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEBUG_SINGLE_PNG, DEBUG_SUMMARY_PNG, SHAP_BG_PATH
from .models import StructuredModel

# config 已经设好 MPLCONFIGDIR，这里才 import matplotlib
import matplotlib

matplotlib.use("Agg")  # 无 GUI 的服务器环境必须用 Agg 后端
import matplotlib.pyplot as plt  # noqa: E402

try:
    import shap
except ImportError:
    shap = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

TOP_K = 10
SUMMARY_MAX_DISPLAY = 20


def _load_background() -> Optional[np.ndarray]:
    """可选的 SHAP 背景数据集，用于更稳定的期望值估计。"""
    if not SHAP_BG_PATH.exists():
        return None
    try:
        bg = np.load(SHAP_BG_PATH)
        print(f"[SHAP] 已加载背景数据集：{bg.shape}")
        return bg
    except Exception as exc:
        print(f"[SHAP] 背景数据集加载失败，忽略：{exc!r}")
        return None


def _to_2d(values) -> np.ndarray:
    """把各版本 SHAP 的返回值统一成 (n_samples, n_features)。

    不同 shap 版本 / 不同 explainer 会返回 list、二维数组或三维数组
    （三维时中间那一维是类别），这里全部归一化。
    """
    if isinstance(values, list):
        arr = np.stack([np.asarray(v, dtype=float) for v in values], axis=1)
    else:
        arr = np.asarray(values)

    if arr.ndim == 3:
        # 二分类取正类那一层；类别数异常时对类别维取均值兜底
        arr = arr[:, 1, :] if arr.shape[1] > 1 else arr[:, 0, :]
    elif arr.ndim == 1:
        arr = arr[np.newaxis, :]

    return np.asarray(arr, dtype=float)


def _fig_to_base64(save_path=None) -> str:
    """把当前 figure 存成 base64 PNG，顺带落盘一份便于排查。"""
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()

    if save_path is not None:
        try:
            save_path.write_bytes(buf.getvalue())
        except OSError as exc:
            print(f"[SHAP] 调试图落盘失败（不影响返回）：{exc!r}")

    return base64.b64encode(buf.getvalue()).decode("ascii")


def _barh(labels, values, xlabel, title=None) -> None:
    """降级用的水平条形图：按贡献绝对值排序的 Top-K 特征。"""
    plt.figure(figsize=(6.4, 3.8))
    plt.barh(labels, values)
    plt.xlabel(xlabel)
    if title:
        plt.title(title)


class ShapExplainer:
    """围绕一个 StructuredModel 的解释器，explainer 实例懒加载并复用。"""

    def __init__(self, structured: StructuredModel):
        self._structured = structured
        self._background = _load_background()
        self._explainer = None
        self._explainer_ready = False

    # ---------- explainer ----------

    def _get_explainer(self):
        if self._explainer_ready:
            return self._explainer
        self._explainer_ready = True

        if shap is None:
            print("[SHAP] 未安装 shap，将回退到 xgboost pred_contribs")
            return None

        estimator = self._structured.model
        # 如果模型被包在 sklearn Pipeline 里，取最后一步的真实分类器
        if hasattr(estimator, "steps"):
            estimator = estimator.steps[-1][1]

        for name, build in (
            ("TreeExplainer", lambda: shap.TreeExplainer(estimator, data=self._background)
                if self._background is not None else shap.TreeExplainer(estimator)),
            ("Explainer", lambda: shap.Explainer(estimator, self._background)
                if self._background is not None else shap.Explainer(estimator)),
        ):
            try:
                self._explainer = build()
                print(f"[SHAP] 已初始化 {name}")
                return self._explainer
            except Exception as exc:
                print(f"[SHAP] {name} 初始化失败：{exc!r}")

        return None

    @property
    def _expected_value(self) -> float:
        """力图的基准值。取不到就用 0，此时图上的 base value 不具解释意义。"""
        exp = self._explainer
        raw = getattr(exp, "expected_value", None) if exp is not None else None
        if raw is None:
            return 0.0
        arr = np.atleast_1d(np.asarray(raw, dtype=float))
        # 二分类下 expected_value 可能是长度 2 的数组，取正类
        return float(arr[1] if arr.size > 1 else arr[0])

    # ---------- 贡献值 ----------

    def values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """返回 (n_samples, n_features) 的贡献值，全部路径失败则返回 None。"""
        frame = self._structured.as_frame(X)

        exp = self._get_explainer()
        if exp is not None:
            try:
                try:
                    result = exp(frame)
                    raw = getattr(result, "values", result)
                except Exception:
                    # 老版本 shap 只有 shap_values 这个接口
                    raw = exp.shap_values(frame)
                return _to_2d(raw)
            except Exception as exc:
                print(f"[SHAP] 贡献值计算失败，尝试 xgboost 回退：{exc!r}")

        contribs = self._xgb_contribs(frame)
        # pred_contribs 最后一列是 base value，不是特征贡献
        return contribs[:, :-1] if contribs is not None else None

    def _xgb_contribs(self, frame: pd.DataFrame) -> Optional[np.ndarray]:
        if xgb is None:
            return None
        model = self._structured.model
        booster = model.get_booster() if hasattr(model, "get_booster") else None
        if booster is None:
            return None
        try:
            dmat = xgb.DMatrix(
                frame.values,
                feature_names=self._structured.feature_order,
                missing=np.nan,
            )
            return booster.predict(dmat, pred_contribs=True)
        except Exception as exc:
            print(f"[SHAP] xgboost pred_contribs 失败：{exc!r}")
            return None

    def top_features(self, X: pd.DataFrame, k: int = TOP_K) -> list[dict]:
        """单样本贡献最大的 k 个特征，带正负号（正=推高风险）。"""
        values = self.values(X)
        order = self._structured.feature_order

        if values is None:
            # 最后的兜底：用模型自带的全局特征重要性，虽不针对该样本
            importances = getattr(self._structured.model, "feature_importances_", None)
            if importances is None:
                return []
            idx = np.argsort(importances)[::-1][:k]
            return [{"feature": order[i], "value": float(importances[i])} for i in idx]

        row = values[0]
        idx = np.argsort(np.abs(row))[::-1][:k]
        return [{"feature": order[i], "value": float(row[i])} for i in idx]

    # ---------- 画图 ----------

    def single_plot(self, X: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        """单样本解释图：优先 force plot，画不出来就退化成条形图。"""
        values = self.values(X)
        if values is None:
            return None, "shap 与 xgboost pred_contribs 均不可用"

        frame = self._structured.as_frame(X)
        order = self._structured.feature_order
        row = values[0]

        if shap is not None:
            try:
                plt.figure(figsize=(6.4, 3.8))
                shap.force_plot(
                    self._expected_value, row, frame.iloc[0, :],
                    matplotlib=True, show=False,
                )
                return _fig_to_base64(DEBUG_SINGLE_PNG), None
            except Exception as exc:
                print(f"[SHAP] force plot 失败，改用条形图：{exc!r}")

        idx = np.argsort(np.abs(row))[::-1][:TOP_K][::-1]
        _barh([order[i] for i in idx], np.abs(row[idx]), "abs(SHAP)")
        return _fig_to_base64(DEBUG_SINGLE_PNG), None

    def summary_plot(self, X: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        """整批样本的 SHAP summary 图。"""
        values = self.values(X)
        if values is None:
            return None, "shap 与 xgboost pred_contribs 均不可用"

        frame = self._structured.as_frame(X)

        if shap is not None:
            try:
                plt.figure(figsize=(7.2, 4.2))
                shap.summary_plot(values, frame, show=False, max_display=SUMMARY_MAX_DISPLAY)
                return _fig_to_base64(DEBUG_SUMMARY_PNG), None
            except Exception as exc:
                print(f"[SHAP] summary plot 失败，改用条形图：{exc!r}")

        mean_abs = np.mean(np.abs(values), axis=0)
        order = self._structured.feature_order
        idx = np.argsort(mean_abs)[::-1][:SUMMARY_MAX_DISPLAY][::-1]
        plt.figure(figsize=(7.2, 4.2))
        plt.barh([order[i] for i in idx], mean_abs[idx])
        plt.xlabel("mean |SHAP|")
        return _fig_to_base64(DEBUG_SUMMARY_PNG), None
