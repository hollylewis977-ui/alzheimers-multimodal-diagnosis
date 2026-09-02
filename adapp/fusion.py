# -*- coding: utf-8 -*-
"""双模型融合：按各自的置信度加权。"""
from __future__ import annotations

from typing import Optional

# 防止两个模型都恰好输出 0.5 时除以零
_EPSILON = 1e-6


def fuse(p_mri: Optional[float], p_xgb: Optional[float]) -> Optional[float]:
    """把 MRI 与 XGB 的 AD 概率融合成一个。

    以「离 0.5 有多远」作为置信度：越远说明模型越笃定，权重越高。
    只有一个模型可用时直接返回它；都不可用返回 None。

        conf_i   = |p_i - 0.5|
        w_i      = conf_i / (conf_mri + conf_xgb + eps)
        p_fused  = w_mri * p_mri + w_xgb * p_xgb
    """
    if p_mri is None and p_xgb is None:
        return None
    if p_mri is None:
        return p_xgb
    if p_xgb is None:
        return p_mri

    conf_mri = abs(p_mri - 0.5)
    conf_xgb = abs(p_xgb - 0.5)
    total = conf_mri + conf_xgb + _EPSILON

    return (conf_mri / total) * p_mri + (conf_xgb / total) * p_xgb
