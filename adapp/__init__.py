# -*- coding: utf-8 -*-
"""阿尔茨海默病风险预测应用。

工厂函数 create_app() 负责在启动时加载好模型，之后每个请求直接复用，
不必反复读盘。
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional


def _ensure_utf8_stdout() -> None:
    """日志里有中文，Windows 上重定向输出时默认编码可能是 cp1252，直接 UnicodeEncodeError。

    强制标准输出走 UTF-8；编码不了的字符退化成占位符，也不要让日志把应用打挂。
    要在任何模块打印之前执行，所以放在其余 import 之前。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # 流被替换过或不支持 reconfigure，忽略即可


_ensure_utf8_stdout()
warnings.filterwarnings("ignore")

from flask import Flask  # noqa: E402

from . import config  # noqa: E402
from .explain import ShapExplainer  # noqa: E402
from .features import ALL_FEATURES  # noqa: E402
from .models import StructuredModel  # noqa: E402
from .mri import MriPredictor  # noqa: E402

__version__ = "1.0.0"


@dataclass
class AppContext:
    """一次性加载、全局复用的模型与配置。"""

    structured: Optional[StructuredModel]
    mri: Optional[MriPredictor]
    explainer: Optional[ShapExplainer]
    threshold: float
    threshold_source: str
    class_names: list[str]
    ad_index: int
    feature_order: list[str] = field(default_factory=list)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    structured = StructuredModel.load()
    mri = MriPredictor.load()
    threshold, threshold_source = config.load_decision_threshold()
    class_names, ad_index = config.load_class_names()

    ctx = AppContext(
        structured=structured,
        mri=mri,
        explainer=ShapExplainer(structured) if structured else None,
        threshold=threshold,
        threshold_source=threshold_source,
        class_names=class_names,
        ad_index=ad_index,
        feature_order=structured.feature_order if structured else list(ALL_FEATURES),
    )
    app.extensions["adapp"] = ctx

    _print_startup_banner(ctx)

    from .routes import bp

    app.register_blueprint(bp)
    return app


def _print_startup_banner(ctx: AppContext) -> None:
    """把真正生效的配置打出来。

    旧版本无论有没有读到配置文件都打印 "[SUCCESS] 阈值相关配置已加载"，
    结果阈值一直静默停在 0.5 也没人发现。
    """
    print("=" * 62)
    print(f"  结构化分支 (XGBoost) : {'已就绪' if ctx.structured else '不可用'}")
    print(f"  影像分支 (ResNet50)  : {'已就绪' if ctx.mri else '不可用（仅用结构化数据）'}")
    print(f"  类别                 : {ctx.class_names}（AD 下标 {ctx.ad_index}）")
    print(f"  决策阈值             : {ctx.threshold:.4f}")
    print(f"  阈值来源             : {ctx.threshold_source}")
    print("=" * 62)
