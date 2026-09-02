# -*- coding: utf-8 -*-
"""集中管理路径、常量与可调参数。

原来这些散落在 app.py 各处（还重复定义了两遍），改动一个路径要翻好几个地方。
所有模块统一从这里取值。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ==== 目录 ====
# config.py 在 adapp/ 下，上一级就是仓库根目录
BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
CONFIG_DIR = ARTIFACTS_DIR / "config"      # 推理必需的小配置，随仓库走
MODELS_DIR = ARTIFACTS_DIR / "models"      # 模型权重，体积大，不入库
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"    # 运行时产物，不入库
REPORTS_DIR = ARTIFACTS_DIR / "reports"    # 训练评估结果，入库
DATA_DIR = BASE_DIR / "data"

# matplotlib 的字体缓存必须在 import matplotlib 之前设好，否则它会写到用户目录
MPL_CACHE_DIR = ARTIFACTS_DIR / ".mpl-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ==== 模型与配置文件 ====
XGB_MODEL_PATH = MODELS_DIR / "alzheimers_xgb_model.pkl"
MRI_RESNET_PATH = MODELS_DIR / "best_model.pth"

SCALER_PATH = CONFIG_DIR / "scaler.pkl"
FEATURE_ORDER_PATH = CONFIG_DIR / "feature_order.json"
FEATURE_SCHEMA_PATH = CONFIG_DIR / "feature_schema.json"
CLASS_NAMES_PATH = CONFIG_DIR / "class_names.json"
THRESHOLD_PATH = CONFIG_DIR / "threshold.json"
SHAP_BG_PATH = CONFIG_DIR / "shap_bg.npy"

# ==== 运行时产物 ====
SHAP_IMAGES_DIR = OUTPUTS_DIR / "shap_individual"
BATCH_SHAP_CSV = OUTPUTS_DIR / "batch_shap_values.csv"
SHAP_IMAGES_ZIP = OUTPUTS_DIR / "shap_individual_images.zip"
DEBUG_SINGLE_PNG = OUTPUTS_DIR / "debug_single_shap.png"
DEBUG_SUMMARY_PNG = OUTPUTS_DIR / "debug_summary_shap.png"

# ==== MRI 推理参数（与 best_model.pth 的训练预处理保持一致）====
IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# ==== 决策阈值 ====
#
# 注意：这里默认是 0.5，是刻意的。
#
# 仓库里存过一份 threshold.json（best_threshold = 0.6049），那是 2025-10-17
# 用 Youden's J 为 **VGG16** 模型算出来的（见 artifacts/reports/mri_vgg16/）。
# 而现在线上跑的是 2025-10-30 训练的 **ResNet50**（best_model.pth），而且阈值
# 最终作用在「MRI 与 XGB 融合之后」的概率上 —— 拿旧模型的阈值套新模型的融合输出
# 没有依据，所以不把它当默认值。
#
# 旧代码把这份文件放在 artifacts/新建文件夹/ 里，而读取路径是 artifacts/ 根目录，
# 于是加载永远失败、静默退回 0.5，启动日志却打印 "[SUCCESS] 阈值相关配置已加载"。
# 现在改成显式的三级优先级，并在启动时打印真正生效的值和来源。
#
# 优先级：环境变量 AD_DECISION_THRESHOLD > artifacts/config/threshold.json > 0.5
DEFAULT_THRESHOLD = 0.5
THRESHOLD_ENV_VAR = "AD_DECISION_THRESHOLD"


def load_decision_threshold() -> tuple[float, str]:
    """返回 (阈值, 来源说明)，来源会打进启动日志，避免再出现静默回退。"""
    raw = os.environ.get(THRESHOLD_ENV_VAR)
    if raw:
        try:
            return _validate_threshold(float(raw)), f"环境变量 {THRESHOLD_ENV_VAR}"
        except (TypeError, ValueError) as exc:
            print(f"[配置] {THRESHOLD_ENV_VAR}={raw!r} 无法解析为阈值：{exc}")

    if THRESHOLD_PATH.exists():
        try:
            payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
            # 兼容两种历史格式：MRI 训练写的 {"best_threshold": x}，
            # XGB train.py 写的 {"selected": "recall90", "values": {...}}
            if "best_threshold" in payload:
                value = float(payload["best_threshold"])
            else:
                value = float(payload["values"][payload["selected"]])
            return _validate_threshold(value), str(THRESHOLD_PATH)
        except Exception as exc:
            print(f"[配置] 读取 {THRESHOLD_PATH} 失败，回退默认阈值：{exc!r}")

    return DEFAULT_THRESHOLD, "默认值（未提供 threshold.json）"


def _validate_threshold(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"阈值必须落在 (0, 1) 区间，收到 {value}")
    return value


def load_class_names() -> tuple[list[str], int]:
    """返回 (类别名列表, AD 所在下标)。文件缺失时退回 ["AD", "CN"]。"""
    names = ["AD", "CN"]
    if CLASS_NAMES_PATH.exists():
        try:
            loaded = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and loaded:
                names = [str(n) for n in loaded]
        except Exception as exc:
            print(f"[配置] 读取 {CLASS_NAMES_PATH} 失败，使用默认类别：{exc!r}")

    ad_index = next(
        (i for i, n in enumerate(names) if "ad" in n.lower() or "demented" in n.lower()),
        0,
    )
    return names, ad_index


# ==== 前端表单：下拉选项 ====
ENUMS = {
    "Gender": [{"label": "Male", "value": 0}, {"label": "Female", "value": 1}],
    "Ethnicity": [
        {"label": "White", "value": 0},
        {"label": "African American", "value": 1},
        {"label": "Asian", "value": 2},
        {"label": "Others", "value": 3},
    ],
    "EducationLevel": [
        {"label": "No education", "value": 0},
        {"label": "Primary school", "value": 1},
        {"label": "Secondary school", "value": 2},
        {"label": "Higher education", "value": 3},
    ],
    "Smoking": [{"label": "Non-smoking", "value": 0}, {"label": "Smoking", "value": 1}],
}
_YES_NO = [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}]

# ==== 前端表单：数值范围（同时用于服务端裁剪，防止越界值喂进模型）====
BOUNDS = {
    "Age": (0, 120),
    "BMI": (10, 60),
    "AlcoholConsumption": (0, 20),
    "PhysicalActivity": (0, 10),
    "DietQuality": (0, 10),
    "SleepQuality": (4, 10),
    "SystolicBP": (70, 250),
    "DiastolicBP": (40, 150),
    "CholesterolTotal": (100, 400),
    "CholesterolLDL": (0, 300),
    "CholesterolHDL": (10, 120),
    "CholesterolTriglycerides": (30, 800),
    "MMSE": (0, 30),
    "FunctionalAssessment": (0, 10),
    "ADL": (0, 10),
}

BINARY_COLS = [
    "MemoryComplaints",
    "BehavioralProblems",
    "Confusion",
    "Disorientation",
    "PersonalityChanges",
    "DifficultyCompletingTasks",
    "Forgetfulness",
]

for _col in BINARY_COLS:
    ENUMS[_col] = list(_YES_NO)

# 批量 CSV 里用来识别患者编号的候选列名，按优先级排列
ID_COLUMN_CANDIDATES = ["PatientID", "patient_id", "ID", "Id", "id"]


def clamp(name: str, value: float) -> float:
    """把数值裁剪回 BOUNDS 允许的范围；未定义范围的特征原样返回。"""
    low, high = BOUNDS.get(name, (None, None))
    if low is not None:
        value = max(value, low)
    if high is not None:
        value = min(value, high)
    return value
