# -*- coding: utf-8 -*-
"""结构化特征的定义与预处理（原 utils.py）。

训练脚本和推理服务共用这一份，保证「训练时怎么标准化，推理时就怎么标准化」。
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_SCHEMA_PATH

# 默认 schema：与 alzheimers_disease_patient_data.csv 的列一致。
# 如果 artifacts/config/feature_schema.json 存在，则以文件为准。
_DEFAULT_SCHEMA = {
    "ALL_FEATURES": [
        "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
        "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
        "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL",
        "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
        "FunctionalAssessment", "ADL", "MemoryComplaints", "BehavioralProblems",
        "Confusion", "Disorientation", "PersonalityChanges",
        "DifficultyCompletingTasks", "Forgetfulness",
    ],
    # 只有数值列参与标准化；类别列是整数编码，标准化反而会破坏其语义
    "NUMERIC_COLS": [
        "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
        "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal",
        "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
        "FunctionalAssessment", "ADL",
    ],
    "CATEGORICAL_COLS": [
        "Gender", "Ethnicity", "EducationLevel", "Smoking", "MemoryComplaints",
        "BehavioralProblems", "Confusion", "Disorientation", "PersonalityChanges",
        "DifficultyCompletingTasks", "Forgetfulness",
    ],
    "TARGET_COL": "Diagnosis",
}

if FEATURE_SCHEMA_PATH.exists():
    _schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
else:
    _schema = _DEFAULT_SCHEMA

ALL_FEATURES: list[str] = _schema["ALL_FEATURES"]
NUMERIC_COLS: list[str] = _schema["NUMERIC_COLS"]
CATEGORICAL_COLS: list[str] = _schema["CATEGORICAL_COLS"]
TARGET_COL: str = _schema["TARGET_COL"]


def fit_scaler(df: pd.DataFrame, path) -> StandardScaler:
    """在数值列上拟合 StandardScaler 并落盘，供推理时复用。"""
    scaler = StandardScaler()
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    if cols:
        scaler.fit(df[cols])
    joblib.dump(scaler, path)
    return scaler


def transform_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """补齐缺失列、标准化数值列，并按 ALL_FEATURES 的顺序返回。

    列顺序必须和训练时一致，否则模型会把 Gender 当成 Age 来用。
    """
    df = df.copy()
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    if cols:
        df[cols] = scaler.transform(df[cols])
    return df[ALL_FEATURES]
