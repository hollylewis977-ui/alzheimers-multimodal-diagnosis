# -*- coding: utf-8 -*-
"""HTTP 路由层。业务逻辑都在 models / mri / fusion / explain 里，这里只负责编排。"""
from __future__ import annotations

import io
import shutil
import zipfile
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, render_template, request, send_file

from . import config
from .config import (
    BATCH_SHAP_CSV,
    BINARY_COLS,
    DEBUG_SINGLE_PNG,
    DEBUG_SUMMARY_PNG,
    ID_COLUMN_CANDIDATES,
    SHAP_IMAGES_DIR,
    SHAP_IMAGES_ZIP,
)
from .features import NUMERIC_COLS
from .fusion import fuse

bp = Blueprint("adapp", __name__)

# 批量预测时每行单独出一张 SHAP 图，1000 行就是 1000 个文件。
# 默认关闭，需要时在表单里勾选（对应 shap_images 字段）。
_TRUTHY = {"1", "true", "on", "yes"}


def _ctx():
    """取出 create_app 时装好的模型上下文。"""
    return current_app.extensions["adapp"]


def _wants_shap_images() -> bool:
    return str(request.form.get("shap_images", "")).lower() in _TRUTHY


def _resolve_id_column(df: pd.DataFrame) -> tuple[list[str], str]:
    """找出患者编号列；找不到就用行号，返回 (编号列表, 列名)。"""
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return df[candidate].astype(str).tolist(), candidate
    return [str(i) for i in range(1, len(df) + 1)], "Row"


def _read_csv_upload() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    file = request.files.get("file")
    if not file or not file.filename:
        return None, "没有上传文件"
    try:
        return pd.read_csv(file), None
    except Exception as exc:
        return None, f"CSV 解析失败：{exc}"


def _missing_columns(df: pd.DataFrame, feature_order: list[str]) -> list[str]:
    return [c for c in feature_order if c not in df.columns]


# ============================ 页面 ============================

@bp.route("/")
def index():
    """渲染表单。meta 告诉前端每个特征该用什么控件、什么范围。"""
    feature_order = _ctx().feature_order
    meta = {}
    for name in feature_order:
        if name in NUMERIC_COLS:
            low, high = config.BOUNDS.get(name, (None, None))
            meta[name] = {"kind": "number", "min": low, "max": high, "step": 0.01}
        elif name in BINARY_COLS:
            meta[name] = {"kind": "binary"}
        else:
            meta[name] = {"kind": "int", "min": 0, "max": 5, "step": 1}
    return render_template("index.html", features=feature_order, meta=meta)


# ============================ 单样本预测 ============================

@bp.route("/predict", methods=["POST"])
def predict():
    ctx = _ctx()
    if ctx.structured is None:
        return jsonify(error="XGBoost 模型未加载，请先运行 scripts/train_xgb.py"), 503

    payload = request.get_json(silent=True) or request.form.to_dict()

    row = {}
    for name in ctx.feature_order:
        try:
            value = float(payload.get(name, 0))
        except (TypeError, ValueError):
            value = 0.0
        row[name] = config.clamp(name, value)

    prepared = ctx.structured.prepare(pd.DataFrame([row], columns=ctx.feature_order))
    p_xgb = float(ctx.structured.predict_proba(prepared)[0])

    # MRI 是可选输入：没传、模型没加载、文件有问题，都只是让融合退化成纯 XGB
    p_mri, mri_message = None, None
    upload = request.files.get("mri_zip")
    if upload and upload.filename:
        if ctx.mri is None:
            mri_message = "MRI 模型未加载，本次仅使用结构化数据"
        else:
            p_mri, mri_message = ctx.mri.predict_upload(upload)

    p_final = fuse(p_mri, p_xgb)
    if p_final is None:
        p_final = p_xgb

    is_ad = p_final >= ctx.threshold
    label = ctx.class_names[ctx.ad_index if is_ad else 1 - ctx.ad_index]

    shap_png, shap_error = ctx.explainer.single_plot(prepared.iloc[[0]])

    response = {
        "probability": p_final,
        "prediction": int(is_ad),
        "prediction_label": label,
        "threshold_used": ctx.threshold,
        "xgb_probability": p_xgb,
        "mri_message": mri_message,
        "top": ctx.explainer.top_features(prepared.iloc[[0]]),
        "shap_plot": shap_png,
        "shap_error": shap_error,
    }
    if p_mri is not None:
        response["mri_probability"] = p_mri
    return jsonify(response)


# ============================ 批量预测 ============================

@bp.route("/batch_predict_json", methods=["POST"])
def batch_predict_json():
    ctx = _ctx()
    if ctx.structured is None:
        return jsonify(error="XGBoost 模型未加载"), 503

    df_raw, error = _read_csv_upload()
    if error:
        return jsonify(error=error), 400

    missing = _missing_columns(df_raw, ctx.feature_order)
    if missing:
        return jsonify(error=f"CSV 缺少必需列：{missing}"), 400

    ids, id_label = _resolve_id_column(df_raw)
    prepared = ctx.structured.prepare(df_raw[ctx.feature_order].copy())
    probs = ctx.structured.predict_proba(prepared)
    preds = (probs >= ctx.threshold).astype(int)

    summary_png, shap_error = ctx.explainer.summary_plot(prepared)

    shap_csv_url = None
    shap_zip_url = None
    values = ctx.explainer.values(prepared)
    if values is not None:
        _write_shap_csv(values, ids, id_label, probs, preds, ctx.feature_order)
        shap_csv_url = "/download/shap_csv"
        if _wants_shap_images():
            _write_shap_images(values, ids, ctx.feature_order)
            shap_zip_url = "/download/shap_images_zip"

    rows = [
        {id_label: pid, "Prediction": int(p), "RiskProbability": float(r)}
        for pid, p, r in zip(ids, preds, probs)
    ]
    return jsonify({
        "id_field": id_label,
        "rows": rows,
        "threshold_used": ctx.threshold,
        "shap_summary": summary_png,
        "shap_error": shap_error,
        "shap_csv_url": shap_csv_url,
        "shap_images_zip_url": shap_zip_url,
    })


@bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    """前端 JSON 接口不可用时的降级出口，直接下载 Excel。"""
    ctx = _ctx()
    if ctx.structured is None:
        return "XGBoost 模型未加载", 503

    df_raw, error = _read_csv_upload()
    if error:
        return error, 400

    missing = _missing_columns(df_raw, ctx.feature_order)
    if missing:
        # 旧版本这里用 0 填补缺失列，会拿一堆零值算出看着很像样的概率，去掉了
        return f"CSV 缺少必需列：{missing}", 400

    ids, id_label = _resolve_id_column(df_raw)
    prepared = ctx.structured.prepare(df_raw[ctx.feature_order].copy())
    probs = ctx.structured.predict_proba(prepared)
    preds = (probs >= ctx.threshold).astype(int)

    buf = io.BytesIO()
    pd.DataFrame({
        id_label: ids,
        "RiskProbability": probs,
        "Prediction": preds,
    }).to_excel(buf, index=False)
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="batch_predictions.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _write_shap_csv(values, ids, id_label, probs, preds, feature_order) -> None:
    frame = pd.DataFrame(values, columns=[f"SHAP_{c}" for c in feature_order])
    frame.insert(0, id_label, ids)
    frame["RiskProbability"] = probs
    frame["Prediction"] = preds
    # utf-8-sig 让 Excel 直接打开不乱码
    frame.to_csv(BATCH_SHAP_CSV, index=False, encoding="utf-8-sig")


def _write_shap_images(values, ids, feature_order) -> None:
    """每行一张 Top-10 条形图，打包成 ZIP。

    先清空目录：旧实现是往里追加、再把整个目录打包，结果 ZIP 里混着上几次批量
    预测残留的图（仓库里那 2149 张就是这么攒出来的）。
    """
    if SHAP_IMAGES_DIR.exists():
        shutil.rmtree(SHAP_IMAGES_DIR)
    SHAP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(values):
        order = np.argsort(np.abs(row))[::-1][:10][::-1]
        plt.figure(figsize=(6, 3.6))
        plt.barh([feature_order[j] for j in order], np.abs(row[order]))
        plt.xlabel("abs(SHAP)")
        plt.title(f"Row {ids[i]} - Top10")
        plt.tight_layout()
        plt.savefig(SHAP_IMAGES_DIR / f"shap_row_{i + 1}_{ids[i]}.png", dpi=140, bbox_inches="tight")
        plt.close()

    with zipfile.ZipFile(SHAP_IMAGES_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for png in sorted(SHAP_IMAGES_DIR.glob("*.png")):
            zf.write(png, arcname=png.name)


# ============================ 下载 ============================

def _send_if_exists(path, download_name, as_attachment=True):
    if not path.exists():
        return f"文件尚未生成：{download_name}", 404
    return send_file(path, as_attachment=as_attachment, download_name=download_name)


@bp.route("/download/shap_csv")
def download_shap_csv():
    return _send_if_exists(BATCH_SHAP_CSV, "batch_shap_values.csv")


@bp.route("/download/shap_images_zip")
def download_shap_images_zip():
    return _send_if_exists(SHAP_IMAGES_ZIP, "shap_individual_images.zip")


@bp.route("/_debug_last/<kind>")
def debug_last(kind):
    """查看最近一次生成的 SHAP 图，排查用。"""
    path = DEBUG_SINGLE_PNG if kind == "single" else DEBUG_SUMMARY_PNG
    return _send_if_exists(path, path.name, as_attachment=False)


@bp.route("/download_template")
def download_template():
    """下载批量预测的 CSV 模板，列名和取值都是可直接使用的示例。"""
    sample = {
        "PatientID": "SAMPLE001", "Age": 70, "Gender": 0, "Ethnicity": 0,
        "EducationLevel": 2, "BMI": 25.5, "Smoking": 0, "AlcoholConsumption": 2,
        "PhysicalActivity": 5.0, "DietQuality": 7, "SleepQuality": 7.0,
        "SystolicBP": 120, "DiastolicBP": 80, "CholesterolTotal": 200,
        "CholesterolLDL": 100, "CholesterolHDL": 50, "CholesterolTriglycerides": 150,
        "MMSE": 28, "FunctionalAssessment": 8, "ADL": 9, "MemoryComplaints": 0,
        "BehavioralProblems": 0, "Confusion": 0, "Disorientation": 0,
        "PersonalityChanges": 0, "DifficultyCompletingTasks": 0, "Forgetfulness": 0,
    }
    frame = pd.DataFrame([sample])[["PatientID"] + _ctx().feature_order]

    # to_csv 不写文件时返回字符串，编码在这一步做；utf-8-sig 让 Excel 打开不乱码
    buf = io.BytesIO(frame.to_csv(index=False).encode("utf-8-sig"))
    return send_file(
        buf,
        as_attachment=True,
        download_name="alzheimers_prediction_template.csv",
        mimetype="text/csv",
    )
