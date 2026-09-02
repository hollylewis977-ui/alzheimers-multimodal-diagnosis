# -*- coding: utf-8 -*-
"""XGBoost 结构化风险模型训练（SHAP 可选，不装也不报错）。

    python scripts/train_xgb.py

产出分别落在各自的目录：
    artifacts/models/alzheimers_xgb_model.pkl   模型权重
    artifacts/config/scaler.pkl                 标准化器（推理必须，与模型配套）
    artifacts/config/feature_order.json         特征顺序（推理必须）
    artifacts/reports/xgb/                      ROC、混淆矩阵、阈值、SHAP 概览
    artifacts/outputs/                          验证集/测试集的预测概率，便于复盘
"""
import os, json, argparse, sys, joblib
from pathlib import Path

# 让脚本能 import 到仓库根目录下的 adapp 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, f1_score, ConfusionMatrixDisplay,
    precision_score, recall_score, precision_recall_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# —— 让 SHAP 成为可选，不装也不报错 ——
try:
    import shap
except Exception:
    shap = None

from adapp import config
from adapp.features import TARGET_COL, ALL_FEATURES, NUMERIC_COLS, fit_scaler, transform_features

DEFAULT_DATA_CSV = config.DATA_DIR / "structured" / "alzheimers_disease_patient_data.csv"
XGB_REPORTS_DIR = config.REPORTS_DIR / "xgb"


def choose_thresholds(y_true, prob, target_recall=0.90, recall_policy="conservative"):
    # F1 最大
    p, r, thr_pr = precision_recall_curve(y_true, prob)
    f1 = 2 * p * r / (p + r + 1e-12)
    thr_f1 = float(thr_pr[np.nanargmax(f1[:-1])])

    # Youden 最大
    fpr, tpr, thr_roc = roc_curve(y_true, prob)
    thr_youden = float(thr_roc[np.argmax(tpr - fpr)])

    # Recall 目标（在真实概率上扫描）
    uniq = np.unique(prob); uniq.sort()
    thr_r90 = float(uniq[-1])
    if recall_policy == "conservative":          # 最高阈值且 Recall≥target（少误报）
        it = uniq[::-1]                           # 高→低
    else:                                         # 激进：尽快达标（多抓正例）
        it = uniq                                  # 低→高
    for t in it:
        if recall_score(y_true, (prob >= t).astype(int)) >= target_recall:
            thr_r90 = float(t); break

    return {"f1": thr_f1, "youden": thr_youden, "recall90": thr_r90}


def eval_at_threshold(y, prob, thr):
    # ✅ 加入 zero_division=0，避免边界情况下的 UndefinedMetricWarning
    pred = (prob >= thr).astype(int)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "f1":        float(f1_score(y, pred, zero_division=0))
    }


def main(args):
    # 1) 读取数据
    df = pd.read_csv(args.data_csv)
    drop_cols = [c for c in ['PatientID', 'DoctorInCharge'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    assert TARGET_COL in df.columns, f'Missing target column: {TARGET_COL}'

    feat_cols = [c for c in df.columns if c != TARGET_COL]
    X, y = df[feat_cols], df[TARGET_COL].astype(int)

    # 2) 划分数据集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    # 3) 标准化器（只对数值列拟合）
    for d in (config.MODELS_DIR, config.CONFIG_DIR, config.OUTPUTS_DIR, XGB_REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    scaler_path = config.SCALER_PATH
    scaler = fit_scaler(pd.concat([X_train, X_valid], axis=0), scaler_path)

    Xt_tr = transform_features(X_train, scaler)
    Xt_va = transform_features(X_valid, scaler)
    Xt_te = transform_features(X_test,  scaler)

    # 4) 采样平衡
    over = RandomOverSampler(random_state=42)
    under = RandomUnderSampler(random_state=42)
    X_over, y_over = over.fit_resample(Xt_tr, y_train)
    X_bal, y_bal = under.fit_resample(X_over, y_over)

    # 5) 训练 XGBoost（早停）
    clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.05,
        max_depth=5,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        n_estimators=6000,
        tree_method='hist',
        n_jobs=-1,
        # xgboost 2.0 起 fit() 不再接受 early_stopping_rounds，必须写在构造函数里
        early_stopping_rounds=100,
    )
    clf.fit(X_bal, y_bal, eval_set=[(Xt_va, y_valid)], verbose=100)

    best_iter = getattr(clf, "best_iteration", None)
    PRED_KW = {}
    if best_iter is not None:
        PRED_KW["iteration_range"] = (0, best_iter + 1)
    else:
        best_ntree = getattr(clf, "best_ntree_limit", None)
        if best_ntree is not None:
            PRED_KW["ntree_limit"] = best_ntree


    # 6) 评估
    def eva(name, X_, y_):
        p = clf.predict_proba(X_, **PRED_KW)[:, 1]
        auc = roc_auc_score(y_, p)
        print(f"{name} AUC: {auc:.4f}")
        return p, auc

    _, _   = eva("Train(balanced subset)", X_bal, y_bal)
    pv, _  = eva("Valid", Xt_va, y_valid)
    pt, at = eva("Test",  Xt_te, y_test)
       
    
    # 选择阈值（基于验证集）
    thr_dict = choose_thresholds(y_valid, pv)
    # 你可以改成 "youden" 或 "recall90"
    selected_key = "recall90"
    best_thr = float(thr_dict[selected_key])

    print(f"[Thresholds] {thr_dict}  | selected={selected_key}:{best_thr:.4f}")
    print("[Valid@thr]", eval_at_threshold(y_valid, pv, best_thr))
    print("[Test @thr]",  eval_at_threshold(y_test,  pt, best_thr))

    # ✅ 保存阈值与 AUC（确保在 main 内，缩进一致）
    thr_path = XGB_REPORTS_DIR / "threshold.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump({
            "selected": selected_key,
            "values": thr_dict,
            "valid_auc": float(roc_auc_score(y_valid, pv)),
            "test_auc": float(at)
        }, f, indent=2, ensure_ascii=False)
    print("Threshold saved to", thr_path)

    # ✅ （可选）把验证/测试集概率也保存，方便以后复盘（仍在 main 内）
    pd.DataFrame({"y": y_valid, "prob": pv}).to_csv(config.OUTPUTS_DIR / "valid_preds.csv", index=False)
    pd.DataFrame({"y": y_test,  "prob": pt}).to_csv(config.OUTPUTS_DIR / "test_preds.csv", index=False)

    # ✅ 7) 持久化（仍在 main 内）
    model_path = config.XGB_MODEL_PATH
    joblib.dump(clf, model_path)
    print("Model saved to", model_path)
    print("Scaler saved to", scaler_path)

    with open(config.FEATURE_ORDER_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(Xt_tr.columns), f, ensure_ascii=False, indent=2)

    # 8) ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, pt)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC={at:.3f})')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on Test Set')
    plt.legend(loc='lower right')
    plt.savefig(XGB_REPORTS_DIR / 'roc_test.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 9) 混淆矩阵 + 报告 —— ✅ 使用选出来的 best_thr，而不是 0.5
    y_pred = (pt >= best_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix @thr={best_thr:.6f}:\n', cm)
    print('Classification Report @thr:\n', classification_report(y_test, y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No AD", "AD"])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix on Test Set (thr={best_thr:.3f})')
    plt.savefig(XGB_REPORTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 10) （可选）SHAP 总结图：只有在 shap 可用时才生成
    try:
        if shap is not None:
            explainer = shap.TreeExplainer(clf)
            n_bg = min(200, len(Xt_va))
            rng = np.random.RandomState(42)
            idx = rng.choice(len(Xt_va), size=n_bg, replace=False)
            background = Xt_va.iloc[idx]
            shap_vals = explainer(background)
            shap.summary_plot(shap_vals.values, background, show=False)
            plt.savefig(XGB_REPORTS_DIR / 'shap_summary.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
        else:
            print("SHAP not available: skipped shap_summary.png")
    except Exception as e:
        print("SHAP summary failed:", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练 XGBoost 结构化风险模型')
    parser.add_argument('--data_csv', type=str, default=str(DEFAULT_DATA_CSV),
                        help='训练用的患者数据 CSV')
    args = parser.parse_args()
    main(args)
