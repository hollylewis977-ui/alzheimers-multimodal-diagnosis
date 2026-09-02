# -*- coding: utf-8 -*-
import os, json, io, warnings, base64, zipfile
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, abort
import joblib, numpy as np, pandas as pd

# ==== 第二份代码中的导入（不可删） ====
from utils import ALL_FEATURES, NUMERIC_COLS, transform_features

# ==== Matplotlib headless & cache（来自第一份） ====
MPL_CACHE = os.path.join(os.path.dirname(__file__), "artifacts", "mpl-cache")
os.makedirs(MPL_CACHE, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ==== 可选依赖（来自第一份） ====
try:
    import shap
    print("[SHAP] version:", getattr(shap, "__version__", "unknown"))
except Exception as _e:
    shap = None
    print("[SHAP] import failed:", _e)

try:
    import xgboost as xgb
except Exception as _e:
    xgb = None
    print("[XGBoost] import failed:", _e)

# ==== 基础路径 ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, 'artifacts')
os.makedirs(ART_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ART_DIR, 'alzheimers_xgb_model.pkl')
SCALER_PATH = os.path.join(ART_DIR, 'scaler.pkl')
FEATURE_ORDER_PATH = os.path.join(ART_DIR, 'feature_order.json')
BG_PATH = os.path.join(ART_DIR, "shap_bg.npy")

app = Flask(__name__, template_folder="templates")

# ==== 加载模型/Scaler/特征顺序（保持第二份行为） ====
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
if os.path.exists(FEATURE_ORDER_PATH):
    with open(FEATURE_ORDER_PATH,'r',encoding='utf-8') as f:
        feature_order = json.load(f)
else:
    feature_order = ALL_FEATURES
    
THR_PATH = os.path.join(ART_DIR, "threshold.json")
THRESHOLD = float(os.environ.get("MODEL_THRESHOLD", "0.5"))  # 允许环境变量覆盖

try:
    with open(THR_PATH, "r", encoding="utf-8") as f:
        _thrdata = json.load(f)
        sel = _thrdata.get("selected", "f1")
        THRESHOLD = float(_thrdata.get("values", {}).get(sel, THRESHOLD))
except Exception as e:
    print("No threshold.json or parse failed -> use default:", THRESHOLD, e)


# ==== 第一份里的 SHAP 背景（可选） ====
bg = None
if os.path.exists(BG_PATH):
    try:
        bg = np.load(BG_PATH)
        print(f"[SHAP] loaded background: {bg.shape}")
    except Exception as e:
        print("[SHAP] background load failed:", repr(e))

# ==== 枚举&范围（保留第二份） ====
ENUMS = {
    'Gender': [{'label':'Male','value':0},{'label':'Female','value':1}],
    'Ethnicity': [
        {'label':'White','value':0},
        {'label':'African American','value':1},
        {'label':'Asian','value':2},
        {'label':'Others','value':3},
    ],
    'EducationLevel': [
        {'label':'No education','value':0},
        {'label':'Primary school','value':1},
        {'label':'Secondary school','value':2},
        {'label':'Higher education','value':3},
    ],
    'Smoking': [{'label':'Non-smoking','value':0},{'label':'Smoking','value':1}],
    'MemoryComplaints': [{'label':'No','value':0},{'label':'Yes','value':1}],
    'BehavioralProblems':[{'label':'No','value':0},{'label':'Yes','value':1}],
    'Confusion':[{'label':'No','value':0},{'label':'Yes','value':1}],
    'Disorientation':[{'label':'No','value':0},{'label':'Yes','value':1}],
    'PersonalityChanges':[{'label':'No','value':0},{'label':'Yes','value':1}],
    'DifficultyCompletingTasks':[{'label':'No','value':0},{'label':'Yes','value':1}],
    'Forgetfulness':[{'label':'No','value':0},{'label':'Yes','value':1}],
}

BOUNDS = {
    'Age': (0, 120), 'BMI': (10, 60),
    'AlcoholConsumption': (0, 20), 'PhysicalActivity': (0, 10), 'DietQuality': (0, 10),
    'SleepQuality': (4, 10),
    'SystolicBP': (70, 250), 'DiastolicBP': (40, 150),
    'CholesterolTotal': (100, 400), 'CholesterolLDL': (0, 300), 'CholesterolHDL': (10, 120),
    'CholesterolTriglycerides': (30, 800),
    'MMSE': (0, 30), 'FunctionalAssessment': (0, 10), 'ADL': (0, 10),
}

BINARY_COLS = [
    'MemoryComplaints','BehavioralProblems','Confusion','Disorientation',
    'PersonalityChanges','DifficultyCompletingTasks','Forgetfulness'
]

BMI_OPTIONS = [i for i in range(15, 41)]  # 保留

# ========= 辅助函数（来自第一份，做了小封装） =========
def _clamp(name, v):
    lo, hi = BOUNDS.get(name, (None, None))
    if lo is not None: v = max(v, lo)
    if hi is not None: v = min(v, hi)
    return v

def _ensure_df(X):
    if isinstance(X, pd.DataFrame):
        return X[feature_order]
    X = np.asarray(X)
    return pd.DataFrame(X, columns=feature_order)

def _values_to_2d(values):
    arr = np.asarray(values)
    if arr.ndim == 3:
        # (n_samples, n_classes, n_features) 或 (n_classes, n_samples, n_features)
        if arr.shape[1] in (2, 3):
            arr = arr[:, 1 if arr.shape[1] > 1 else 0, :]
        else:
            arr = arr[1 if arr.shape[0] > 1 else 0, :, :]
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr

def _fig_b64(save_path=None):
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    if save_path:
        try:
            with open(save_path, "wb") as f:
                f.write(buf.getvalue())
        except Exception as e:
            print("[save debug image failed]", repr(e))
    buf.seek(0)
    out = base64.b64encode(buf.getvalue()).decode("ascii")
    plt.close()
    return out

_SHAP_EXPLAINER = None
def _get_explainer():
    """优先 TreeExplainer，失败则回退通用 Explainer。"""
    global _SHAP_EXPLAINER
    if shap is None or model is None:
        return None
    if _SHAP_EXPLAINER is not None:
        return _SHAP_EXPLAINER
    est = model
    try:
        from sklearn.pipeline import Pipeline
        if hasattr(model, "steps"):  # 兼容 pipeline
            est = model.steps[-1][1]
    except Exception:
        pass
    try:
        _SHAP_EXPLAINER = shap.TreeExplainer(est, data=bg) if bg is not None else shap.TreeExplainer(est)
        print("[SHAP] TreeExplainer initialized")
    except Exception as e:
        print("[SHAP] TreeExplainer failed:", repr(e))
        try:
            _SHAP_EXPLAINER = shap.Explainer(est, bg) if bg is not None else shap.Explainer(est)
            print("[SHAP] Generic Explainer initialized")
        except Exception as e2:
            print("[SHAP] Generic Explainer failed:", repr(e2))
            _SHAP_EXPLAINER = None
    return _SHAP_EXPLAINER

def _xgb_contribs(df_t):
    """当 shap 不可用时，回退到 xgboost 的 pred_contribs。"""
    if xgb is None or model is None: 
        return None
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else None
        if booster is None:
            return None
        dmat = xgb.DMatrix(df_t.values, feature_names=feature_order, missing=np.nan)
        return booster.predict(dmat, pred_contribs=True)  # (n_samples, F+1)
    except Exception as e:
        print("[xgb contribs] failed:", repr(e))
        return None

def _single_plot(df_t_row):
    """返回单样本 SHAP 图（base64）。失败则回退到 contribs 的条形图。"""
    # try shap
    try:
        exp = _get_explainer()
        if exp is not None:
            X = _ensure_df(df_t_row)
            try:
                expl = exp(X)
                values = getattr(expl, "values", expl)
            except Exception:
                values = exp.shap_values(X)
            v2 = _values_to_2d(values)
            # force plot 优先
            try:
                plt.figure(figsize=(6.4, 3.8))
                shap.force_plot(0.0, v2[0], X.iloc[0, :], matplotlib=True, show=False)
                return _fig_b64(os.path.join(ART_DIR, "debug_single_shap.png")), None
            except Exception:
                # 次选：Top10 abs(SHAP) 柱图
                idx = np.argsort(np.abs(v2[0]))[::-1][:10]
                plt.figure(figsize=(6.4, 3.8))
                plt.barh([feature_order[i] for i in idx[::-1]], np.abs(v2[0][idx])[::-1])
                plt.xlabel("abs(SHAP)")
                return _fig_b64(os.path.join(ART_DIR, "debug_single_bar.png")), None
    except Exception as e:
        print("[single shap] failed:", repr(e))

    # fallback xgb contribs
    try:
        contribs = _xgb_contribs(_ensure_df(df_t_row))
        if contribs is not None:
            vals = contribs[0, :-1]
            idx = np.argsort(np.abs(vals))[::-1][:10]
            plt.figure(figsize=(6.4, 3.8))
            plt.barh([feature_order[i] for i in idx[::-1]], np.abs(vals[idx])[::-1])
            plt.xlabel("abs(SHAP) (xgboost contribs)")
            return _fig_b64(os.path.join(ART_DIR, "debug_single_contribs.png")), None
    except Exception as e2:
        print("[single contribs] failed:", repr(e2))
    return None, "shap_and_xgb_contribs_failed"

def _summary_plot(df_t):
    """返回全体样本 SHAP summary 图（base64），失败则回退 contribs 的 mean|SHAP|。"""
    try:
        exp = _get_explainer()
        if exp is not None:
            X = _ensure_df(df_t)
            try:
                expl = exp(X)
                values = getattr(expl, "values", expl)
            except Exception:
                values = exp.shap_values(X)
            v2 = _values_to_2d(values)
            plt.figure(figsize=(7.2, 4.2))
            shap.summary_plot(v2, X, show=False, max_display=20)
            return _fig_b64(os.path.join(ART_DIR, "debug_summary_shap.png")), None
    except Exception as e:
        print("[summary shap] failed:", repr(e))

    try:
        contribs = _xgb_contribs(_ensure_df(df_t))
        if contribs is not None:
            vals = np.mean(np.abs(contribs[:, :-1]), axis=0)
            idx = np.argsort(vals)[::-1][:20]
            plt.figure(figsize=(7.2, 4.2))
            plt.barh([feature_order[i] for i in idx[::-1]], vals[idx][::-1])
            plt.xlabel("mean |SHAP| (xgboost contribs)")
            return _fig_b64(os.path.join(ART_DIR, "debug_summary_contribs.png")), None
    except Exception as e2:
        print("[summary contribs] failed:", repr(e2))
    return None, "shap_and_xgb_contribs_failed"

# ========= 路由（尽量原样保留第二份；在原路由中补充返回字段） =========

@app.route('/')
def index():
    # 与第二份一致：构造前端元信息
    meta = {}
    for f in feature_order:
        if f in NUMERIC_COLS:
            lo, hi = BOUNDS.get(f, (None, None))
            meta[f] = {'kind': 'number', 'min': lo, 'max': hi, 'step': 0.01}
        elif f in BINARY_COLS:
            meta[f] = {'kind': 'binary'}
        else:
            meta[f] = {'kind': 'int', 'min': 0, 'max': 5, 'step': 1}
    return render_template('index.html', features=feature_order, meta=meta)

@app.route('/predict', methods=['POST'])
def predict():
    # —— 原有逻辑：读取请求、边界钳制、预测、top 特征（尽量不改） ——
    data = request.get_json(silent=True) or request.form.to_dict()
    row = {}
    for f in feature_order:
        try:
            v = float(data.get(f, 0))
        except:
            v = 0.0
        row[f] = _clamp(f, v)
    df = pd.DataFrame([row], columns=feature_order)
    if scaler is None or model is None:
        return jsonify({'error':'Model not trained yet. Please run train.py first.'}), 400
    df_t = transform_features(df, scaler)
    if not isinstance(df_t, pd.DataFrame):
        df_t = pd.DataFrame(df_t, columns=feature_order)

    prob = float(model.predict_proba(df_t)[:,1][0])
    pred = int(prob >= THRESHOLD)

    # —— 新增：单样本 SHAP 图（base64），字段名与第一份保持一致（前端已兼容） ——
    shap_b64, shap_err = _single_plot(df_t.iloc[[0]])

    # —— 原有 Top10 贡献表（保留） ——
    top = []
    try:
        if shap is not None:
            explainer = _get_explainer()
            vals = getattr(explainer(df_t), "values", explainer.shap_values(df_t))
            v2 = _values_to_2d(vals)[0]
            idx = np.argsort(np.abs(v2))[::-1][:10]
            top = [{'feature': feature_order[i], 'value': float(v2[i])} for i in idx]
        else:
            fi = model.feature_importances_
            idx = np.argsort(fi)[::-1][:10]
            top = [{'feature': feature_order[i], 'value': float(fi[i])} for i in idx]
    except Exception:
        try:
            fi = model.feature_importances_
            idx = np.argsort(fi)[::-1][:10]
            top = [{'feature': feature_order[i], 'value': float(fi[i])} for i in idx]
        except Exception:
            top = []

    # —— 合并返回（向前兼容所有字段名） ——
    return jsonify({
        'probability': prob, 'prediction': pred, 'top': top,
        'shap_plot': shap_b64, 'shap_error': shap_err,
        'shapImg': shap_b64, 'shap_image': shap_b64, 'shap': shap_b64, 'shap_png': shap_b64,
        'threshold': THRESHOLD        # ← 就把这一行放在这里
    })

@app.route('/batch_predict_json', methods=['POST'])
def batch_predict_json():
    file = request.files.get('file')
    if not file:
        return jsonify(error='no file'), 400
    df_raw = pd.read_csv(file)

    # 选择ID列（原逻辑保留）
    id_col = None
    for cand in ['PatientID', 'patient_id', 'ID', 'Id', 'id']:
        if cand in df_raw.columns:
            id_col = cand
            break
    if id_col is not None:
        ids = df_raw[id_col].astype(str).tolist()
        id_label = id_col
    else:
        ids = (pd.Series(range(1, len(df_raw) + 1))).astype(str).tolist()
        id_label = 'Row'

    # 强校验（原逻辑保留）
    missing = [c for c in feature_order if c not in df_raw.columns]
    if missing:
        return jsonify(error=f'missing columns: {missing}'), 400

    # 预测
    df_features = df_raw[feature_order].copy()
    if scaler is None or model is None:
        return jsonify(error='Model not loaded'), 500
    df_t = transform_features(df_features, scaler)
    if not isinstance(df_t, pd.DataFrame):
        df_t = pd.DataFrame(df_t, columns=feature_order)

    probs = model.predict_proba(df_t)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # === 新增：批量 SHAP summary 图（base64） ===
    shap_sum_b64, shap_err = _summary_plot(df_t)

    # === 新增：为每个患者保存 SHAP 宽表 CSV（每个特征一列） ===
    shap_values = None
    try:
        exp = _get_explainer()
        if exp is not None:
            expl = exp(_ensure_df(df_t))
            shap_values = getattr(expl, "values", expl)
        else:
            contribs = _xgb_contribs(_ensure_df(df_t))
            if contribs is not None:
                shap_values = contribs[:, :-1]  # 去掉 bias
    except Exception as e:
        print("[batch shap values] failed:", repr(e))
        shap_values = None

    shap_csv_url = None
    shap_zip_url = None
    if shap_values is not None:
        v2 = _values_to_2d(shap_values)
        # 宽表：ID + 每个特征的 SHAP + 概率 + 预测
        shap_cols = [f"SHAP_{c}" for c in feature_order]
        shap_df = pd.DataFrame(v2, columns=shap_cols)
        shap_df.insert(0, id_label, ids)
        shap_df["RiskProbability"] = probs
        shap_df["Prediction"] = preds
        shap_csv_path = os.path.join(ART_DIR, "batch_shap_values.csv")
        shap_df.to_csv(shap_csv_path, index=False, encoding="utf-8-sig")
        shap_csv_url = "/download/shap_csv"

        # 为每位患者生成 Top10 条形图并打包
        img_dir = os.path.join(ART_DIR, "shap_individual")
        os.makedirs(img_dir, exist_ok=True)
        for idx_row in range(v2.shape[0]):
            vals = v2[idx_row]
            order = np.argsort(np.abs(vals))[::-1][:10]
            plt.figure(figsize=(6, 3.6))
            plt.barh([feature_order[i] for i in order[::-1]], np.abs(vals[order])[::-1])
            plt.xlabel("abs(SHAP)")
            plt.title(f"Row {ids[idx_row]} - Top10")
            img_path = os.path.join(img_dir, f"shap_row_{idx_row+1}_{ids[idx_row]}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=140, bbox_inches="tight")
            plt.close()
        # 打包 zip
        zip_path = os.path.join(ART_DIR, "shap_individual_images.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fn in os.listdir(img_dir):
                if fn.lower().endswith(".png"):
                    zf.write(os.path.join(img_dir, fn), arcname=fn)
        shap_zip_url = "/download/shap_images_zip"

    # —— 原返回结构（保留），并追加 shap_summary & 下载链接 ——
    out = []
    for pid, p, rprob in zip(ids, preds, probs):
        row_obj = {id_label: str(pid), 'Prediction': int(p), 'RiskProbability': float(rprob)}
        row_obj['PatientID'] = str(pid)  # 兼容固定取名
        out.append(row_obj)
        
    payload = {
        'id_field': id_label,
        'rows': out,
        'shap_summary': shap_sum_b64, 'shap_error': shap_err,
        'shapSummary': shap_sum_b64, 'shap_summary_png': shap_sum_b64,
        'shap_csv_url': shap_csv_url,
        'shap_images_zip_url': shap_zip_url,
    }
    return jsonify(payload)

# —— 原 Excel 回退端点（保留原样） ——
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400
    df_raw = pd.read_csv(file)

    id_col = None
    for cand in ['PatientID', 'patient_id', 'ID', 'Id', 'id']:
        if cand in df_raw.columns:
            id_col = cand
            break
    id_series = df_raw[id_col].astype(str) if id_col is not None else pd.Series(range(1, len(df_raw)+1), name='Row')

    for c in feature_order:
        if c not in df_raw.columns:
            df_raw[c] = 0
    df_features = df_raw[feature_order]
    df_t = transform_features(df_features, scaler)
    if not isinstance(df_t, pd.DataFrame):
        df_t = pd.DataFrame(df_t, columns=feature_order)

    probs = model.predict_proba(df_t)[:, 1]
    preds = (probs >= 0.5).astype(int)

    out = pd.DataFrame({
        (id_col or 'Row'): id_series.values,
        'RiskProbability': probs,
        'Prediction': preds
    })
    buf = io.BytesIO()
    out.to_excel(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='batch_predictions.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# ==== 新增：调试图下载（来自第一份） ====
@app.route("/_debug_last/<kind>")
def _debug_last(kind):
    path = os.path.join(ART_DIR,
        "debug_single_shap.png" if kind=="single" else "debug_summary_shap.png")
    if not os.path.exists(path):
        return "no debug image", 404
    return send_file(path, as_attachment=False, download_name=os.path.basename(path))

# ==== 新增：批量 SHAP CSV / ZIP 下载 ====
@app.route("/download/shap_csv")
def download_shap_csv():
    path = os.path.join(ART_DIR, "batch_shap_values.csv")
    if not os.path.exists(path):
        return "no shap csv", 404
    return send_file(path, as_attachment=True, download_name="batch_shap_values.csv")

@app.route("/download/shap_images_zip")
def download_shap_images_zip():
    path = os.path.join(ART_DIR, "shap_individual_images.zip")
    if not os.path.exists(path):
        return "no shap images zip", 404
    return send_file(path, as_attachment=True, download_name="shap_individual_images.zip")

if not os.path.exists(os.path.join(ART_DIR, "sample.csv")):
    print("[WARN] sample.csv not found in ART_DIR:", ART_DIR)


@app.route("/download/<path:filename>")
def download(filename):
    ALLOWED = {"sample.csv"}
    if filename not in ALLOWED:
        abort(404)
    fp = os.path.join(ART_DIR, filename)
    print("[DOWNLOAD] ART_DIR =", ART_DIR, "| file =", fp, "| exists =", os.path.exists(fp))
    return send_from_directory(ART_DIR, filename, as_attachment=True, mimetype="text/csv")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
