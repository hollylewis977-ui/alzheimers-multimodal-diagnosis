# -*- coding: utf-8 -*-
import os, json, io, warnings, base64, zipfile
from flask import Flask, request, jsonify, render_template, send_file
import joblib, numpy as np, pandas as pd

# ==== 第二份代码中的导入（不可删） ====
from utils import ALL_FEATURES, NUMERIC_COLS, transform_features
from typing import Optional
from PIL import Image
from tensorflow.keras.applications import vgg16

IMG_SIZE = (224, 224)

def _img_bytes_to_batch(img_bytes: bytes) -> np.ndarray:
    """把二进制图片转成 (1,H,W,3) 并做 VGG16 预处理"""
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    im = im.resize(IMG_SIZE)
    arr = np.array(im).astype("float32")[None, ...]   # (1, H, W, 3)
    arr = vgg16.preprocess_input(arr)
    return arr

def predict_mri_from_filestorage(fs, model, ad_idx: int = 1) -> Optional[float]:
    """
    既支持 .zip（聚合多张）也支持 .jpg/.png（单张）。
    返回 AD 概率（0~1），无有效图片返回 None。
    """
    fname = (fs.filename or "").lower()
    blob = fs.read()
    fs.stream.seek(0)  # 复位文件指针

    if fname.endswith(".zip"):
        probs = []
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in zf.namelist():
                nl = name.lower()
                if nl.endswith((".jpg", ".jpeg", ".png")):
                    with zf.open(name) as f:
                        x = _img_bytes_to_batch(f.read())
                        p = float(model.predict(x, verbose=0)[0, ad_idx])
                        probs.append(p)
        return float(np.mean(probs)) if probs else None

    elif fname.endswith((".jpg", ".jpeg", ".png")):
        x = _img_bytes_to_batch(blob)
        p = float(model.predict(x, verbose=0)[0, ad_idx])
        return p

    else:
        return None

# ==== Matplotlib headless & cache（来自第一份） ====
MPL_CACHE = os.path.join(os.path.dirname(__file__), "artifacts", "mpl-cache")
os.makedirs(MPL_CACHE, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ==== SHAP 可视化偏好（不影响模型与融合）====
SHAP_STYLE = "force"  # 选 "force" 可回到你原来的长条推拉图；"waterfall" 为当前新版
USE_RAW_FEATURES_FOR_LABELS = True  # 左侧标注使用原始特征值（例如 MMSE=20.0）


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

# ==== MRI models (2D via Keras; 3D via PyTorch optional) ====
MRI_VGG_PATH = os.path.join(ART_DIR, "best_model.pth")
mri_model_vgg = None
try:
    from tensorflow.keras.models import load_model as _keras_load
    if os.path.exists(MRI_VGG_PATH):
        mri_model_vgg = _keras_load(MRI_VGG_PATH)
        print("[MRI] 2D model loaded:", MRI_VGG_PATH)
    else:
        print("[MRI] 2D model not found (skip):", MRI_VGG_PATH)
except Exception as e:
    print("[MRI] 2D model init failed:", repr(e))
    mri_model_vgg = None

MRI_RESNET_PATH = os.path.join(ART_DIR, "mri_resnet_model.pth")
mri_model_resnet = None
try:
    import torch, torch.nn as nn, torch.nn.functional as F, nibabel as nib
    class BasicBlock(nn.Module):
        def __init__(self, in_size, out_size, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv3d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm3d(out_size)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(out_size, out_size, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm3d(out_size)
            self.downsample = downsample
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
            out = self.conv2(out); out = self.bn2(out)
            out += identity; out = self.relu(out)
            return out
    class ResNet3D(nn.Module):
        def __init__(self, block, layers, num_classes=3):
            super().__init__()
            self.in_size = 64
            self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm3d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.dropout = nn.Dropout(p=0.3)
            self.fc = nn.Linear(512, num_classes)
        def _make_layer(self, block, out_size, blocks, stride=1):
            downsample = None
            if stride != 1 or self.in_size != out_size:
                downsample = nn.Sequential(nn.Conv3d(self.in_size, out_size, kernel_size=1, stride=stride),
                                           nn.BatchNorm3d(out_size))
            layers = [block(self.in_size, out_size, stride, downsample)]
            self.in_size = out_size
            for _ in range(1, blocks):
                layers.append(block(self.in_size, out_size))
            return nn.Sequential(*layers)
        def forward(self, x):
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
            x = self.avgpool(x)
            import torch
            x = torch.flatten(x, 1)
            x = self.dropout(x); x = self.fc(x)
            return x
    def ResNet18_3D(num_classes=3): return ResNet3D(BasicBlock, [2,2,2,2], num_classes)
    if os.path.exists(MRI_RESNET_PATH):
        mri_model_resnet = ResNet18_3D(num_classes=3)
        mri_model_resnet.load_state_dict(torch.load(MRI_RESNET_PATH, map_location='cpu'))
        mri_model_resnet.eval()
        print("[MRI] 3D model loaded:", MRI_RESNET_PATH)
except Exception as e:
    mri_model_resnet = None
    print("[MRI] 3D model init failed or deps missing:", repr(e))

# <<< MODIFIED START: 在App启动时加载阈值相关配置 >>>
BEST_THRESHOLD = 0.5  # 如果加载失败，则使用默认阈值
mri_class_names = ["AD", "CN"] # 默认类别名称
ad_idx = 1 # 默认AD索引为1

try:
    # 加载类别名称
    CLASS_JSON_PATH = os.path.join(ART_DIR, "class_names.json")
    if os.path.exists(CLASS_JSON_PATH):
        with open(CLASS_JSON_PATH, 'r', encoding='utf-8') as f:
            mri_class_names = json.load(f)

    # 加载计算出的最佳阈值
    THRESHOLD_PATH = os.path.join(ART_DIR, "threshold.json")
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, 'r') as f:
            config = json.load(f)
        BEST_THRESHOLD = config['best_threshold']

    # 动态找到 AD 类的索引
    for i, name in enumerate(mri_class_names):
        n_lower = name.lower()
        if "ad" in n_lower or "demented" in n_lower:
            ad_idx = i
            break
            
    print("="*50)
    print(f"[SUCCESS] 阈值相关配置已加载。")
    print(f"          类别: {mri_class_names}, AD索引: {ad_idx}")
    print(f"          最佳决策阈值: {BEST_THRESHOLD:.4f}")
    print("="*50)

except Exception as e:
    print("="*50)
    print(f"[WARN] 加载阈值或类别配置文件失败: {e}")
    print(f"       将使用默认阈值 0.5 和默认类别进行预测。")
    print("="*50)
# <<< MODIFIED END >>>


def _predict_mri_from_zip(file_storage):
    if file_storage is None or not file_storage.filename:
        return None, "No MRI provided"
    try:
        with zipfile.ZipFile(file_storage, 'r') as zf:
            names = [n for n in zf.namelist() if not n.endswith('/')]
            if not names: return None, "Empty MRI ZIP"
            has_nifti = any(n.lower().endswith(('.nii','.nii.gz')) for n in names)
            probs = []
            if has_nifti and mri_model_resnet is not None:
                import numpy as np, torch, torch.nn.functional as F, nibabel as nib
                for n in names:
                    if not n.lower().endswith(('.nii','.nii.gz')): continue
                    data = zf.read(n)
                    tmp = os.path.join(ART_DIR, f"tmp_{os.path.basename(n)}")
                    with open(tmp, 'wb') as f: f.write(data)
                    img = nib.load(tmp)
                    vol = img.get_fdata().astype(np.float32)
                    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
                    ten = torch.tensor(vol).unsqueeze(0).unsqueeze(0)
                    ten = F.interpolate(ten, size=(64,64,64), mode='trilinear', align_corners=False)
                    with torch.no_grad():
                        logits = mri_model_resnet(ten)
                        p = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    if p.shape[0] == 2: p_ad = float(p[1])
                    elif p.shape[0] == 3: p_ad = float(p[0])
                    elif p.shape[0] == 4: p_ad = float(1.0 - p[2])
                    else: p_ad = float(np.max(p))
                    probs.append(p_ad)
            else:
                if mri_model_vgg is None:
                    return None, "2D MRI model not loaded"
                from PIL import Image
                import numpy as np
                for n in names:
                    if not n.lower().endswith(('.jpg','.jpeg','.png')): continue
                    data = zf.read(n)
                    try:
                        img = Image.open(io.BytesIO(data)).convert('RGB').resize((224,224))
                    except Exception:
                        continue
                    arr = np.array(img, dtype=np.float32)/255.0
                    pred = mri_model_vgg.predict(np.expand_dims(arr, axis=0), verbose=0)
                    if pred.shape[1] == 1: p_ad = float(pred[0,0])
                    elif pred.shape[1] == 2: p_ad = float(pred[0,1])
                    elif pred.shape[1] == 3: p_ad = float(pred[0,0])
                    elif pred.shape[1] == 4: p_ad = float(1.0 - pred[0,2])
                    else: p_ad = float(np.max(pred[0]))
                    probs.append(p_ad)
            if probs:
                return float(sum(probs)/len(probs)), None
            return None, "No valid MRI images found"
    except zipfile.BadZipFile:
        return None, "Invalid ZIP"
    except Exception as e:
        return None, str(e)

def _fuse_probs(p_mri, p_xgb):
    if p_mri is None and p_xgb is None: return None
    if p_mri is None: return p_xgb
    if p_xgb is None: return p_mri

    # 1. 分别计算两个模型的置信度
    conf_mri = abs(p_mri - 0.5)
    conf_xgb = abs(p_xgb - 0.5)
    
    # 2. 对置信度进行归一化，得到权重
    # 添加一个小的平滑项epsilon防止除以零
    epsilon = 1e-6 
    total_conf = conf_mri + conf_xgb + epsilon
    
    w_mri = conf_mri / total_conf
    w_xgb = conf_xgb / total_conf
    
    # 3. 加权融合
    return w_mri * p_mri + w_xgb * p_xgb



MODEL_PATH = os.path.join(ART_DIR, 'alzheimers_xgb_model.pkl')
SCALER_PATH = os.path.join(ART_DIR, 'scaler.pkl')
FEATURE_ORDER_PATH = os.path.join(ART_DIR, 'feature_order.json')
BG_PATH = os.path.join(ART_DIR, "shap_bg.npy")

# app = Flask(__name__, template_folder="templates") # app 已在上面定义

# ==== 加载模型/Scaler/特征顺序（保持第二份行为） ====
# model 和 scaler 等已在上面加载

# ==== 第一份里的 SHAP 背景（可选） ====
bg = None
if os.path.exists(BG_PATH):
    try:
        bg = np.load(BG_PATH)
        print(f"[SHAP] loaded background: {bg.shape}")
    except Exception as e:
        print("[SHAP] background load failed:", repr(e))

# ==== 枚举&范围（保留第二份） ====web下拉菜单
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
    """
    Normalize SHAP outputs to (n_samples, n_features) float array.
    """
    import numpy as np
    if isinstance(values, list):
        arr = np.stack([np.asarray(v, dtype=float) for v in values], axis=1)
    else:
        arr = np.asarray(values)

    if arr.ndim == 3:
        if arr.shape[1] > 1:
            try:
                arr = arr[:, 1, :]
            except Exception:
                arr = np.nanmean(arr, axis=1)
        else:
            arr = arr[:, 0, :]
    elif arr.ndim == 1:
        arr = arr[np.newaxis, :]

    return np.asarray(arr, dtype=float)


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
        if hasattr(model, "steps"):
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
        return booster.predict(dmat, pred_contribs=True)
    except Exception as e:
        print("[xgb contribs] failed:", repr(e))
        return None

def _single_plot(df_t_row):
    """返回单样本 SHAP 图（base64）。"""
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
            try:
                plt.figure(figsize=(6.4, 3.8))
                shap.force_plot(0.0, v2[0], X.iloc[0, :], matplotlib=True, show=False)
                return _fig_b64(os.path.join(ART_DIR, "debug_single_shap.png")), None
            except Exception:
                idx = np.argsort(np.abs(v2[0]))[::-1][:10]
                plt.figure(figsize=(6.4, 3.8))
                plt.barh([feature_order[i] for i in idx[::-1]], np.abs(v2[0][idx])[::-1])
                plt.xlabel("abs(SHAP)")
                return _fig_b64(os.path.join(ART_DIR, "debug_single_bar.png")), None
    except Exception as e:
        print("[single shap] failed:", repr(e))
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
    """返回全体样本 SHAP summary 图（base64）。"""
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

# ========= 路由 =========

@app.route('/')
def index():
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
        return jsonify({'error': 'Model not trained yet. Please run train.py first.'}), 400

    df_t = transform_features(df, scaler)
    if not isinstance(df_t, pd.DataFrame):
        df_t = pd.DataFrame(df_t, columns=feature_order)

    prob_xgb = float(model.predict_proba(df_t)[0, 1])

    p_mri = None
    mri_msg = None
    file_mri = request.files.get('mri_zip')

    if file_mri and file_mri.filename:
        fname = file_mri.filename.lower()
        try:
            if fname.endswith('.zip'):
                p_mri, mri_msg = _predict_mri_from_zip(file_mri)
            elif fname.endswith(('.jpg', '.jpeg', '.png')):
                if mri_model_vgg is None:
                    mri_msg = "2D MRI model not loaded"
                else:
                    p_mri = predict_mri_from_filestorage(file_mri, mri_model_vgg, ad_idx=ad_idx)
                    if p_mri is None:
                        mri_msg = "No valid MRI image"
            else:
                mri_msg = "Unsupported file type (please upload .zip/.jpg/.png)"
        except Exception as e:
            p_mri, mri_msg = None, str(e)

    p_final = _fuse_probs(p_mri, prob_xgb) or prob_xgb
    
    # <<< MODIFIED START: 使用最佳阈值进行最终判断 >>>
    if p_final >= BEST_THRESHOLD:
        prediction_label = mri_class_names[ad_idx]
        pred = 1 # 假设 AD 对应 1
    else:
        cn_idx = 1 - ad_idx
        prediction_label = mri_class_names[cn_idx]
        pred = 0 # 假设 CN 对应 0
    # <<< MODIFIED END >>>

    shap_b64, shap_err = _single_plot(df_t.iloc[[0]])
    top = []
    try:
        exp = _get_explainer()
        if exp is not None:
            X = _ensure_df(df_t.iloc[[0]])
            try:
                expl = exp(X)
                values = getattr(expl, "values", expl)
            except Exception:
                values = exp.shap_values(X)
            v2 = _values_to_2d(values)
            v = v2[0]
            idx = np.argsort(np.abs(v))[::-1][:10]
            top = [{'feature': feature_order[i], 'value': float(v[i])} for i in idx]
        elif hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            idx = np.argsort(fi)[::-1][:10]
            top = [{'feature': feature_order[i], 'value': float(fi[i])} for i in idx]
    except Exception:
        pass

    return jsonify({
        'probability': p_final,
        'prediction': pred,
        'prediction_label': prediction_label,
        'threshold_used': BEST_THRESHOLD,
        'xgb_probability': prob_xgb,
        **({'mri_probability': p_mri} if p_mri is not None else {}),
        'mri_message': mri_msg,
        'top': top,
        'shap_plot': shap_b64, 'shap_error': shap_err,
        'shapImg': shap_b64, 'shap_image': shap_b64, 'shap': shap_b64, 'shap_png': shap_b64
    })



@app.route('/batch_predict_json', methods=['POST'])
def batch_predict_json():
    file = request.files.get('file')
    if not file:
        return jsonify(error='no file'), 400
    df_raw = pd.read_csv(file)

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

    missing = [c for c in feature_order if c not in df_raw.columns]
    if missing:
        return jsonify(error=f'missing columns: {missing}'), 400

    df_features = df_raw[feature_order].copy()
    if scaler is None or model is None:
        return jsonify(error='Model not loaded'), 500
    df_t = transform_features(df_features, scaler)
    if not isinstance(df_t, pd.DataFrame):
        df_t = pd.DataFrame(df_t, columns=feature_order)

    probs = model.predict_proba(df_t)[:, 1]
    
    # <<< MODIFIED START: 在批量预测中也使用最佳阈值 >>>
    preds = (probs >= BEST_THRESHOLD).astype(int)
    # <<< MODIFIED END >>>

    shap_sum_b64, shap_err = _summary_plot(df_t)

    shap_values = None
    try:
        exp = _get_explainer()
        if exp is not None:
            expl = exp(_ensure_df(df_t))
            shap_values = getattr(expl, "values", expl)
        else:
            contribs = _xgb_contribs(_ensure_df(df_t))
            if contribs is not None:
                shap_values = contribs[:, :-1]
    except Exception as e:
        print("[batch shap values] failed:", repr(e))
        shap_values = None

    shap_csv_url = None
    shap_zip_url = None
    if shap_values is not None:
        v2 = _values_to_2d(shap_values)
        shap_cols = [f"SHAP_{c}" for c in feature_order]
        shap_df = pd.DataFrame(v2, columns=shap_cols)
        shap_df.insert(0, id_label, ids)
        shap_df["RiskProbability"] = probs
        shap_df["Prediction"] = preds
        shap_csv_path = os.path.join(ART_DIR, "batch_shap_values.csv")
        shap_df.to_csv(shap_csv_path, index=False, encoding="utf-8-sig")
        shap_csv_url = "/download/shap_csv"

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
        zip_path = os.path.join(ART_DIR, "shap_individual_images.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fn in os.listdir(img_dir):
                if fn.lower().endswith(".png"):
                    zf.write(os.path.join(img_dir, fn), arcname=fn)
        shap_zip_url = "/download/shap_images_zip"

    out = [
        {id_label: pid, 'Prediction': int(p), 'RiskProbability': float(r)}
        for pid, p, r in zip(ids, preds, probs)
    ]
    payload = {
        'id_field': id_label,
        'rows': out,
        'shap_summary': shap_sum_b64, 'shap_error': shap_err,
        'shapSummary': shap_sum_b64, 'shap_summary_png': shap_sum_b64,
        'shap_csv_url': shap_csv_url,
        'shap_images_zip_url': shap_zip_url,
    }
    return jsonify(payload)


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
    
    # <<< MODIFIED START: 在批量Excel预测中也使用最佳阈值 >>>
    preds = (probs >= BEST_THRESHOLD).astype(int)
    # <<< MODIFIED END >>>

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

@app.route("/_debug_last/<kind>")
def _debug_last(kind):
    path = os.path.join(ART_DIR,
        "debug_single_shap.png" if kind=="single" else "debug_summary_shap.png")
    if not os.path.exists(path):
        return "no debug image", 404
    return send_file(path, as_attachment=False, download_name=os.path.basename(path))

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


@app.route('/download_template')
def download_template():
    template_data = {
        'PatientID': ['SAMPLE001'],
        'Age': [70],
        'Gender': [0],
        'Ethnicity': [0],
        'EducationLevel': [2],
        'BMI': [25.5],
        'Smoking': [0],
        'AlcoholConsumption': [2],
        'PhysicalActivity': [5.0],
        'DietQuality': [7],
        'SleepQuality': [7.0],
        'SystolicBP': [120],
        'DiastolicBP': [80],
        'CholesterolTotal': [200],
        'CholesterolLDL': [100],
        'CholesterolHDL': [50],
        'CholesterolTriglycerides': [150],
        'MMSE': [28],
        'FunctionalAssessment': [8],
        'ADL': [9],
        'MemoryComplaints': [0],
        'BehavioralProblems': [0],
        'Confusion': [0],
        'Disorientation': [0],
        'PersonalityChanges': [0],
        'DifficultyCompletingTasks': [0],
        'Forgetfulness': [0]
    }

    template_df = pd.DataFrame(template_data)
    template_df = template_df[['PatientID'] + feature_order]

    buf = io.BytesIO()
    template_df.to_csv(buf, index=False, encoding='utf-8-sig')
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name='alzheimers_prediction_template.csv',
        mimetype='text/csv'
    )
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)