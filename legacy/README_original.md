
# 阿尔茨海默症 多模态融合诊断（MRI + XGBoost）

> **重要**：不修改你原有的训练/模型代码。原项目完整保留在 `originals/` 下；本集成仅**新增**后端 `app.py` 与前端 `templates/index.html`，并在后端实现**门控加权（MRI 优先）**融合。

## 目录结构
```
AD_Integrated_App/
├─ app.py                      # 集成后的 Flask 后端（新增加）
├─ templates/
│  └─ index.html               # 集成后的前端页面（新增加）
├─ artifacts/                  # 统一放置已训练模型（可选）
│  ├─ mri_vgg_model.h5         # 2D MRI Keras 模型（训练后放入）
│  └─ mri_resnet_model.pth     # 3D MRI PyTorch 模型（可选）
├─ data/
│  ├─ mri/Alzheimer_s Dataset/ # 你的 2D MRI 数据集（已放好）
│  └─ structured/alzheimers_disease_patient_data.csv  # 结构化数据（已放好）
├─ originals/
│  ├─ ad_risk_xgb_app_pro2/    # 你原有的 XGB 工程（**原样保留**）
│  └─ MRI-Alzheimers-CNN-main/ # 你原有的 MRI 工程（**原样保留**）
└─ train_scripts/
   └─ mri_notebooks_converted/ # 你的 MRI .ipynb 已原样转成 .py（仅便于本地直接运行）
```

> 注：同时也将数据复制到了 `originals` 工程内常见位置（例如 `originals/ad_risk_xgb_app_pro2/.../alzheimers_disease_patient_data.csv`、`originals/MRI-Alzheimers-CNN-main/Alzheimer_s Dataset/` ），这样**无需改你的 Notebook/脚本路径**也能直接训练。

## 环境依赖
- Python 3.8+
- Flask, pandas, numpy, joblib, scikit-learn, xgboost
- Keras/TensorFlow（用于 2D MRI 推理）
- （可选）PyTorch、nibabel（用于 3D MRI 推理；前端默认不暴露3D上传）

建议：`pip install -r originals/ad_risk_xgb_app_pro2/ad_risk_xgb_app_pro2/ad_risk_xgb_app_pro/requirements.txt`，并额外安装：
```
pip install flask tensorflow pillow nibabel torch torchvision xlsxwriter
```

## 训练（**不修改你的训练逻辑**）

### 1) XGBoost 风险模型
进入你的原工程目录：
```
cd originals/ad_risk_xgb_app_pro2/ad_risk_xgb_app_pro2/ad_risk_xgb_app_pro
```
确认此处已存在 `alzheimers_disease_patient_data.csv`（本包已放好）。然后直接使用你原有的 `train.py`（保持不变）训练：
```
python train.py
```
训练完成后会生成/覆盖 `artifacts/alzheimers_xgb_model.pkl`（你的工程已自带一个可用模型）。

> 若你的 `utils.py` 需要 `scaler.pkl`，本集成在运行时会自动尝试根据训练 CSV 拟合并生成（不会修改你的代码）。

### 2) MRI 影像模型（2D）
你的 MRI 训练 Notebook 已**原样转换为 .py**，位于：
```
train_scripts/mri_notebooks_converted/...
```
同样，**不修改任何逻辑**。为避免手动改路径，数据已复制到：
```
originals/MRI-Alzheimers-CNN-main/Alzheimer_s Dataset/
```
请在该 MRI 工程中，按你 Notebook 原先的方式运行/训练（或直接在对应 `.py` 中运行同样的单元）。训练得到的 2D Keras 模型请保存为：
```
artifacts/mri_vgg_model.h5
```
> 如果你的 Notebook 保存到其它路径，请在训练后手动将模型文件复制到 `artifacts/mri_vgg_model.h5`。

（可选）如你有 3D 体数据训练，可在 MRI 工程中训练并把权重放到：`artifacts/mri_resnet_model.pth`。前端当前不提供 3D 上传入口，但后端已兼容。

## 启动应用
在项目根目录：
```
python app.py
```
浏览器打开 `http://127.0.0.1:5000/`。

- **单人预测**：填写结构化数据，可选上传该患者的 MRI ZIP（2D切片）。
- **批量预测**：上传 CSV（需包含 `PatientID` 列以便与 MRI ZIP 匹配），可选上传 MRI ZIP（按ID命名文件夹或前缀）。

## 融合策略（MRI 优先，门控加权）
设 MRI 概率为 `p_mri`，XGB 概率为 `p_xgb`：
```
w = 2 * |p_mri - 0.5|   （裁剪到 0~1）
p_final = w * p_mri + (1 - w) * p_xgb
```
当 MRI 置信度高（`p_mri` 远离 0.5）时，更依赖 MRI；当 MRI 接近 0.5 时，更倚重 XGB。阈值 0.5 输出二分类。

## 不修改原则
- `originals/` 下的所有代码（你的 XGB 与 MRI 工程）**不做任何改动**；
- 集成功能仅新增在根目录的 `app.py` 与 `templates/index.html`；
- 训练与推理流程尊重你已有逻辑，必要时通过**复制数据到原工程内**的方式来“对齐路径”，避免改代码。

## 备注
- 首次运行如果找不到 `scaler.pkl`，后端会自动用训练 CSV 拟合一个并保存在 `originals/.../artifacts/scaler.pkl`（仅用于推理管线对齐）。
- 如尚未训练 2D MRI 模型，单人/批量推理页面仍可仅用 XGB 输出（页面会提示 MRI 未加载）。
- 若需在前端开启 3D 上传，后续仅需在 `index.html` 增加一个文件输入框并上传 `.nii/.nii.gz` 到 ZIP 中即可（后端已支持）。
