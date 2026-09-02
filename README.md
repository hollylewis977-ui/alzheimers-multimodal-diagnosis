# 阿尔茨海默病风险预测（多模态融合）

一个 Flask Web 应用，把两个独立训练的模型融合起来做阿尔茨海默病（AD）风险评估：

| 分支 | 模型 | 输入 |
|---|---|---|
| 结构化 | XGBoost | 26 项临床指标（年龄、MMSE、ADL、血脂、行为症状……） |
| 影像 | ResNet50 (PyTorch) | 2D MRI 切片，单张或 ZIP 打包 |

两路概率按各自置信度加权融合，再用 SHAP 解释结构化分支的每一项贡献。
支持单人预测和 CSV 批量预测。

> ⚠️ 研究与教学用途。模型未经临床验证，不可用于实际诊断。

---

## 快速开始

```bash
pip install -r requirements.txt
python run.py
```

打开 http://127.0.0.1:5000/

启动时会打印真正生效的配置，先确认这几行再用：

```
  结构化分支 (XGBoost) : 已就绪
  影像分支 (ResNet50)  : 已就绪
  类别                 : ['AD', 'CN']（AD 下标 0）
  决策阈值             : 0.5000
  阈值来源             : 默认值（未提供 threshold.json）
```

### 需要自己准备的文件

模型权重和数据集都没有入库（体积超过 GitHub 限制），需要放到对应位置：

```
artifacts/models/best_model.pth              ResNet50 权重（282MB）
artifacts/models/alzheimers_xgb_model.pkl    XGBoost 模型（622KB）
data/structured/alzheimers_disease_patient_data.csv
data/mri/Alzheimer_s Dataset/{train,test}/{AD,CN}/
```

`artifacts/config/` 下的 scaler、特征顺序、类别名随仓库提供，不用另外准备。

缺哪个分支就少哪个分支的能力，应用不会启动失败：
没有 `best_model.pth` 就退化成纯结构化预测，没有 XGBoost 模型则预测接口返回 503。

两个模型都可以自己重新训练，见下面的「训练」。

---

## 目录结构

```
.
├── run.py                      启动入口
├── adapp/                      应用本体
│   ├── config.py               路径、阈值、表单枚举与取值范围
│   ├── features.py             特征定义与标准化（训练/推理共用）
│   ├── models.py               XGBoost + scaler + 特征顺序
│   ├── mri.py                  ResNet50 影像推理
│   ├── fusion.py               置信度加权融合
│   ├── explain.py              SHAP 解释与绘图
│   ├── routes.py               HTTP 路由
│   └── templates/index.html    前端页面
├── scripts/                    训练脚本
│   ├── train_xgb.py            XGBoost
│   ├── train_mri_vgg.py        VGG16（历史对比实验）
│   └── notebooks/              原始 Kaggle notebook 转成的 .py，仅供参考
├── artifacts/
│   ├── config/                 推理必需的小配置（入库）
│   ├── models/                 模型权重（不入库）
│   ├── outputs/                运行时产物（不入库）
│   └── reports/                训练评估结果（入库）
│       ├── xgb/
│       ├── mri_resnet50/
│       └── mri_vgg16/
├── data/                       数据集（不入库）
├── docs/
│   ├── architecture.md         详细架构说明
│   └── xgb-training.md         XGBoost 训练流程详解
└── legacy/                     重构前的原始代码，仅作存档
```

---

## 工作原理

### 融合策略

以「离 0.5 有多远」衡量每个模型的置信度，越笃定权重越高：

```
conf_i  = |p_i - 0.5|
w_i     = conf_i / (conf_mri + conf_xgb + ε)
p_final = w_mri · p_mri + w_xgb · p_xgb
```

只上传了临床数据、没传 MRI 时，`p_final` 就等于 XGBoost 的输出。

### 决策阈值

`p_final >= 阈值` 判为 AD。阈值按以下优先级解析，**启动日志会打印实际生效的值和来源**：

1. 环境变量 `AD_DECISION_THRESHOLD`
2. `artifacts/config/threshold.json` 里的 `best_threshold`
3. 默认 `0.5`

默认停在 0.5 是刻意的，原因见下面的「已知问题」。

调整方式：

```bash
AD_DECISION_THRESHOLD=0.55 python run.py    # 临时
```

---

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 表单页面 |
| POST | `/predict` | 单人预测。表单字段 + 可选 `mri_zip`（.zip/.jpg/.png） |
| POST | `/batch_predict_json` | 批量预测，返回 JSON。字段 `file`（CSV） |
| POST | `/batch_predict` | 批量预测，直接下载 Excel |
| GET | `/download_template` | 下载批量预测的 CSV 模板 |
| GET | `/download/shap_csv` | 上一次批量预测的逐行 SHAP 值 |
| GET | `/download/shap_images_zip` | 上一次批量预测的逐行 SHAP 图（需先开启，见下） |

`/predict` 返回：

```json
{
  "probability": 0.72,          // 融合后的最终概率
  "prediction": 1,
  "prediction_label": "AD",
  "threshold_used": 0.5,
  "xgb_probability": 0.68,
  "mri_probability": 0.81,      // 未上传 MRI 时不返回这个字段
  "top": [{"feature": "MMSE", "value": -0.42}],
  "shap_plot": "<base64 PNG>"
}
```

**逐行 SHAP 图默认关闭。** 1000 行 CSV 会生成 1000 个 PNG 文件，需要时在批量表单里带上
`shap_images=1` 才生成。逐行 SHAP 数值的 CSV 是照常输出的，开销很小。

---

## 训练

```bash
pip install -r requirements.txt -r requirements-train.txt

# XGBoost：模型 -> artifacts/models/，scaler 和特征顺序 -> artifacts/config/
python scripts/train_xgb.py

# VGG16（历史对比实验，非线上模型）
python scripts/train_mri_vgg.py
```

MRI 数据目录可用环境变量覆盖：`AD_MRI_DATA_DIR=/path/to/dataset`

### 已有的评估结果

| 模型 | 指标 |
|---|---|
| XGBoost | 验证集 AUC 0.955，测试集 AUC 0.941 |
| VGG16 | 测试集 AUC 0.745 |
| ResNet50（线上） | 见 `artifacts/reports/mri_resnet50/` 的曲线图，未留下数值报告 |

完整报告在 `artifacts/reports/` 各子目录下。

---

## 已知问题与限制

这些是重构时发现但没有一并改掉的，改动会影响模型行为，留给使用者判断：

1. **线上 ResNet50 没有配套的阈值。** 仓库里那份 `best_threshold = 0.6049` 是
   2025-10-17 为 **VGG16** 算的（Youden's J，见 `artifacts/reports/mri_vgg16/`），
   而线上跑的是 10-30 训练的 ResNet50，阈值又作用在融合之后的概率上。
   拿旧模型的阈值套新模型没有依据，所以默认值停在 0.5。
   想用调优阈值，需要在融合输出上重新标定。

2. **批量预测不接受 MRI。** 前端有 `mri_zip_batch` 上传框，但后端的批量接口只跑
   XGBoost，不读这个字段。批量结果里的概率是纯结构化模型的输出。

3. **ResNet50 权重用 `strict=False` 加载。** 结构对不上时不会报错，只会打印
   缺失/多余的参数数量。换权重后请留意启动日志里的这行提示。

4. **SHAP 只解释结构化分支。** 影像分支没有可解释性输出，融合后的概率也没有归因。

---

## 文档

- [`docs/architecture.md`](docs/architecture.md) — 各模块、数据流、持久化文件的详细说明
- [`docs/xgb-training.md`](docs/xgb-training.md) — XGBoost 训练流程逐步拆解

## 数据来源

- 结构化数据：Alzheimer's Disease Patient Data（Kaggle 公开数据集）
- MRI 影像：Alzheimer's Dataset（Kaggle 公开数据集），已按 AD / CN 二分类重新组织
