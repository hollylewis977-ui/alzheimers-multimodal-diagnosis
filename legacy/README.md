# 存档

重构前的原始代码，**不参与运行**，保留下来是为了可追溯和方便对照。

| 路径 | 说明 |
|---|---|
| `app_monolithic_reference.py` | 重构前实际在用的单文件版本（原 `app_resnet2d_only.py`）。`docs/architecture.md` 里的行号引用对应的就是这个文件。 |
| `app_keras_vgg16.py` | 更早的 Keras/VGG16 版本（原 `app.py`）。**已失效**：它用 `keras.load_model()` 去读 `best_model.pth`，而那是 PyTorch 权重，必然加载失败。 |
| `xgb_app/` | 原始 XGBoost 工程 `ad_risk_xgb_app_pro_new`（原本嵌套了三层同名目录）。其中的 `train.py` 已整理为 `scripts/train_xgb.py`。 |
| `mri_cnn/` | 原始 MRI 工程 `MRI-Alzheimers-CNN-main`，含 notebook 与文档。 |
| `README_original.md` | 重构前的 README，其中描述的目录结构与融合公式均已过时。 |

重复的 `alzheimers_disease_patient_data.csv` 原本在三处各存了一份（内容完全相同），
现在只保留 `data/structured/` 下的那份。
