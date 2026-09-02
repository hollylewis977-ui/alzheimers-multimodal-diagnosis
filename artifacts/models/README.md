# 模型权重

这个目录不入版本库（`best_model.pth` 282MB、四个 `.h5` 共 450MB，都超过
GitHub 单文件 100MB 的上限）。需要自己放置或重新训练。

| 文件 | 用途 | 来源 |
|---|---|---|
| `best_model.pth` | **线上 MRI 模型**，PyTorch ResNet50 | 2025-10-30 训练 |
| `alzheimers_xgb_model.pkl` | **线上结构化模型**，XGBoost | `python scripts/train_xgb.py` |
| `mri_vgg_model.h5` | VGG16 对比实验 | `python scripts/train_mri_vgg.py` |
| `mri_vgg_model_finetuned.h5` | VGG16 微调后 | 同上 |
| `mri_resnet_model.h5` | Keras ResNet 早期实验 | 已弃用 |
| `mri_resnet_model_finetuned.h5` | 同上 | 已弃用 |

只有前两个是应用运行需要的。
