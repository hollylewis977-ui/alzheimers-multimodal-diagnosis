# 原始 Notebook（转 .py）

来自 MRI-Alzheimers-CNN 项目的 Jupyter notebook，用 `jupytext` 之类的工具转成 `.py`
后原样保留，**仅供参考，不能直接在本地跑**：里面的数据路径指向 Kaggle 环境
（`/kaggle/input/...`），需要改成本地路径才能执行。

| 文件 | 内容 |
|---|---|
| `preprocessing.py` | 数据预处理 |
| `preprocessing_3d_to_2d.py` | 3D 体数据切成 2D 切片 |
| `resnet18_final.py` | ResNet-18 训练 |
| `vgg16_ml.py` | VGG16 训练 |
| `ml_final.py` | 汇总实验 |

可实际运行的训练脚本是上一级目录的 `train_xgb.py` 和 `train_mri_vgg.py`。
