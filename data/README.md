# 数据集

这个目录不入版本库（MRI 切片约 113MB）。

```
data/
├── structured/alzheimers_disease_patient_data.csv   患者临床数据
├── mri/Alzheimer_s Dataset/                         AD / CN 二分类，线上模型用的就是这份
│   ├── train/{AD,CN}/{axial,coronal,sagittal}/
│   └── test/{AD,CN}/{axial,coronal,sagittal}/
└── mri_kaggle_4class/                               四分类原始数据，历史实验用
    └── {train,test}/{MildDemented,ModerateDemented,NonDemented,VeryMildDemented}/
```

两份都来自 Kaggle 公开数据集。`mri/` 是在四分类基础上按 AD / CN 重新组织并按解剖
切面分目录的版本。

训练脚本默认读 `data/mri/Alzheimer_s Dataset`，可用 `AD_MRI_DATA_DIR` 覆盖。
