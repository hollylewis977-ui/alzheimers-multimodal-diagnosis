# 运行时产物

批量预测每跑一次就重新生成，不入版本库。

- `batch_shap_values.csv` — 逐行 SHAP 值
- `shap_individual/`、`shap_individual_images.zip` — 逐行 SHAP 图（默认不生成，
  需要在批量表单里带 `shap_images=1`）
- `debug_single_shap.png`、`debug_summary_shap.png` — 最近一次的解释图，
  可通过 `/_debug_last/single` 和 `/_debug_last/summary` 查看
- `valid_preds.csv`、`test_preds.csv` — 训练脚本留下的预测概率，便于复盘
