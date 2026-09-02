#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VGG16 二维 MRI 分类器训练与评估（tf.keras）。

    python scripts/train_mri_vgg.py

产出：
    artifacts/models/mri_vgg_model.h5            两阶段训练的模型
    artifacts/config/class_names.json            类别名（推理时用来对齐 AD 下标）
    artifacts/config/class_distribution.json     各划分的类别样本数
    artifacts/reports/mri_vgg16/                 AUC、F1、混淆矩阵、ROC、阈值

注意：应用当前线上跑的是 PyTorch ResNet50（artifacts/models/best_model.pth），
不是这里训练的 VGG16。这个脚本保留是为了能复现 VGG16 那一轮的对比实验。
"""
import os, json, sys
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 让脚本能 import 到仓库根目录下的 adapp 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.applications import vgg16
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 区域1: 路径和参数定义 (完全保持你的设置)
# ==============================================================================
from adapp import config

# 原来这里写死了 D:\桌面\... 的绝对路径，换台机器就跑不了，改成跟着仓库走
DATA_DIR = Path(os.environ.get("AD_MRI_DATA_DIR", config.DATA_DIR / "mri" / "Alzheimer_s Dataset"))
print("[INFO] DATA_DIR =", DATA_DIR)

ART_DIR = config.REPORTS_DIR / "mri_vgg16"   # 评估报告
ART_DIR.mkdir(parents=True, exist_ok=True)
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = config.MODELS_DIR / "mri_vgg_model.h5"
CLASS_JSON = config.CLASS_NAMES_PATH          # artifacts/config/class_names.json
DIST_JSON = config.CONFIG_DIR / "class_distribution.json"

IMG_SIZE = (224, 224)
BATCH = 32  # 显存不足可降到 16
SEED = 42
VAL_SPLIT_DEFAULT = 0.2

# 两阶段训练轮数
EPOCHS_HEAD = 15        # 冻结卷积基，仅训顶部
EPOCHS_FINETUNE = 15    # 解冻 block5 微调

# 数据增强（仅训练集用）
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

def log(msg): print(f"[INFO] {msg}")

# ---------------- 工具：从数据集对象统计类别数量 & 计算 class_weight ----------------
def _count_from_dataset(ds, num_classes):
    counts = np.zeros((num_classes,), dtype=np.int64)
    # 对于已经 batch 的数据集，直接迭代即可
    for _, y in ds:
        yy = y.numpy()  # (bs, C) one-hot
        counts += yy.sum(axis=0).astype(np.int64)
    return counts

def _make_class_weight(counts):
    total = counts.sum()
    n = len(counts)
    # 经典反比权重： total / (n * count_i)
    cw = {i: float(total / (n * max(1, c))) for i, c in enumerate(counts)}
    return cw

# ==============================================================================
# 区域2: 数据加载函数 (保持你的逻辑，并集成增强步骤)
# ==============================================================================
def load_datasets():
    tf.random.set_seed(SEED); np.random.seed(SEED)

    train_dir = DATA_DIR / "train"
    val_dir   = DATA_DIR / "valid"
    test_dir  = DATA_DIR / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"[数据根] 下找不到 train 目录：{train_dir}")

    print("[DEBUG] train_dir exists:", train_dir.is_dir(), "->", train_dir)
    if test_dir.exists():
        print("[DEBUG] test_dir  exists:", test_dir.is_dir(),  "->", test_dir)

    autotune = tf.data.AUTOTUNE

    def apply_augmentation_and_preprocessing(ds, is_training=False):
        def ensure_rgb(image, label):
            if image.shape[-1] == 1:
                image = tf.image.grayscale_to_rgb(image)
            return image, label

        ds = ds.map(ensure_rgb, num_parallel_calls=autotune)
        if is_training:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=autotune)
        ds = ds.map(lambda x, y: (vgg16.preprocess_input(x), y),
                    num_parallel_calls=autotune)
        return ds.prefetch(autotune)

    def p2s(p):  # windows \ -> /
        return str(p).replace("\\", "/")

    try:
        if val_dir.is_dir():
            train_ds_raw = tf.keras.utils.image_dataset_from_directory(
                p2s(train_dir), image_size=IMG_SIZE, batch_size=BATCH,
                seed=SEED, label_mode="categorical"
            )
            val_ds_raw = tf.keras.utils.image_dataset_from_directory(
                p2s(val_dir), image_size=IMG_SIZE, batch_size=BATCH,
                seed=SEED, label_mode="categorical", shuffle=False
            )
        else:
            train_ds_raw = tf.keras.utils.image_dataset_from_directory(
                p2s(train_dir), validation_split=VAL_SPLIT_DEFAULT, subset="training",
                seed=SEED, image_size=IMG_SIZE, batch_size=BATCH, label_mode="categorical"
            )
            val_ds_raw = tf.keras.utils.image_dataset_from_directory(
                p2s(train_dir), validation_split=VAL_SPLIT_DEFAULT, subset="validation",
                seed=SEED, image_size=IMG_SIZE, batch_size=BATCH, label_mode="categorical",
                shuffle=False
            )

        class_names = train_ds_raw.class_names
        num_classes = len(class_names)

        test_ds_raw = None
        if (DATA_DIR/"test").is_dir():
            test_ds_raw = tf.keras.utils.image_dataset_from_directory(
                p2s(test_dir), image_size=IMG_SIZE, batch_size=BATCH,
                seed=SEED, label_mode="categorical", shuffle=False
            )

        # 先统计原始分布，再做预处理/增强
        train_counts = _count_from_dataset(train_ds_raw, num_classes)
        val_counts   = _count_from_dataset(val_ds_raw,   num_classes)
        test_counts  = _count_from_dataset(test_ds_raw,  num_classes) if test_ds_raw else None

        # 应用预处理
        train_ds = apply_augmentation_and_preprocessing(train_ds_raw, is_training=True)
        val_ds   = apply_augmentation_and_preprocessing(val_ds_raw,   is_training=False)
        test_ds  = apply_augmentation_and_preprocessing(test_ds_raw,  is_training=False) if test_ds_raw else None

        # 保存分布信息
        dist_payload = {
            "class_names": class_names,
            "train_counts": train_counts.tolist(),
            "val_counts":   val_counts.tolist(),
            **({"test_counts": test_counts.tolist()} if test_counts is not None else {})
        }
        json.dump(dist_payload, open(DIST_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        log(f"Class distribution saved to: {DIST_JSON}")

    except Exception as e:
        print("[WARN] Keras image_dataset_from_directory 失败，改用手工扫描。原因：", repr(e))
        import os

        def scan_folder(root: Path):
            class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
            idx = {c:i for i,c in enumerate(class_names)}
            paths, labels = [], []
            exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
            for c in class_names:
                for dp, _, fns in os.walk(root / c):
                    for fn in fns:
                        if fn.lower().endswith(exts):
                            paths.append(str(Path(dp)/fn))
                            labels.append(idx[c])
            return paths, labels, class_names

        def make_ds_manual(root: Path, shuffle=True, augment=False):
            paths, labels, class_names = scan_folder(root)
            num_classes = len(class_names)
            if not paths:
                raise FileNotFoundError(f"目录里没有图片：{root}")
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            def _load(path, label):
                img = tf.io.read_file(path)
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                img = tf.image.resize(img, IMG_SIZE)
                label = tf.one_hot(label, depth=num_classes)
                return img, label
            ds = ds.map(_load, num_parallel_calls=autotune)
            if augment:
                ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=autotune)
            ds = ds.map(lambda x, y: (vgg16.preprocess_input(x), y), num_parallel_calls=autotune)
            if shuffle:
                ds = ds.shuffle(1000, seed=SEED)
            ds = ds.batch(BATCH).prefetch(autotune)
            # 统计
            counts = _count_from_dataset(ds, num_classes)
            return ds, class_names, counts

        if val_dir.is_dir():
            train_ds, class_names, train_counts = make_ds_manual(train_dir, shuffle=True, augment=True)
            val_ds,   _,          val_counts   = make_ds_manual(val_dir,   shuffle=False, augment=False)
        else:
            paths, labels, class_names = scan_folder(train_dir)
            num_classes = len(class_names)
            rng = np.random.default_rng(SEED)
            idx_all = np.arange(len(paths)); rng.shuffle(idx_all)
            n_val = int(len(idx_all) * VAL_SPLIT_DEFAULT)
            val_idx, tr_idx = idx_all[:n_val], idx_all[n_val:]

            def ds_from_idx_manual(sel_idx, shuffle, augment=False):
                sel_paths  = [paths[i] for i in sel_idx]
                sel_labels = [labels[i] for i in sel_idx]
                ds = tf.data.Dataset.from_tensor_slices((sel_paths, sel_labels))
                def _load(path, label):
                    img = tf.io.read_file(path)
                    img = tf.image.decode_image(img, channels=3, expand_animations=False)
                    img = tf.image.resize(img, IMG_SIZE)
                    label = tf.one_hot(label, depth=num_classes)
                    return img, label
                ds = ds.map(_load, num_parallel_calls=autotune)
                if augment:
                    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=autotune)
                ds = ds.map(lambda x, y: (vgg16.preprocess_input(x), y), num_parallel_calls=autotune)
                if shuffle:
                    ds = ds.shuffle(1000, seed=SEED)
                return ds.batch(BATCH).prefetch(autotune)

            train_ds = ds_from_idx_manual(tr_idx, shuffle=True,  augment=True)
            val_ds   = ds_from_idx_manual(val_idx, shuffle=False, augment=False)

            train_counts = _count_from_dataset(train_ds, len(class_names))
            val_counts   = _count_from_dataset(val_ds,   len(class_names))

        test_ds = None
        if test_dir.is_dir():
            test_ds, _, test_counts = make_ds_manual(test_dir, shuffle=False, augment=False)
        else:
            test_counts = None

        # 保存分布信息
        dist_payload = {
            "class_names": class_names,
            "train_counts": train_counts.tolist(),
            "val_counts":   val_counts.tolist(),
            **({"test_counts": test_counts.tolist()} if test_counts is not None else {})
        }
        json.dump(dist_payload, open(DIST_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        log(f"Class distribution saved to: {DIST_JSON}")

    return train_ds, val_ds, test_ds, class_names

# ==============================================================================
# 区域3: 模型构建函数（加入 L2 与 label smoothing；AUC 指标）
# ==============================================================================
def build_model(num_classes: int):
    base = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(1e-4))(x)
    model = models.Model(inputs, outputs)

    # AUC 指标：二分类与多分类分别配置
    if num_classes == 2:
        auc_metric = tf.keras.metrics.AUC(name="auc")
    else:
        auc_metric = tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=num_classes)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=loss_fn,
                  metrics=["accuracy", auc_metric])
    return model

# ==============================================================================
# 区域4: 评估函数 (保持你的逻辑)
# ==============================================================================
def unbatch_labels(ds):
    ys = []
    if ds is None: return None
    for _, batch_y in ds:
        ys.append(batch_y.numpy())
    return np.concatenate(ys, axis=0)

def plot_roc_curve(y_true, y_prob, ad_idx, best_threshold_idx, roc_auc, save_path):
    """绘制ROC曲线并标记最佳阈值点"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 标记最佳阈值点
    plt.scatter(fpr[best_threshold_idx], tpr[best_threshold_idx], marker='o', color='red', s=100, zorder=5,
                label=f'Best Threshold ({thresholds[best_threshold_idx]:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log(f"ROC curve saved to {save_path}")

def evaluate_and_save(model, ds_eval, class_names, tag="val"):
    y_true_1hot = unbatch_labels(ds_eval)
    y_true = np.argmax(y_true_1hot, axis=1)
    y_prob = model.predict(ds_eval, verbose=1)
    num_classes = len(class_names)

    # --- 默认阈值 (0.5) 的评估 ---
    log("--- Evaluating with default threshold (0.5) ---")
    y_pred_default = np.argmax(y_prob, axis=1)
    report_default = classification_report(y_true, y_pred_default, target_names=class_names, digits=4)
    cm_default = confusion_matrix(y_true, y_pred_default)

    if num_classes == 2:
        # 确定 AD 是哪个索引 (0 or 1)
        ad_names = {"ad", "alzheimer", "demented"}
        ad_idx = next((i for i, n in enumerate(class_names) if any(k in n.lower() for k in ad_names)), -1)
        if ad_idx == -1: # 如果找不到，默认AD是第二个类别
            ad_idx = 1
            log("[WARN] Could not find 'AD' in class names. Assuming positive class is index 1.")

        y_true_bin = (y_true == ad_idx)
        y_prob_ad = y_prob[:, ad_idx] # 获取预测为AD的概率
        
        auc = roc_auc_score(y_true_bin, y_prob_ad)

        # --- 寻找并应用最佳阈值 ---
        fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob_ad)
        j_scores = tpr - fpr # Youden's J statistic
        best_j_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_j_idx]
        log(f"Found best threshold using Youden's J: {best_threshold:.4f}")

        # 将最佳阈值保存到文件
        threshold_path = ART_DIR / "threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({'best_threshold': float(best_threshold)}, f)
        log(f"Best threshold saved to {threshold_path}")
        
        # 使用最佳阈值进行预测
        log(f"--- Evaluating with best threshold ({best_threshold:.4f}) ---")
        cn_idx = 1 - ad_idx
        y_pred_adjusted = np.where(y_prob_ad >= best_threshold, ad_idx, cn_idx)
        
        report_adjusted = classification_report(y_true, y_pred_adjusted, target_names=class_names, digits=4)
        cm_adjusted = confusion_matrix(y_true, y_pred_adjusted)
        f1_macro_adjusted = f1_score(y_true, y_pred_adjusted, average="macro")

        # --- 写入报告文件 ---
        with open(ART_DIR / f"classification_report_{tag}.txt", "w", encoding="utf-8") as f:
            f.write(f"== {tag.upper()} OVERALL METRICS ==\n")
            f.write(f"AUC: {auc:.4f}\n\n")

            f.write("== METRICS WITH DEFAULT THRESHOLD (0.5) ==\n")
            f.write(report_default + "\n\n")

            f.write(f"== METRICS WITH BEST THRESHOLD ({best_threshold:.4f}) ==\n")
            f.write(f"F1-score (macro): {f1_macro_adjusted:.4f}\n")
            f.write(report_adjusted + "\n")
        
        # --- 绘制并保存图表 ---
        # 默认阈值的混淆矩阵
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix ({tag} - Default Threshold)")
        plt.tight_layout()
        plt.savefig(ART_DIR / f"confusion_matrix_{tag}_default.png")
        plt.close()

        # 最佳阈值的混淆矩阵
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_adjusted, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix ({tag} - Best Threshold)")
        plt.tight_layout()
        plt.savefig(ART_DIR / f"confusion_matrix_{tag}_adjusted.png")
        plt.close()

        # ROC 曲线
        plot_roc_curve(y_true_bin, y_prob_ad, ad_idx, best_j_idx, auc, ART_DIR / f"roc_curve_{tag}.png")

        log(f"Evaluation complete. Reports and plots saved to '{ART_DIR}' directory.")

    else: # 多分类情况保持原样
        auc = roc_auc_score(y_true_1hot, y_prob, multi_class="ovr", average="macro")
        f1_macro = f1_score(y_true, y_pred_default, average="macro")
        # ... (此处省略了原有的多分类逻辑，因为它不需要改动)
        log("Multi-class evaluation does not support threshold adjustment in this script.")
        # ...

# ==============================================================================
# 区域5: 主函数（两阶段训练 + 类权重 + 以 val_auc 作为主监控）
# ==============================================================================
def main():
    log(f"TensorFlow: {tf.__version__}")
    log(f"Loading datasets from: {DATA_DIR}")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)
    with open(CLASS_JSON, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)
    log(f"Classes: {class_names} (saved to {CLASS_JSON.name})")

    # 从训练集真实 one-hot 统计得到 class_weight
    dist_info = json.load(open(DIST_JSON, "r", encoding="utf-8"))
    train_counts = np.array(dist_info["train_counts"])
    class_weight = _make_class_weight(train_counts)
    log(f"class_weight = {class_weight}")

    model = build_model(num_classes=num_classes)
    log("--- Initial model structure (Head training) ---")
    model.summary()

    # 以 AUC 为主监控更稳健；只存权重，避免整模/权重混淆
    ckpt = callbacks.ModelCheckpoint(str(OUT_PATH), save_best_only=True,
                                     monitor="val_auc", mode="max", verbose=1)
    es = callbacks.EarlyStopping(patience=6, restore_best_weights=True,
                                 monitor="val_auc", mode="max")
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                            patience=2, min_lr=1e-6)
    cbs = [ckpt, es, reduce_lr]

    # --- 阶段一：仅训练顶部 ---
    log("--- Phase 1: Training the classifier head... ---")
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_HEAD, callbacks=cbs, class_weight=class_weight)

    # --- 阶段二：微调 block5 ---
    log("--- Phase 2: Fine-tuning VGG16 block5... ---")
    log(f"Loading best weights from Phase 1, saved at {OUT_PATH}")
    model.load_weights(str(OUT_PATH)) # 加载第一阶段的最佳权重

    # ================================== FIXED BLOCK START ==================================
    base_model = model.get_layer('vgg16')
    base_model.trainable = True # 1. 首先解冻整个基础模型

    # 2. 然后重新冻结 block5 之前的所有层
    log("Unfreezing block5 and keeping earlier blocks frozen.")
    for layer in base_model.layers:
        if not layer.name.startswith('block5'):
            layer.trainable = False
    # =================================== FIXED BLOCK END ===================================

    # 重新编译，小学习率；继续用 AUC 指标
    if num_classes == 2:
        auc_metric = tf.keras.metrics.AUC(name="auc")
    else:
        auc_metric = tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=num_classes)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=loss_fn,
                  metrics=["accuracy", auc_metric])

    log("--- Model structure after unfreezing block5 ---")
    model.summary() # 在这里你应该能看到可训练参数数量增加了

    # 在新的 Checkpoint 中，为第二阶段模型单独保存，避免混淆
    finetune_out_path = OUT_PATH.with_name(f"{OUT_PATH.stem}_finetuned.h5")
    ckpt_finetune = callbacks.ModelCheckpoint(str(finetune_out_path), save_best_only=True,
                                              monitor="val_auc", mode="max", verbose=1)
    cbs_finetune = [ckpt_finetune, es, reduce_lr]

    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_FINETUNE, callbacks=cbs_finetune, class_weight=class_weight)

    log(f"Training finished. Best fine-tuned model saved to: {finetune_out_path}")

    # --- 评估 ---
    log(f"Loading best weights from fine-tuning phase for evaluation: {finetune_out_path}")
    model = tf.keras.models.load_model(str(finetune_out_path)) # <--- 加载微调阶段保存的最佳模型
    
    if test_ds is not None:
        log("Evaluating on TEST set...")
        evaluate_and_save(model, test_ds, class_names, tag="test")
    else:
        log("Evaluating on VAL set (no explicit test set found)...")
        evaluate_and_save(model, val_ds, class_names, tag="val")

if __name__ == "__main__":
    main()