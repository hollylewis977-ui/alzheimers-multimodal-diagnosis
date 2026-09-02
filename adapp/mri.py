# -*- coding: utf-8 -*-
"""MRI 影像分支：PyTorch ResNet50 二分类推理。

对应权重 artifacts/models/best_model.pth。
torch / torchvision 是可选依赖 —— 没装的话这个分支整体禁用，应用退化成纯 XGB，
不会启动失败。
"""
from __future__ import annotations

import io
import zipfile
from typing import Optional

from PIL import Image

from .config import IMAGE_SUFFIXES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MRI_RESNET_PATH


class MriPredictor:
    """封装 ResNet50 权重与预处理，对外只暴露两个预测入口。"""

    def __init__(self, model, transform, torch_module):
        self._model = model
        self._transform = transform
        self._torch = torch_module

    @classmethod
    def load(cls) -> "MriPredictor | None":
        """加载失败一律返回 None 并打印原因，调用方据此降级到纯 XGB。"""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
        except ImportError as exc:
            print(f"[MRI] 未安装 torch/torchvision，MRI 分支禁用：{exc}")
            return None

        if not MRI_RESNET_PATH.exists():
            print(f"[MRI] 未找到权重 {MRI_RESNET_PATH}，MRI 分支禁用")
            return None

        class AlzheimerClassifier(nn.Module):
            """ResNet50 骨干 + 单 logit 输出头，与训练时的结构一致。"""

            def __init__(self):
                super().__init__()
                self.model = models.resnet50(weights=None)
                self.model.fc = nn.Linear(self.model.fc.in_features, 1)

            def forward(self, x):
                return self.model(x).squeeze(1)

        try:
            ckpt = torch.load(MRI_RESNET_PATH, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

            model = AlzheimerClassifier()
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                # strict=False 会静默吞掉结构不匹配，这里显式报出来
                print(f"[MRI] 权重未完全匹配：缺失 {len(missing)} 项，多余 {len(unexpected)} 项")
            model.eval()

            transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        except Exception as exc:
            print(f"[MRI] 权重加载失败，MRI 分支禁用：{exc!r}")
            return None

        print(f"[MRI] 已加载 ResNet50 权重 {MRI_RESNET_PATH.name}")
        return cls(model, transform, torch)

    def predict_image(self, img_bytes: bytes) -> Optional[float]:
        """单张切片 -> AD 概率；图片损坏返回 None。"""
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            print(f"[MRI] 跳过无法解码的图片：{exc!r}")
            return None

        tensor = self._transform(image).unsqueeze(0)  # (1, 3, 224, 224)
        with self._torch.no_grad():
            logit = self._model(tensor)
            return float(self._torch.sigmoid(logit).item())

    def predict_zip(self, file_storage) -> tuple[Optional[float], Optional[str]]:
        """ZIP 内所有切片的 AD 概率取平均。

        返回 (概率, 错误信息)，两者恒有其一为 None。
        """
        try:
            with zipfile.ZipFile(file_storage, "r") as zf:
                names = [
                    n for n in zf.namelist()
                    if not n.endswith("/") and n.lower().endswith(IMAGE_SUFFIXES)
                ]
                if not names:
                    return None, "ZIP 内没有 .jpg/.jpeg/.png 切片"

                probs = [p for n in names if (p := self.predict_image(zf.read(n))) is not None]
                if not probs:
                    return None, "ZIP 内的图片都无法解码"
                return sum(probs) / len(probs), None
        except zipfile.BadZipFile:
            return None, "不是有效的 ZIP 文件"
        except Exception as exc:
            return None, str(exc)

    def predict_upload(self, file_storage) -> tuple[Optional[float], Optional[str]]:
        """按扩展名分派到 ZIP 或单图，供路由层直接调用。"""
        name = (file_storage.filename or "").lower()
        if name.endswith(".zip"):
            return self.predict_zip(file_storage)
        if name.endswith(IMAGE_SUFFIXES):
            raw = file_storage.read()
            file_storage.stream.seek(0)
            prob = self.predict_image(raw)
            return (prob, None) if prob is not None else (None, "图片无法解码")
        return None, "不支持的文件类型（请上传 .zip / .jpg / .png）"
