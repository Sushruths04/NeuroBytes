"""DINOv2 (small + base) image embedders, via HF transformers.

DINOv2 is self-supervised (no labels, no language), so its features are driven purely
by visual structure — a useful contrast to CLIP (language-aligned) and AlexNet/ResNet
(ImageNet-classification-aligned). It is strong at dense/spatial structure, which is
why we expose BOTH poolings:

    pooled      = concat(CLS, mean-patch) of the last layer
    layers      = per-block CLS AND per-block mean-patch tokens

CLS captures the global summary; mean-patch averages all patch tokens and tends to
carry more layout/texture information. Exposing both lets the comparison find which
DINOv2 reading aligns with EEG without us pre-judging it. The top-level timeseries
output uses the last-layer concat(CLS, mean-patch).

DINOv2 uses patch 14, so HF resizes to 224 (16x16 patches) — 268 -> 224 downsample,
bicubic, with ImageNet normalisation, applied identically per frame.
"""
from __future__ import annotations

import numpy as np

from winfuse.common.embed_api import BaseEmbedder, register_embedder
from .base_visual import FrameBackbone


class _Dinov2Base(FrameBackbone, BaseEmbedder):
    modality = "visual"
    agent = "V-A"
    frame_variant = "pooled"

    def _load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained(
            self.source, revision=self.revision).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoImageProcessor.from_pretrained(self.source, revision=self.revision)
        self._torch = torch

    def _frame_features(self, frames: np.ndarray):
        torch = self._torch
        imgs = [f for f in frames]
        inputs = self.proc(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values, output_hidden_states=True)
            layers: dict[str, np.ndarray] = {}
            hs = out.hidden_states  # tuple: [embeddings, block1..blockN]
            for i, h in enumerate(hs[1:], start=1):
                cls = h[:, 0, :]                 # (T, D)
                patch = h[:, 1:, :].mean(dim=1)  # mean over patch tokens (T, D)
                layers[f"block{i:02d}_cls"] = cls.float().cpu().numpy()
                layers[f"block{i:02d}_meanpatch"] = patch.float().cpu().numpy()
            last = out.last_hidden_state
            cls = last[:, 0, :]
            patch = last[:, 1:, :].mean(dim=1)
            pooled = torch.cat([cls, patch], dim=-1).float().cpu().numpy()  # (T, 2D)
        return pooled, layers


@register_embedder
class Dinov2Small(_Dinov2Base):
    name = "dinov2_small"
    description = ("DINOv2-small (self-supervised ViT-S/14): concat(CLS, mean-patch) 768-d, "
                   "per-block CLS and mean-patch, mean over 8 pooled frames.")
    source = "facebook/dinov2-small"
    revision = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


@register_embedder
class Dinov2BaseModel(_Dinov2Base):
    name = "dinov2_base"
    description = ("DINOv2-base (self-supervised ViT-B/14): concat(CLS, mean-patch) 1536-d, "
                   "per-block CLS and mean-patch, mean over 8 pooled frames.")
    source = "facebook/dinov2-base"
    revision = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
