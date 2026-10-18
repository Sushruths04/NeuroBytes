"""CLIP ViT-B/32 image encoder (OpenAI, via HF transformers).

The pooled output is the projected image embedding (`get_image_features`, 512-d) —
the vector that lives in CLIP's shared image/text space, which is what makes CLIP
interesting here: it is a *semantic* representation, unlike AlexNet's conv features.

Per-block CLS states are also exposed (`output_hidden_states=True`): the CLS token of
each of the 12 transformer blocks (768-d, pre-projection). These let the comparison
ask at what depth CLIP's representation best aligns with EEG, analogous to the AlexNet
layer sweep. We expose the CLS token (not mean-patch) for CLIP because CLIP is trained
to put the whole-image summary in CLS via the final projection.

268 -> 224 is a DOWNsample; the HF CLIPImageProcessor does bicubic resize + centre
crop + CLIP normalisation, applied identically to every frame.
"""
from __future__ import annotations

import numpy as np

from winfuse.common.embed_api import BaseEmbedder, register_embedder
from .base_visual import FrameBackbone


@register_embedder
class ClipVitB32(FrameBackbone, BaseEmbedder):
    name = "clip_vit_b32"
    modality = "visual"
    description = ("OpenAI CLIP ViT-B/32 image encoder: projected 512-d image embedding "
                   "(shared space) + per-block CLS states, mean over 8 pooled frames.")
    source = "openai/clip-vit-base-patch32"
    revision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    agent = "V-A"
    frame_variant = "pooled"

    def _load(self) -> None:
        import torch
        from transformers import CLIPImageProcessor, CLIPModel

        self.model = CLIPModel.from_pretrained(
            self.source, revision=self.revision).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = CLIPImageProcessor.from_pretrained(self.source, revision=self.revision)
        self._torch = torch

    def _frame_features(self, frames: np.ndarray):
        torch = self._torch
        imgs = [f for f in frames]  # list of (H,W,3) uint8 RGB
        inputs = self.proc(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            vis = self.model.vision_model(pixel_values=pixel_values,
                                          output_hidden_states=True)
            # projected image features (shared CLIP space), per frame
            pooled_hidden = vis.pooler_output                       # (T, 768) post-LN CLS
            img_feats = self.model.visual_projection(pooled_hidden)  # (T, 512)
            img_feats = img_feats.float().cpu().numpy()
            # per-block CLS tokens (12 blocks + embedding layer); drop the embedding (idx 0)
            layers = {}
            hs = vis.hidden_states  # tuple len 13: [embeddings, block1..block12]
            for i, h in enumerate(hs[1:], start=1):
                layers[f"block{i:02d}_cls"] = h[:, 0, :].float().cpu().numpy()  # (T, 768)
        return img_feats, layers
