"""torchvision CNN embedders: AlexNet (all layers) and ResNet-50 (block-wise).

AlexNet
-------
`baseline_alexnet` (orchestrator) uses only `features.3` with `variant="baseline"` to
reproduce the organizers' notebook. This is the PROPER version, `alexnet_layers`:
it uses `variant="pooled"` and exposes every named stage of the architecture, so the
EEG-vs-feature comparison can ask *which depth* of AlexNet aligns with EEG rather than
being locked to one early conv layer.

Layers exposed (all spatially global-average-pooled to a vector per frame):
    conv1 = features[2]  (ReLU after conv1)      96-d
    conv2 = features[5]  (ReLU after conv2)     256-d
    conv3 = features[7]  (ReLU after conv3)     384-d
    conv4 = features[9]  (ReLU after conv4)     384-d
    conv5 = features[12] (ReLU after conv5)     256-d
    fc6   = classifier[2] (ReLU after fc6)     4096-d
    fc7   = classifier[5] (ReLU after fc7)     4096-d
The top-level `pooled`/`timeseries` output is fc7, the richest penultimate feature.

ResNet-50
---------
Weights IMAGENET1K_V2 (the stronger recipe). Layers exposed, each global-avg-pooled:
    layer1  256-d, layer2 512-d, layer3 1024-d, layer4 2048-d, final (avgpool) 2048-d.
Top-level pooled/timeseries output is `final` (2048-d), the standard image feature.
"""
from __future__ import annotations

import numpy as np

from winfuse.common.embed_api import BaseEmbedder, register_embedder
from .base_visual import FrameBackbone


def _to_batch(frames: np.ndarray, tf, device):
    """Apply torchvision transforms to (T,H,W,3) uint8 RGB -> (T,3,224,224) on device."""
    import torch

    return torch.stack(
        [tf(torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)) for f in frames]
    ).to(device)


@register_embedder
class AlexNetLayers(FrameBackbone, BaseEmbedder):
    name = "alexnet_layers"
    modality = "visual"
    description = ("torchvision AlexNet (ImageNet), all conv1-5 + fc6/fc7 layers, "
                   "global-avg-pooled per frame, mean over 8 pooled-variant frames.")
    source = "torchvision.models.alexnet(AlexNet_Weights.IMAGENET1K_V1)"
    revision = "torchvision-0.29.1"
    agent = "V-A"
    frame_variant = "pooled"

    # feature-module split points: idx in model.features -> layer name
    _FEAT_TAPS = {2: "conv1", 5: "conv2", 7: "conv3", 9: "conv4", 12: "conv5"}
    _CLS_TAPS = {2: "fc6", 5: "fc7"}

    def _load(self) -> None:
        import torch
        import torchvision

        w = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        self.model = torchvision.models.alexnet(weights=w).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tf = w.transforms()
        self._torch = torch

    def _frame_features(self, frames: np.ndarray):
        torch = self._torch
        batch = _to_batch(frames, self.tf, self.device)
        conv_feats: dict[str, np.ndarray] = {}
        with torch.no_grad():
            x = batch
            for i, layer in enumerate(self.model.features):
                x = layer(x)
                if i in self._FEAT_TAPS:
                    # global average pool over spatial dims -> (T, C)
                    conv_feats[self._FEAT_TAPS[i]] = (
                        x.mean(dim=(2, 3)).float().cpu().numpy())
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            cls_feats: dict[str, np.ndarray] = {}
            for i, layer in enumerate(self.model.classifier):
                x = layer(x)
                if i in self._CLS_TAPS:
                    cls_feats[self._CLS_TAPS[i]] = x.float().cpu().numpy()

        layers = {**conv_feats, **cls_feats}
        pooled_per_frame = layers["fc7"]        # top-level feature = penultimate fc7
        return pooled_per_frame, layers


@register_embedder
class ResNet50Layers(FrameBackbone, BaseEmbedder):
    name = "resnet50"
    modality = "visual"
    description = ("torchvision ResNet-50 (ImageNet V2), layer1-4 + final avgpool, "
                   "global-avg-pooled per frame, mean over 8 pooled-variant frames.")
    source = "torchvision.models.resnet50(ResNet50_Weights.IMAGENET1K_V2)"
    revision = "torchvision-0.29.1"
    agent = "V-A"
    frame_variant = "pooled"

    def _load(self) -> None:
        import torch
        import torchvision

        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=w).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tf = w.transforms()
        self._torch = torch

    def _frame_features(self, frames: np.ndarray):
        torch = self._torch
        m = self.model
        batch = _to_batch(frames, self.tf, self.device)
        with torch.no_grad():
            x = m.conv1(batch)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)
            feats: dict[str, np.ndarray] = {}
            x = m.layer1(x); feats["layer1"] = x.mean(dim=(2, 3)).float().cpu().numpy()
            x = m.layer2(x); feats["layer2"] = x.mean(dim=(2, 3)).float().cpu().numpy()
            x = m.layer3(x); feats["layer3"] = x.mean(dim=(2, 3)).float().cpu().numpy()
            x = m.layer4(x); feats["layer4"] = x.mean(dim=(2, 3)).float().cpu().numpy()
            x = m.avgpool(x)
            final = torch.flatten(x, 1).float().cpu().numpy()
            feats["final"] = final
        return final, feats
