"""Agent V-A visual embedders.

Importing this package runs every @register_embedder in the submodules, which is how
`src.common.embed_api.discover_embedders()` finds them. Keep every embedder module
imported here.
"""
from __future__ import annotations

from . import torchvision_models  # noqa: F401  alexnet_layers, resnet50
from . import clip_model          # noqa: F401  clip_vit_b32
from . import dinov2_models       # noqa: F401  dinov2_small, dinov2_base
from . import video_models        # noqa: F401  videomae_base, xclip_base

__all__ = ["torchvision_models", "clip_model", "dinov2_models", "video_models"]
