"""Shared helpers for Agent V-A visual embedders.

Every visual embedder consumes frames from `src.common.frames.sample_frames` and
produces three kinds of output through the frozen `BaseEmbedder` contract:

    pooled      (D,)      mean over the 8 `pooled`-variant frames
    timeseries  (T, D)    one vector per `timeresolved`-variant frame (T=12 @ 0.25 s)
    layers      {name:(D,)} per-layer pooled vectors, mean over the 8 pooled frames

Frame-level models (AlexNet, ResNet, CLIP image, DINOv2) run the backbone on each
frame independently and then pool over frames. Video models (VideoMAE, X-CLIP) ingest
a clip of frames at once; their `timeseries` is produced by sliding the clip, which
each video model documents in its own module.

The two frame variants are FIXED by the shared sampler and must never be re-rolled
here. Resizing 268 -> 224 is a DOWNsample (constraint note in the brief): we use
bilinear interpolation with antialias=True, which is what torchvision/HF processors do.
"""
from __future__ import annotations

import numpy as np

from winfuse.common.frames import sample_frames


def frames_pooled(video_path: str) -> np.ndarray:
    """8 RGB uint8 frames (8, 268, 268, 3) from the fixed `pooled` variant."""
    return sample_frames(video_path, variant="pooled")["frames"]


def frames_timeresolved(video_path: str):
    """12 RGB uint8 frames + their timestamps from the fixed `timeresolved` variant."""
    out = sample_frames(video_path, variant="timeresolved")
    return out["frames"], out["times"]


def pool_frames(per_frame: np.ndarray) -> np.ndarray:
    """Mean over the frame axis. per_frame is (T, D) -> (D,)."""
    return np.asarray(per_frame, dtype=np.float32).mean(axis=0)


class FrameBackbone:
    """Mixin for frame-level CNN/ViT backbones.

    Subclasses implement `_frame_features(frames_uint8) -> (pooled_per_frame (T,Dp),
    layers_per_frame {name:(T,Dl)})`. This mixin turns that into the contract dict for
    both the pooled and the time-resolved variant.
    """

    def _frame_features(self, frames: np.ndarray):
        raise NotImplementedError

    def _embed(self, video_path: str) -> dict:
        # pooled variant -> pooled vector + layer vectors (mean over 8 frames)
        pf, lf = self._frame_features(frames_pooled(video_path))
        pooled = pool_frames(pf)
        layers = {k: pool_frames(v) for k, v in lf.items()}

        # time-resolved variant -> (T, D) using the top-level pooled feature per frame
        tr_frames, times = frames_timeresolved(video_path)
        tpf, _ = self._frame_features(tr_frames)
        return {
            "pooled": pooled,
            "layers": layers,
            "timeseries": np.ascontiguousarray(tpf, dtype=np.float32),
            "times": times,
        }
