"""Video embedders: VideoMAE-base and X-CLIP-base-patch32 (HF transformers).

These are TRUE video models: they ingest a clip of frames jointly and model temporal
structure, unlike the frame-level CNNs/ViTs which see each frame independently. That is
exactly the capability the AlexNet baseline lacks, so these are the models most likely
to add something over it IF the EEG carries motion/temporal information.

Both are true video models that ingest a clip of frames jointly.

16-frame mapping for VideoMAE (logged as a decision)
----------------------------------------------------
VideoMAE-base expects 16 frames; our shared sampler yields 8 (`pooled`) or 12
(`timeresolved`). We map onto 16 by NEAREST-neighbour index resampling along time:
`idx = round(linspace(0, n_in-1, 16))`. This repeats frames rather than interpolating
pixels, which keeps every input frame a real decoded frame (no synthetic blended
frames) and preserves temporal ordering. For `pooled` (8 in) each frame is seen ~2x.
We do NOT interpolate in pixel space because a blended frame is not something the model
saw in training and would inject motion blur the EEG never experienced.

X-CLIP-base-patch32 natively expects 8 frames (its MIT position embedding is fixed at
8), so for `pooled` (8 frames) it is a 1:1 mapping with no resampling, and for the
time-resolved windows we resample each window to 8 the same way.

pooled      : run the model once on the resampled clip -> one vector.
timeseries  : a video model has no native per-0.25-s output. We slide a window of
              `WIN` consecutive timeresolved frames (centred on each of the 12 grid
              points, edge-padded by repetition), resample each window to the model's
              frame count, and embed it -> (12, D). Each point summarises a short
              neighbourhood rather than an instant. Logged as a decision.
"""
from __future__ import annotations

import numpy as np

from winfuse.common.embed_api import BaseEmbedder, register_embedder
from .base_visual import frames_pooled, frames_timeresolved

MODEL_N_FRAMES = 16   # VideoMAE default; X-CLIP-base-patch32 overrides to 8


def _resample_to(frames: np.ndarray, n: int) -> np.ndarray:
    """Nearest-neighbour time resample (T,H,W,3) -> (n,H,W,3). Repeats, never blends."""
    t = frames.shape[0]
    idx = np.round(np.linspace(0, t - 1, n)).astype(int)
    return frames[idx]


def _windows(frames: np.ndarray, win: int):
    """Centred windows of `win` consecutive frames at each index, edge-padded."""
    t = frames.shape[0]
    half = win // 2
    padded = np.concatenate(
        [np.repeat(frames[:1], half, axis=0), frames,
         np.repeat(frames[-1:], win - half, axis=0)], axis=0)
    return [padded[i:i + win] for i in range(t)]


class _VideoEmbedder(BaseEmbedder):
    modality = "visual"
    agent = "V-A"
    frame_variant = "pooled"
    _WIN = 6  # consecutive timeresolved frames per sliding window (1.25 s span)
    _N_FRAMES = MODEL_N_FRAMES  # frames fed to the model; X-CLIP overrides to 8

    def _clip_vector(self, frames16: np.ndarray) -> tuple[np.ndarray, dict]:
        """Embed a single (N,H,W,3) uint8 clip -> (pooled_vec (D,), layers {name:(D,)})."""
        raise NotImplementedError

    def _embed(self, video_path: str) -> dict:
        # pooled: 8-frame clip -> N -> one vector + layers
        pooled_frames = frames_pooled(video_path)
        pooled_vec, layers = self._clip_vector(_resample_to(pooled_frames, self._N_FRAMES))

        # timeseries: slide a window over the 12 timeresolved frames
        tr_frames, times = frames_timeresolved(video_path)
        series = []
        for w in _windows(tr_frames, self._WIN):
            v, _ = self._clip_vector(_resample_to(w, self._N_FRAMES))
            series.append(v)
        ts = np.stack(series).astype(np.float32)
        return {"pooled": pooled_vec, "layers": layers, "timeseries": ts, "times": times}


@register_embedder
class VideoMAEBase(_VideoEmbedder):
    name = "videomae_base"
    description = ("VideoMAE-base (self-supervised masked video model): mean-pooled last "
                   "hidden state over the 16-frame clip + per-block mean tokens.")
    source = "MCG-NJU/videomae-base"
    revision = "dc740ceda42fce44faed2ea03c6d447db72f6af9"

    def _load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, VideoMAEModel

        self.model = VideoMAEModel.from_pretrained(
            self.source, revision=self.revision).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoImageProcessor.from_pretrained(self.source, revision=self.revision)
        self._torch = torch

    def _clip_vector(self, frames16: np.ndarray):
        torch = self._torch
        # HF VideoMAE processor expects a list of frames (list of HxWx3)
        inputs = self.proc([list(frames16)], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values, output_hidden_states=True)
            last = out.last_hidden_state                 # (1, n_tokens, D)
            pooled = last.mean(dim=1)[0].float().cpu().numpy()
            layers = {}
            for i, h in enumerate(out.hidden_states[1:], start=1):
                layers[f"block{i:02d}_mean"] = h.mean(dim=1)[0].float().cpu().numpy()
        return pooled, layers


@register_embedder
class XClipBase(_VideoEmbedder):
    name = "xclip_base"
    description = ("X-CLIP-base-patch32 vision side: MIT-integrated video embedding in the "
                   "video-text shared space (8-frame clip) + per-frame CLS mean.")
    source = "microsoft/xclip-base-patch32"
    revision = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
    _N_FRAMES = 8   # X-CLIP-base-patch32's MIT position embedding is fixed at 8 frames

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, XCLIPModel

        self.full = XCLIPModel.from_pretrained(
            self.source, revision=self.revision).eval().to(self.device)
        self.model = self.full  # for n_params
        for p in self.full.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(self.source, revision=self.revision)
        self._torch = torch

    def _clip_vector(self, frames16: np.ndarray):
        torch = self._torch
        # VideoMAEImageProcessor: positional list-of-videos; each video a list of frames
        pv = self.proc.image_processor(
            [list(frames16)], return_tensors="pt")["pixel_values"].to(self.device)
        b, nf, c, h, w = pv.shape
        with torch.no_grad():
            # Replicate get_video_features manually: the installed transformers' helper
            # assumes mit returns a dataclass but it returns a tuple here (version skew).
            vis = self.full.vision_model(pixel_values=pv.reshape(-1, c, h, w))
            cls = self.full.visual_projection(vis[1]).reshape(b, nf, -1)  # (b, nf, 512)
            mit = self.full.mit(cls)
            video_embed = (mit[1] if isinstance(mit, tuple) else mit.pooler_output)
            pooled = video_embed[0].float().cpu().numpy()        # (512,) MIT-integrated
            vision_mean = cls.mean(dim=1)[0].float().cpu().numpy()  # (512,) pre-MIT mean
        return pooled, {"vision_cls_mean": vision_mean, "mit_video": pooled}
