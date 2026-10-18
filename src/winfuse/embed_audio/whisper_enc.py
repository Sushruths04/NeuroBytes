"""Whisper encoder — mean-pooled + time-resolved audio representations.

Encoder only. We run WhisperModel.encoder on the log-mel features produced by the
model's own feature extractor, take the mean over the encoder time axis as `pooled`,
and sub-sample the encoder sequence to the 0.25 s grid as `timeseries`. No decoding,
no transcription — the decoder is never instantiated.

Two sizes are registered: whisper_tiny (384-dim) and whisper_base (512-dim).
"""
from __future__ import annotations

import numpy as np

from winfuse.common.embed_api import BaseEmbedder, register_embedder
from winfuse.common.frames import AUDIO_SR, extract_audio

from . import constants as K

_TARGET_T = K.TIMERES_N


class _WhisperEncoder(BaseEmbedder):
    modality = "audio"
    agent = "V-B"
    frame_variant = "n/a (audio)"

    def _load(self) -> None:
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel

        self._torch = torch
        self.fe = WhisperFeatureExtractor.from_pretrained(self.source, revision=self.revision)
        full = WhisperModel.from_pretrained(self.source, revision=self.revision)
        self.model = full.encoder.eval().to(self.device)  # encoder only
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _embed(self, video_path: str) -> dict:
        torch = self._torch
        a = extract_audio(video_path, sr=AUDIO_SR)
        feats = self.fe(a["waveform"], sampling_rate=AUDIO_SR, return_tensors="pt")
        inp = feats.input_features.to(self.device)
        with torch.no_grad():
            enc = self.model(inp).last_hidden_state[0]  # (T_enc, D)
        seq = enc.float().cpu().numpy()
        pooled = seq.mean(0).astype(np.float32)
        T = seq.shape[0]
        idx = np.linspace(0, T - 1, _TARGET_T).round().astype(int)
        ts = seq[idx].astype(np.float32)
        times = np.arange(_TARGET_T) * K.TIMERES_STEP_S
        return {"pooled": pooled, "timeseries": ts, "times": times}


@register_embedder
class WhisperTiny(_WhisperEncoder):
    name = "whisper_tiny"
    description = ("Whisper-tiny encoder (384-dim), mean-pooled + 0.25 s time-resolved. "
                   "Encoder only; no decoding.")
    source = "openai/whisper-tiny"
    revision = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"


@register_embedder
class WhisperBase(_WhisperEncoder):
    name = "whisper_base"
    description = ("Whisper-base encoder (512-dim), mean-pooled + 0.25 s time-resolved. "
                   "Encoder only; no decoding.")
    source = "openai/whisper-base"
    revision = "e37978b90ca9030d5170a5c07aadb050351a65bb"
