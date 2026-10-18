"""CLAP audio tower — general-sound embedding.

laion/clap-htsat-unfused. We use the audio projection output (get_audio_features), a
single pooled vector per clip in the shared text-audio space. CLAP captures general
acoustic semantics (music, ambience, environmental sound) rather than speech phonetics,
complementing wav2vec2/Whisper.

Pooled only — CLAP's audio tower produces one projected vector per clip, not a
per-timestep sequence usable on the 0.25 s grid without re-engineering the model, so no
timeseries is emitted (documented in a decision).

Sample rate: CLAP's feature extractor was trained at 48 kHz and rejects 16 kHz input.
We resample the shared 16 kHz waveform up to 48 kHz with librosa before feeding it in.
This is a pure resample of the identical shared, normalised 3 s waveform — no new audio
decode — so CLAP still sees the same content every other audio model sees.
"""
from __future__ import annotations

import numpy as np

from winfuse.common.embed_api import BaseEmbedder, register_embedder
from winfuse.common.frames import AUDIO_SR, extract_audio


@register_embedder
class ClapAudio(BaseEmbedder):
    name = "clap_audio"
    modality = "audio"
    description = ("CLAP (laion/clap-htsat-unfused) audio tower; projected pooled "
                   "embedding in the shared audio-text space. General-sound semantics. "
                   "Pooled only.")
    source = "laion/clap-htsat-unfused"
    revision = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
    agent = "V-B"
    frame_variant = "n/a (audio)"
    optional = True

    def _load(self) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        self._torch = torch
        self.proc = ClapProcessor.from_pretrained(self.source, revision=self.revision)
        self.model = ClapModel.from_pretrained(self.source, revision=self.revision)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _embed(self, video_path: str) -> dict:
        import librosa

        torch = self._torch
        a = extract_audio(video_path, sr=AUDIO_SR)
        # CLAP's feature extractor was trained at 48 kHz; the shared extractor gives
        # 16 kHz, so resample up here. This is a pure resample of the SAME shared
        # waveform, so every model still sees the identical normalised audio content.
        wav48 = librosa.resample(a["waveform"].astype(np.float64),
                                 orig_sr=AUDIO_SR, target_sr=48000).astype(np.float32)
        inputs = self.proc(audio=wav48, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_audio_features(**inputs)[0]
        pooled = feats.float().cpu().numpy().astype(np.float32)
        return {"pooled": pooled}
