"""wav2vec2-base — layer-wise mean-pooled speech representations.

facebook/wav2vec2-base-960h, encoder run with output_hidden_states=True to expose all
13 hidden states (embeddings + 12 transformer layers). We mean-pool each layer over
time for the `layers` dict, use the last hidden state's mean as `pooled`, and keep the
last hidden state's full sequence (sub-sampled to the 0.25 s grid) as `timeseries`.

MPS is permitted; device is resolved from EEG_EMBED_DEVICE / resolve_device(). The run
driver calls verify_device_equivalence before the full extraction and records the
verdict in the manifest; on failure it falls back to CPU.
"""
from __future__ import annotations

import numpy as np

from winfuse.common.embed_api import BaseEmbedder, register_embedder
from winfuse.common.frames import AUDIO_SR, extract_audio

from . import constants as K

_TARGET_T = K.TIMERES_N  # 12-step grid for the timeseries


@register_embedder
class Wav2Vec2Base(BaseEmbedder):
    name = "wav2vec2_base"
    modality = "audio"
    description = ("wav2vec2-base-960h encoder; 13 hidden states each mean-pooled over "
                   "time (layers), last hidden state mean-pooled (pooled) and sub-sampled "
                   "to the 0.25 s grid (timeseries). Encoder only, no CTC decoding.")
    source = "facebook/wav2vec2-base-960h"
    revision = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
    agent = "V-B"
    frame_variant = "n/a (audio)"

    def _load(self) -> None:
        import torch
        from transformers import Wav2Vec2Model

        self._torch = torch
        self.model = Wav2Vec2Model.from_pretrained(self.source, revision=self.revision)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _embed(self, video_path: str) -> dict:
        torch = self._torch
        a = extract_audio(video_path, sr=AUDIO_SR)
        wav = torch.from_numpy(a["waveform"].astype(np.float32)).to(self.device)[None, :]
        with torch.no_grad():
            out = self.model(wav, output_hidden_states=True)
        hidden = out.hidden_states  # tuple of (1, T, 768), length 13
        layers = {}
        for i, h in enumerate(hidden):
            layers[f"L{i:02d}"] = h[0].mean(0).float().cpu().numpy().astype(np.float32)
        last = hidden[-1][0].float().cpu().numpy()  # (T, 768)
        pooled = last.mean(0).astype(np.float32)

        # sub-sample the time axis to 12 steps for the timeseries variant
        T = last.shape[0]
        if T >= _TARGET_T:
            idx = np.linspace(0, T - 1, _TARGET_T).round().astype(int)
            ts = last[idx]
        else:
            ts = np.vstack([last, np.repeat(last[-1:], _TARGET_T - T, axis=0)])
        times = np.arange(_TARGET_T) * K.TIMERES_STEP_S
        return {"pooled": pooled, "timeseries": ts.astype(np.float32),
                "times": times, "layers": layers}
