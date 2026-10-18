"""AST (Audio Spectrogram Transformer), AudioSet-finetuned — general-sound embedding.

MIT/ast-finetuned-audioset-10-10-0.4593. We take the encoder's pooled representation
(the [CLS]-equivalent via the model's pooler / mean of the last hidden state) as the
clip embedding, plus the patch-token sequence sub-sampled to the 0.25 s grid as a
time-resolved variant. AudioSet pretraining makes this a complementary general-sound
model to CLAP; the run driver picks which general-sound model(s) to actually run based
on what loads cleanly in 16 GB, and logs the choice.
"""
from __future__ import annotations

import numpy as np

from winfuse.common.embed_api import BaseEmbedder, register_embedder
from winfuse.common.frames import AUDIO_SR, extract_audio

from . import constants as K

_TARGET_T = K.TIMERES_N


@register_embedder
class AstAudioSet(BaseEmbedder):
    name = "ast_audioset"
    modality = "audio"
    description = ("AST (Audio Spectrogram Transformer) finetuned on AudioSet; mean of "
                   "the last hidden state as pooled, patch tokens sub-sampled to the "
                   "0.25 s grid as timeseries. General-sound semantics.")
    source = "MIT/ast-finetuned-audioset-10-10-0.4593"
    revision = "f826b80d28226b62986cc218e5cec390b1096902"
    agent = "V-B"
    frame_variant = "n/a (audio)"
    optional = True

    def _load(self) -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTModel

        self._torch = torch
        self.fe = ASTFeatureExtractor.from_pretrained(self.source, revision=self.revision)
        self.model = ASTModel.from_pretrained(self.source, revision=self.revision)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _embed(self, video_path: str) -> dict:
        torch = self._torch
        a = extract_audio(video_path, sr=AUDIO_SR)
        inputs = self.fe(a["waveform"], sampling_rate=AUDIO_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        seq = out.last_hidden_state[0].float().cpu().numpy()  # (n_tokens, D)
        pooled = seq.mean(0).astype(np.float32)
        T = seq.shape[0]
        idx = np.linspace(0, T - 1, _TARGET_T).round().astype(int)
        ts = seq[idx].astype(np.float32)
        times = np.arange(_TARGET_T) * K.TIMERES_STEP_S
        return {"pooled": pooled, "timeseries": ts, "times": times}
