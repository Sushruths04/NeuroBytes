"""Agent V-B: audio, handcrafted and joint stimulus embedders.

Importing this package runs @register_embedder on every embedder below, so
`src.common.embed_api.discover_embedders()` finds them without knowing their
internals.

Embedders owned here
--------------------
neural audio (MPS-capable, device-equivalence gated):
    wav2vec2_base       facebook/wav2vec2-base-960h, layer-wise mean-pooled
    whisper_tiny        openai/whisper-tiny encoder, mean-pooled + time-resolved
    whisper_base        openai/whisper-base encoder, mean-pooled + time-resolved
    clap_audio          laion/clap-htsat-unfused audio tower (general sound)
    ast_audioset        MIT/ast-finetuned-audioset-10-10-0.4593 (general sound)

handcrafted CPU descriptors (numpy / librosa / cv2, no device check needed):
    hc_audio            RMS, mel summary, MFCC+deltas, spectral shape, ZCR, onset, tempo
    hc_visual           luminance, HSV histograms, Gabor energy, edges, optic flow, cuts
    hc_av               concatenation of hc_audio + hc_visual pooled vectors

joint (concatenation of best visual + audio, z-scored on train only):
    joint_av            built at run time from available caches; see joint.py
"""
from __future__ import annotations

# Importing each module triggers its @register_embedder decorators.
from . import handcrafted  # noqa: F401
from . import wav2vec2     # noqa: F401
from . import whisper_enc  # noqa: F401
from . import clap_audio   # noqa: F401
from . import ast_audio    # noqa: F401
from . import joint        # noqa: F401

__all__ = ["handcrafted", "wav2vec2", "whisper_enc", "clap_audio", "ast_audio", "joint"]
