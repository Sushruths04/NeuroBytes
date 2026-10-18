"""Shared constants for the V-B audio/handcrafted embedders.

The silent-video IDs are copied verbatim from reports/10_stimulus_inspection.md
(measured, not re-derived here) so every audio embedder reports the same list.
"""
from __future__ import annotations

# ---- the 43 videos with no usable audio (reports/10 §4), as plain ints -----------
# 34 with no audio stream at all:
NO_AUDIO_STREAM_IDS = [
    8, 14, 40, 207, 211, 230, 232, 374, 385, 392, 406, 478, 487, 491, 493, 501,
    516, 533, 551, 552, 567, 593, 611, 660, 718, 812, 828, 916, 947, 950, 963,
    1000, 1015, 1102,
]
# 9 digitally silent (RMS < 1e-5):
SILENT_TRACK_IDS = [110, 214, 625, 674, 237, 243, 263, 390, 1043]

# All 43, sorted and unique.
SILENT_IDS = sorted(set(NO_AUDIO_STREAM_IDS) | set(SILENT_TRACK_IDS))

# Of these, the ones in the 102-video test pool (1001..1102).
SILENT_IDS_TEST = sorted(v for v in SILENT_IDS if 1001 <= v <= 1102)   # [1015, 1043, 1102]

# The brief singles out these two no-audio-stream test videos.
NO_AUDIO_TEST_IDS = sorted(v for v in NO_AUDIO_STREAM_IDS if 1001 <= v <= 1102)  # [1015, 1102]

assert len(SILENT_IDS) == 43, f"expected 43 silent ids, got {len(SILENT_IDS)}"

# ---- time grids (match src.common.frames) ----------------------------------------
EPOCH_DUR_S = 3.0
AUDIO_SR = 16000
# 0.25 s grid, 12 steps 0.00..2.75, aligned with frames.timeresolved and EEG windows.
TIMERES_STEP_S = 0.25
TIMERES_N = 12

# librosa STFT settings chosen so the hop is an integer divisor of a 0.25 s window:
#   0.25 s at 16 kHz = 4000 samples; hop 2000 (0.125 s) gives 2 frames per 0.25 s bin,
#   which averages cleanly onto the 12-step grid. n_fft 2048 (~128 ms) resolves low
#   frequencies without being longer than the quarter-second window.
HOP_LENGTH = 2000       # 0.125 s at 16 kHz
N_FFT = 2048
N_MELS = 40
N_MFCC = 20
