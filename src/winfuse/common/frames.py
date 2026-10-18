"""ONE shared frame-sampling and audio-extraction routine. FROZEN CONTRACT.

Sub-agents must not reimplement this. Every visual model must see identical frames
and every audio model identical waveforms, otherwise differences between models are
confounded with differences in their input.

Why sampling is by TIMESTAMP, not frame index
---------------------------------------------
Measured in reports/10_stimulus_inspection.md: fps is NOT constant across the
corpus. There are 39 distinct fps values spanning 15.0 to 30.5, and container frame
counts range from 45 to 92.

How much this actually matters was measured rather than assumed. Because every clip
is ~3.0 s, frame-index spacing happens to approximate time spacing, so the
organizers' `linspace(0, n_frames-1, 8)` is not badly wrong: on 60 random training
videos its timestamps deviate from even time spacing by 33 ms on average (59 ms worst
case), and the last sampled timestamp varies by 87 ms across videos.

That is negligible for a pooled embedding averaged over 8 frames. It is not
negligible for the time-resolved variant: 33-87 ms is 8-22 samples at the EEG's
250 Hz, and the whole point of the latency/layer analysis is to attribute effects to
specific EEG windows. Timestamp addressing costs nothing and removes the ambiguity,
so it is the default; `baseline` keeps index addressing purely for faithful
reproduction.

Variants
--------
`pooled`        8 frames at the centres of 8 equal bins across 0-3 s
                -> t = 0.1875, 0.5625, ..., 2.8125 s
                Bin centres rather than `linspace(0, 3, 8)` because t=0 is often a
                fade-in/black frame and t=3.0 is past the last frame of a 2.973 s video.
`timeresolved`  12 frames at 0.25 s steps -> t = 0.0, 0.25, ..., 2.75 s
                These align to EEG sliding windows for the latency/layer analysis.
                t=0 is kept here deliberately: the alignment to EEG window 0 matters
                more than the risk of a dark first frame, and the EEG epoch also
                starts at onset.
`baseline`      frame-index `linspace(0, n_frames-1, 8)`, i.e. exactly what the
                organizers' notebook does. Provided ONLY so the reference row
                `baseline_alexnet` is a faithful reproduction, including this flaw.

Audio
-----
Decoded via ffmpeg to 16 kHz mono float32 and padded/trimmed to exactly 3.000 s
(48000 samples). Necessary because the corpus has 10 distinct source sample rates
(8000-96000 Hz) and a mix of mono and stereo. 34 videos have no audio stream and 9
more are digitally silent; both get a zero waveform and `is_silent=True`, which
callers must record so audio results can be recomputed excluding them.
"""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants (frozen)
EPOCH_DUR_S = 3.0           # the EEG epoch length; defines the sampling window

POOLED_N_FRAMES = 8
TIMERES_STEP_S = 0.25
TIMERES_N_FRAMES = 12       # 0.00, 0.25, ..., 2.75
BASELINE_N_FRAMES = 8       # organizers' notebook

AUDIO_SR = 16000
AUDIO_N_SAMPLES = int(AUDIO_SR * EPOCH_DUR_S)   # 48000
SILENT_RMS_THRESHOLD = 1e-5                      # matches reports/10 §1

VARIANTS = ("pooled", "timeresolved", "baseline")


def frame_times(variant: str) -> np.ndarray:
    """Timestamps in seconds at which frames are sampled. `baseline` is index-based."""
    if variant == "pooled":
        edges = np.linspace(0.0, EPOCH_DUR_S, POOLED_N_FRAMES + 1)
        return (edges[:-1] + edges[1:]) / 2.0
    if variant == "timeresolved":
        return np.arange(TIMERES_N_FRAMES, dtype=np.float64) * TIMERES_STEP_S
    if variant == "baseline":
        # Not a true timestamp grid; the notebook uses frame indices. Returned for
        # bookkeeping only, computed per-video in sample_frames.
        return np.full(BASELINE_N_FRAMES, np.nan)
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


@lru_cache(maxsize=4096)
def video_meta(path: str) -> dict:
    """fps / frame count / duration from ffprobe. Cached per path."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-print_format", "json",
           "-show_streams", "-show_format", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr[:200]}")
    import json as _json

    info = _json.loads(r.stdout)
    s = (info.get("streams") or [{}])[0]
    fmt = info.get("format", {})

    def frac(x):
        from fractions import Fraction

        try:
            return float(Fraction(x)) if x and x != "0/0" else None
        except Exception:
            return None

    nb = s.get("nb_frames")
    return {
        "fps": frac(s.get("avg_frame_rate")) or frac(s.get("r_frame_rate")),
        "nb_frames": int(nb) if str(nb).isdigit() else None,
        "duration_s": float(fmt["duration"]) if fmt.get("duration") else None,
        "width": s.get("width"),
        "height": s.get("height"),
    }


def sample_frames(video_path: str | Path, variant: str = "pooled") -> dict:
    """Decode frames at the variant's sample points.

    Returns
    -------
    dict with
        frames : uint8 (T, H, W, 3) in **RGB** order
        times  : float64 (T,) seconds from video onset (NaN for `baseline`)
        n_requested, n_substituted : int
            `n_substituted` counts sample points where the seek produced no frame and
            the last good frame was reused. Non-zero values are recorded in the cache
            manifest so no embedding is silently computed on a duplicated frame.
        meta   : the ffprobe metadata used
    """
    import cv2

    video_path = str(video_path)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    meta = video_meta(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    try:
        # Decode the whole clip once. These are 3 s / 268x268 / <=92 frames files, so
        # a full decode is ~20 MB and far more reliable than per-frame seeking, which
        # is inexact on h264 with non-integer fps.
        decoded = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            decoded.append(fr)
    finally:
        cap.release()

    if not decoded:
        raise RuntimeError(f"no decodable frames in {video_path}")

    n_dec = len(decoded)
    dur = meta.get("duration_s") or EPOCH_DUR_S
    fps_eff = n_dec / dur if dur else (meta.get("fps") or 30.0)

    if variant == "baseline":
        # Faithful reproduction of the notebook: index-based, ignores fps.
        idx = np.linspace(0, n_dec - 1, BASELINE_N_FRAMES).astype(int)
        times = idx / fps_eff
    else:
        times = frame_times(variant)
        idx = np.round(times * fps_eff).astype(int)

    n_sub = int(np.sum(idx > n_dec - 1))
    idx = np.clip(idx, 0, n_dec - 1)

    frames = np.stack([decoded[i][:, :, ::-1] for i in idx])  # BGR -> RGB
    return {
        "frames": np.ascontiguousarray(frames, dtype=np.uint8),
        "times": np.asarray(times, dtype=np.float64),
        "n_requested": int(len(idx)),
        "n_substituted": n_sub,
        "n_decoded": n_dec,
        "fps_effective": float(fps_eff),
        "meta": meta,
    }


def extract_audio(video_path: str | Path, sr: int = AUDIO_SR) -> dict:
    """Decode audio to mono float32 at `sr`, exactly EPOCH_DUR_S long.

    Returns
    -------
    dict with
        waveform  : float32 (sr * 3,)  zero-filled if there is no usable audio
        is_silent : bool   True if no stream, decode failure, or RMS < 1e-5
        rms, peak : float
        reason    : str    why it is silent, when it is
    """
    video_path = str(video_path)
    n_target = int(sr * EPOCH_DUR_S)
    cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    zero = np.zeros(n_target, dtype=np.float32)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        return {"waveform": zero, "is_silent": True, "rms": 0.0, "peak": 0.0,
                "reason": f"ffmpeg raised {type(e).__name__}", "n_source_samples": 0}
    if r.returncode != 0 or not r.stdout:
        return {"waveform": zero, "is_silent": True, "rms": 0.0, "peak": 0.0,
                "reason": "no audio stream or ffmpeg decode failure",
                "n_source_samples": 0}

    y = np.frombuffer(r.stdout, dtype=np.float32).copy()
    n_src = int(y.size)
    if n_src == 0:
        return {"waveform": zero, "is_silent": True, "rms": 0.0, "peak": 0.0,
                "reason": "empty audio stream", "n_source_samples": 0}

    # pad with zeros / trim to exactly 3 s
    if n_src < n_target:
        y = np.concatenate([y, np.zeros(n_target - n_src, dtype=np.float32)])
    elif n_src > n_target:
        y = y[:n_target]

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak = float(np.abs(y).max())
    silent = rms < SILENT_RMS_THRESHOLD
    return {
        "waveform": np.ascontiguousarray(y, dtype=np.float32),
        "is_silent": bool(silent),
        "rms": rms,
        "peak": peak,
        "reason": "digitally silent (RMS < 1e-5)" if silent else "",
        "n_source_samples": n_src,
    }


def sampling_spec() -> dict:
    """The exact sampling configuration, for the cache manifest."""
    return {
        "epoch_dur_s": EPOCH_DUR_S,
        "pooled": {"n_frames": POOLED_N_FRAMES,
                   "times_s": frame_times("pooled").round(6).tolist(),
                   "rule": "centres of 8 equal bins across 0-3 s"},
        "timeresolved": {"n_frames": TIMERES_N_FRAMES, "step_s": TIMERES_STEP_S,
                         "times_s": frame_times("timeresolved").round(6).tolist(),
                         "rule": "0.25 s grid from 0.0 s"},
        "baseline": {"n_frames": BASELINE_N_FRAMES,
                     "rule": "frame-index linspace(0, n_frames-1, 8); reproduces the "
                             "organizers' notebook, which ignores fps"},
        "addressing": "by timestamp, converted to a frame index via the effective fps "
                      "measured as n_decoded_frames / container_duration",
        "colour_order": "RGB",
        "audio": {"sr": AUDIO_SR, "channels": 1, "n_samples": AUDIO_N_SAMPLES,
                  "pad_trim": "zero-pad or trim to exactly 3.000 s",
                  "silent_rms_threshold": SILENT_RMS_THRESHOLD},
    }
