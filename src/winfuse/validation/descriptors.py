"""Metric 4a — cheap CPU video/audio stimulus descriptors, cached to disk.

For each stimulus mp4 we compute a small fixed-length descriptor vector that is
cheap on a CPU and plausibly reflected in the EEG. Descriptors are cached per
stimulus id to `cache/descriptors/<id>.json` so repeated runs are fast.

Descriptor set (extends config.DESCRIPTOR_NAMES; the extension is logged):
  audio_rms_mean        mean RMS of the audio waveform (loudness)
  audio_rms_std         RMS variability over time
  audio_onset_strength  mean librosa onset strength (transient density / rhythm)
  luminance_mean        mean frame luminance
  luminance_std         luminance variability across frames (global flicker)
  motion_energy_mean    mean abs frame-to-frame difference (motion)
  motion_energy_std     motion variability
  contrast_mean         mean per-frame std of luminance (spatial contrast)

Audio is decoded with ffmpeg (, PATH, or imageio-ffmpeg's static binary) to a mono wav that
librosa reads; video frames via OpenCV. If a stimulus has no audio stream the
audio_* fields are 0 and a flag is recorded.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from winfuse.common.config import CACHE_DIR, DESCRIPTOR_NAMES
from winfuse.common import io

DESC_CACHE = CACHE_DIR / "descriptors"
DESC_CACHE.mkdir(parents=True, exist_ok=True)

# Final descriptor ordering actually used (equals config order here, but explicit).
DESCRIPTORS = list(DESCRIPTOR_NAMES)

def _find_ffmpeg():
    """Portable ffmpeg lookup: $FFMPEG, then PATH, then the static binary from imageio-ffmpeg (pip)."""
    import shutil
    p = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _find_ffmpeg()
AUDIO_SR = 22050  # librosa default-ish; downsample for speed


def _video_descriptors(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    lum_means, lum_stds, motion = [], [], []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lum_means.append(float(gray.mean()))
        lum_stds.append(float(gray.std()))
        if prev is not None:
            motion.append(float(np.abs(gray - prev).mean()))
        prev = gray
    cap.release()
    lum_means = np.asarray(lum_means) if lum_means else np.zeros(1)
    lum_stds = np.asarray(lum_stds) if lum_stds else np.zeros(1)
    motion = np.asarray(motion) if motion else np.zeros(1)
    return {
        "luminance_mean": float(lum_means.mean()),
        "luminance_std": float(lum_means.std()),
        "motion_energy_mean": float(motion.mean()),
        "motion_energy_std": float(motion.std()),
        "contrast_mean": float(lum_stds.mean()),
    }


def _audio_descriptors(path: Path) -> dict:
    import librosa

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            FFMPEG, "-y", "-v", "error", "-i", str(path),
            "-ac", "1", "-ar", str(AUDIO_SR), "-vn", tmp.name,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or Path(tmp.name).stat().st_size == 0:
            return {"audio_rms_mean": 0.0, "audio_rms_std": 0.0,
                    "audio_onset_strength": 0.0, "_no_audio": True}
        y, sr = librosa.load(tmp.name, sr=AUDIO_SR, mono=True)
    if y.size == 0:
        return {"audio_rms_mean": 0.0, "audio_rms_std": 0.0,
                "audio_onset_strength": 0.0, "_no_audio": True}
    rms = librosa.feature.rms(y=y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    return {
        "audio_rms_mean": float(rms.mean()),
        "audio_rms_std": float(rms.std()),
        "audio_onset_strength": float(onset.mean()),
        "_no_audio": False,
    }


def compute_one(stim_id: int, split: str = "train") -> dict:
    """Compute (or load cached) descriptor dict for one stimulus id."""
    cp = DESC_CACHE / f"{split}_{int(stim_id):04d}.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except Exception:
            pass
    path = io.train_video_path(stim_id) if split == "train" else io.test_video_path(stim_id)
    d = {}
    d.update(_audio_descriptors(path))
    d.update(_video_descriptors(path))
    cp.write_text(json.dumps(d))
    return d


def matrix_for_ids(ids: list[int], split: str = "train") -> tuple[np.ndarray, list[str]]:
    """Return (D, descriptor_names) with D = (len(ids), n_descriptors), rows aligned to ids."""
    rows = []
    for sid in ids:
        d = compute_one(sid, split)
        rows.append([float(d.get(k, 0.0)) for k in DESCRIPTORS])
    return np.asarray(rows, dtype=np.float32), list(DESCRIPTORS)
