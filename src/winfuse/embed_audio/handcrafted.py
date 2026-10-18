"""Handcrafted CPU descriptors — the richer stimulus representation Phase 1 lacked.

Phase 1 used four scalars (audio_rms, audio_onset_strength, luminance, motion_energy)
plus contrast. The comparison report quantifies how much more these capture.

Everything here is numpy / librosa / cv2 on CPU. There is no neural network and no
torch device, so `verify_device_equivalence` is not applicable; the manifest records
`device="cpu"` and `mps_cpu_check={"applicable": false, ...}` explicitly.

Both a `pooled` vector and a `timeseries` on the 0.25 s grid (12 steps) are produced,
exactly as the brief requires. The time grid matches `src.common.frames.timeresolved`
and the EEG sliding windows.

Determinism: all operations are deterministic given the input; librosa.beat.tempo and
the optic-flow routine use no RNG. SEED is set at import for defensiveness.
"""
from __future__ import annotations

import numpy as np

from winfuse.common import io as cio
from winfuse.common.config import SEED
from winfuse.common.embed_api import BaseEmbedder, register_embedder
from winfuse.common.frames import AUDIO_SR, extract_audio, sample_frames

from . import constants as K

np.random.seed(SEED)

_EPS = 1e-8


# =============================================================================== audio
def _resample_to_grid(times_src: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Average/interpolate a per-STFT-frame series onto the 12-step 0.25 s grid.

    vals: (n_src,) or (n_src, D). Returns (12,) or (12, D). For each 0.25 s bin we take
    the mean of the source frames whose centre falls in the bin; empty bins fall back
    to nearest-neighbour so no NaN is produced for very short series.
    """
    grid = np.arange(K.TIMERES_N) * K.TIMERES_STEP_S
    vals = np.asarray(vals, dtype=np.float64)
    scalar = vals.ndim == 1
    if scalar:
        vals = vals[:, None]
    out = np.zeros((K.TIMERES_N, vals.shape[1]), dtype=np.float64)
    for i, t0 in enumerate(grid):
        t1 = t0 + K.TIMERES_STEP_S
        m = (times_src >= t0) & (times_src < t1)
        if m.any():
            out[i] = vals[m].mean(0)
        else:
            j = int(np.argmin(np.abs(times_src - (t0 + K.TIMERES_STEP_S / 2))))
            out[i] = vals[j]
    return out[:, 0] if scalar else out


def _audio_features(y: np.ndarray, sr: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return (pooled_dict, timeseries (12,Dt), frame_times) for one waveform.

    A silent waveform (all zeros) flows through: every librosa call returns finite
    zeros/constants, so the embedding is finite — verified in the smoke check.
    """
    import librosa

    y = np.asarray(y, dtype=np.float64)
    hop, n_fft = K.HOP_LENGTH, K.N_FFT

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + _EPS      # (1+n_fft/2, T)
    frame_t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)

    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0]        # (T,)
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]
    # spectral flux: positive-rectified frame-to-frame magnitude change
    flux = np.sqrt((np.diff(S, axis=1, prepend=S[:, :1]).clip(min=0) ** 2).sum(0))

    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=K.N_MELS)
    mel_db = librosa.power_to_db(mel + _EPS)                                      # (n_mels, T)
    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=K.N_MFCC)                 # (n_mfcc, T)
    mfcc_d = librosa.feature.delta(mfcc) if mfcc.shape[1] > 1 else np.zeros_like(mfcc)
    mfcc_dd = (librosa.feature.delta(mfcc, order=2) if mfcc.shape[1] > 2
               else np.zeros_like(mfcc))

    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=hop)     # (T,)
    try:
        tempo = float(np.atleast_1d(
            librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop))[0])
    except Exception:
        tempo = 0.0

    # ---- pooled: mean+std of each scalar series, mel/MFCC summaries -------------
    def ms(a):
        a = np.asarray(a, dtype=np.float64)
        return [float(a.mean()), float(a.std())]

    pooled: list[float] = []
    names: list[str] = []

    def add(label, vals):
        m, s = ms(vals)
        pooled.extend([m, s])
        names.extend([f"{label}_mean", f"{label}_std"])

    add("rms", rms)
    add("centroid", cent)
    add("bandwidth", bw)
    add("rolloff", roll)
    add("flatness", flat)
    add("flux", flux)
    add("zcr", zcr)
    add("onset", onset_env)
    pooled.append(tempo); names.append("tempo_bpm")

    # mel summary: per-band mean and std (2*n_mels)
    pooled.extend(mel_db.mean(1).tolist()); names += [f"mel{j}_mean" for j in range(K.N_MELS)]
    pooled.extend(mel_db.std(1).tolist()); names += [f"mel{j}_std" for j in range(K.N_MELS)]
    # MFCC + deltas: mean and std each (6*n_mfcc)
    for lbl, arr in (("mfcc", mfcc), ("mfccd", mfcc_d), ("mfccdd", mfcc_dd)):
        pooled.extend(arr.mean(1).tolist()); names += [f"{lbl}{j}_mean" for j in range(arr.shape[0])]
        pooled.extend(arr.std(1).tolist()); names += [f"{lbl}{j}_std" for j in range(arr.shape[0])]

    pooled_v = np.asarray(pooled, dtype=np.float32)

    # ---- time-resolved: a compact per-0.25 s vector ----------------------------
    scalars = np.stack([rms, cent, bw, roll, flat, flux, zcr, onset_env], axis=1)  # (T, 8)
    ts_scalars = _resample_to_grid(frame_t, scalars)                               # (12, 8)
    ts_mfcc = _resample_to_grid(frame_t, mfcc.T)                                   # (12, n_mfcc)
    timeseries = np.concatenate([ts_scalars, ts_mfcc], axis=1).astype(np.float32)  # (12, 8+n_mfcc)

    return {"vector": pooled_v, "names": names}, timeseries, np.arange(K.TIMERES_N) * K.TIMERES_STEP_S


@register_embedder
class HandcraftedAudio(BaseEmbedder):
    name = "hc_audio"
    modality = "handcrafted"
    description = ("Handcrafted audio descriptors: RMS, mel-spectrogram summary, "
                   "MFCC mean/std + delta + delta-delta, spectral centroid/bandwidth/"
                   "rolloff/flatness/flux, ZCR, onset strength, tempo. Pooled + 0.25 s "
                   "time-resolved. CPU librosa; no device check applies.")
    source = f"librosa (16 kHz mono, n_fft={K.N_FFT}, hop={K.HOP_LENGTH}, n_mels={K.N_MELS})"
    revision = "handcrafted-v1"
    agent = "V-B"
    frame_variant = "n/a (audio)"

    feature_names: list[str] | None = None

    def _embed(self, video_path: str) -> dict:
        a = extract_audio(video_path, sr=AUDIO_SR)
        pooled, ts, t = _audio_features(a["waveform"], AUDIO_SR)
        if self.feature_names is None:
            type(self).feature_names = pooled["names"]
        return {"pooled": pooled["vector"], "timeseries": ts, "times": t}


# ============================================================================== visual
def _gabor_bank():
    import cv2

    kernels = []
    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        for sigma in (2.0, 4.0):
            k = cv2.getGaborKernel((15, 15), sigma, theta, 8.0, 0.5, 0,
                                   ktype=cv2.CV_32F)
            kernels.append(k)
    return kernels


_GABOR = None


def _visual_features(frames: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """frames: (T, H, W, 3) uint8 RGB on the timeresolved grid (12 frames).

    Returns (pooled_vec, names, timeseries (12, Dt), times).
    """
    import cv2

    global _GABOR
    if _GABOR is None:
        _GABOR = _gabor_bank()

    T = frames.shape[0]
    grid = np.arange(T) * K.TIMERES_STEP_S

    lum = np.zeros(T)
    contrast = np.zeros(T)
    edge_density = np.zeros(T)
    gabor_energy = np.zeros(T)
    sat_mean = np.zeros(T)
    hue_hist = np.zeros((T, 12))     # 12-bin hue histogram
    val_mean = np.zeros(T)
    flow_mag = np.zeros(T)
    flow_dir_hist = np.zeros((T, 8))  # 8-bin direction histogram
    frame_diff = np.zeros(T)

    prev_gray = None
    for i in range(T):
        rgb = frames[i]
        bgr = rgb[:, :, ::-1]
        gray = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)

        lum[i] = gray.mean() / 255.0
        contrast[i] = gray.std() / 255.0

        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        edge_density[i] = (edges > 0).mean()

        g = np.mean([np.abs(cv2.filter2D(gray, cv2.CV_32F, k)).mean() for k in _GABOR])
        gabor_energy[i] = g / 255.0

        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        sat_mean[i] = s.mean()
        val_mean[i] = v.mean()
        hh, _ = np.histogram(h, bins=12, range=(0, 180), density=False)
        hue_hist[i] = hh / (hh.sum() + _EPS)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray.astype(np.uint8), gray.astype(np.uint8),
                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag[i] = float(mag.mean())
            dh, _ = np.histogram(ang, bins=8, range=(0, 2 * np.pi),
                                 weights=mag, density=False)
            flow_dir_hist[i] = dh / (dh.sum() + _EPS)
            frame_diff[i] = float(np.abs(gray - prev_gray).mean()) / 255.0
        prev_gray = gray

    # scene cuts: count frame-diff spikes above mean + 2*std
    fd = frame_diff[1:]
    thr = fd.mean() + 2 * fd.std() if fd.size else 0.0
    scene_cuts = float((fd > thr).sum()) if fd.size else 0.0

    def ms(a):
        a = np.asarray(a, dtype=np.float64)
        return [float(a.mean()), float(a.std())]

    pooled: list[float] = []
    names: list[str] = []

    def add(label, vals):
        m, s = ms(vals)
        pooled.extend([m, s]); names.extend([f"{label}_mean", f"{label}_std"])

    add("luminance", lum)
    add("contrast", contrast)
    add("edge_density", edge_density)
    add("gabor_energy", gabor_energy)
    add("saturation", sat_mean)
    add("value", val_mean)
    add("flow_mag", flow_mag)
    add("frame_diff", frame_diff)
    pooled.append(scene_cuts); names.append("scene_cuts")

    pooled.extend(hue_hist.mean(0).tolist()); names += [f"hue{j}_mean" for j in range(12)]
    pooled.extend(flow_dir_hist.mean(0).tolist()); names += [f"flowdir{j}_mean" for j in range(8)]

    pooled_v = np.asarray(pooled, dtype=np.float32)

    # time-resolved: per-frame compact vector
    ts = np.concatenate([
        lum[:, None], contrast[:, None], edge_density[:, None], gabor_energy[:, None],
        sat_mean[:, None], val_mean[:, None], flow_mag[:, None], frame_diff[:, None],
        hue_hist, flow_dir_hist,
    ], axis=1).astype(np.float32)
    return pooled_v, names, ts, grid


@register_embedder
class HandcraftedVisual(BaseEmbedder):
    name = "hc_visual"
    modality = "handcrafted"
    description = ("Handcrafted visual descriptors: per-frame luminance, contrast, HSV "
                   "colour (12-bin hue histogram, saturation, value), Gabor energy, Canny "
                   "edge density, Farneback optic-flow magnitude + 8-bin direction "
                   "histogram, frame-difference motion energy, scene-cut count. Pooled + "
                   "0.25 s time-resolved on the 12-frame grid. CPU cv2; no device check.")
    source = "opencv-headless 4.10 (frames via src.common.frames variant=timeresolved)"
    revision = "handcrafted-v1"
    agent = "V-B"
    frame_variant = "timeresolved"

    feature_names: list[str] | None = None

    def _embed(self, video_path: str) -> dict:
        fr = sample_frames(video_path, variant="timeresolved")["frames"]
        pooled_v, names, ts, t = _visual_features(fr)
        if self.feature_names is None:
            type(self).feature_names = names
        return {"pooled": pooled_v, "timeseries": ts, "times": t}


@register_embedder
class HandcraftedAV(BaseEmbedder):
    name = "hc_av"
    modality = "handcrafted"
    description = ("Concatenation of hc_audio and hc_visual pooled vectors "
                   "(audio first, then visual). Raw concatenation; z-scoring for the "
                   "joint model is applied downstream on train statistics only.")
    source = "hc_audio ++ hc_visual"
    revision = "handcrafted-v1"
    agent = "V-B"
    frame_variant = "timeresolved"

    def __init__(self, device: str | None = None, **params) -> None:
        super().__init__(device=device, **params)
        self._a = HandcraftedAudio()
        self._v = HandcraftedVisual()

    def _embed(self, video_path: str) -> dict:
        ra = self._a.embed(video_path)
        rv = self._v.embed(video_path)
        pooled = np.concatenate([ra["pooled"], rv["pooled"]]).astype(np.float32)
        return {"pooled": pooled}
