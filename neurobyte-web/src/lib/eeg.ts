import { parseNpy, NpyError } from "./npy";

/** Recordings are (trials, 17 channels, 1500 samples) at 500 Hz = 3 s each. */
export const N_CHANNELS = 17;
export const N_SAMPLES = 1500;
export const FS = 500;
export const EPOCH_S = N_SAMPLES / FS;
/** The latency window the winning pipeline restricts EEG to (50–400 ms after onset). */
export const LATENCY_WINDOW_S: [number, number] = [0.05, 0.4];

export interface Recording {
  fileName: string;
  sha256: string;
  nTrials: number;
  /** row-major (trial, channel, sample) */
  data: Float64Array;
  /** per-channel std across the whole file, for display scaling */
  channelStd: Float64Array;
}

export async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function loadRecording(buf: ArrayBuffer, fileName: string): Promise<Recording> {
  const arr = parseNpy(buf);
  const [nTrials, ch, ns] = arr.shape;
  if (arr.shape.length !== 3 || ch !== N_CHANNELS || ns !== N_SAMPLES)
    throw new NpyError(
      `expected shape (trials, ${N_CHANNELS}, ${N_SAMPLES}), got (${arr.shape.join(", ")})`,
    );
  if (arr.dtype !== "<f8" && arr.dtype !== "<f4")
    throw new NpyError(`EEG must be float32/float64, got ${arr.dtype}`);
  if (nTrials < 1) throw new NpyError("file contains no trials");
  for (let i = 0; i < arr.data.length; i += 997)
    if (!Number.isFinite(arr.data[i])) throw new NpyError("file contains NaN or infinite values");
  return {
    fileName,
    sha256: await sha256Hex(buf),
    nTrials,
    data: arr.data,
    channelStd: channelStd(arr.data, nTrials),
  };
}

function channelStd(data: Float64Array, nTrials: number): Float64Array {
  const out = new Float64Array(N_CHANNELS);
  for (let c = 0; c < N_CHANNELS; c++) {
    let s = 0, s2 = 0, n = 0;
    for (let t = 0; t < nTrials; t++) {
      const o = (t * N_CHANNELS + c) * N_SAMPLES;
      for (let k = 0; k < N_SAMPLES; k += 4) {
        const v = data[o + k];
        s += v; s2 += v * v; n++;
      }
    }
    const m = s / n;
    out[c] = Math.sqrt(Math.max(s2 / n - m * m, 1e-12));
  }
  return out;
}

export function trialChannel(rec: Recording, trial: number, ch: number): Float64Array {
  const o = (trial * N_CHANNELS + ch) * N_SAMPLES;
  return rec.data.subarray(o, o + N_SAMPLES);
}

/**
 * Per-channel activity at time tSec: RMS over a ±50 ms window, divided by that
 * channel's file-wide std, squashed to 0..1. Purely a display quantity.
 */
export function channelActivity(rec: Recording, trial: number, tSec: number): number[] {
  const centre = Math.round(Math.min(Math.max(tSec, 0), EPOCH_S) * FS);
  const half = 25;
  const a = Math.max(0, centre - half);
  const b = Math.min(N_SAMPLES, centre + half);
  const out: number[] = [];
  for (let c = 0; c < N_CHANNELS; c++) {
    const x = trialChannel(rec, trial, c);
    let s2 = 0;
    for (let k = a; k < b; k++) s2 += x[k] * x[k];
    const rms = Math.sqrt(s2 / Math.max(1, b - a)) / rec.channelStd[c];
    out.push(1 - Math.exp(-rms * 0.9));
  }
  return out;
}
