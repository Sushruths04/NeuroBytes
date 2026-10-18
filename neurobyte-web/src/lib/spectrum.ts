/**
 * Power spectrum of one 3-second trial, for display (Analytics page).
 * Hann-windowed periodogram via a radix-2 FFT on the first 1024 samples
 * (2.05 s at 500 Hz, 0.49 Hz resolution), averaged over the chosen channels.
 * The recording is used as stored.
 */
import { FS, N_CHANNELS, N_SAMPLES, trialChannel, type Recording } from "./eeg";

export const NFFT = 1024;

export interface Band {
  name: string;
  lo: number;
  hi: number;
}

export const BANDS: Band[] = [
  { name: "Delta", lo: 1, hi: 4 },
  { name: "Theta", lo: 4, hi: 8 },
  { name: "Alpha", lo: 8, hi: 13 },
  { name: "Beta", lo: 13, hi: 30 },
  { name: "Gamma", lo: 30, hi: 45 },
];

/** In-place iterative radix-2 FFT. re/im length must be a power of two. */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const ai = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k + len / 2] = re[i + k] - ar;
        im[i + k + len / 2] = im[i + k] - ai;
        re[i + k] += ar;
        im[i + k] += ai;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

/** Periodogram of one signal: [frequency Hz, power] for bins 1..NFFT/2. */
export function periodogram(x: ArrayLike<number>, fs = FS, n = NFFT): [number, number][] {
  const re = new Float64Array(n), im = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  for (let i = 0; i < n; i++) re[i] = (x[i] - mean) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));
  fft(re, im);
  const out: [number, number][] = [];
  for (let k = 1; k <= n / 2; k++) out.push([(k * fs) / n, (re[k] ** 2 + im[k] ** 2) / n]);
  return out;
}

export interface Spectrum {
  /** [Hz, power] up to maxHz, averaged over channels */
  psd: [number, number][];
  /** relative band power, sums to 1 over BANDS */
  bands: number[];
}

export function spectrum(rec: Recording, trial: number, channels: number[], maxHz = 45): Spectrum {
  const chs = channels.length ? channels : Array.from({ length: N_CHANNELS }, (_, i) => i);
  let acc: [number, number][] | null = null;
  for (const c of chs) {
    const p = periodogram(trialChannel(rec, trial, c).subarray(0, Math.min(NFFT, N_SAMPLES)));
    acc = acc ? acc.map(([f, v], i) => [f, v + p[i][1]] as [number, number]) : p;
  }
  const psd = acc!.filter(([f]) => f <= maxHz).map(([f, v]) => [f, v / chs.length] as [number, number]);
  const raw = BANDS.map((b) => psd.filter(([f]) => f >= b.lo && f < b.hi).reduce((s, [, v]) => s + v, 0));
  const tot = raw.reduce((a, b) => a + b, 0) || 1;
  return { psd, bands: raw.map((v) => v / tot) };
}
