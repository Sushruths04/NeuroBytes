import { describe, expect, it } from "vitest";
import { BANDS, fft, periodogram } from "./spectrum";

describe("spectrum", () => {
  it("fft matches a direct DFT", () => {
    const n = 16, x = Array.from({ length: n }, (_, i) => Math.sin(i) + 0.3 * Math.cos(3 * i));
    const re = Float64Array.from(x), im = new Float64Array(n);
    fft(re, im);
    for (let k = 0; k < n; k++) {
      let r = 0, m = 0;
      for (let t = 0; t < n; t++) {
        r += x[t] * Math.cos((2 * Math.PI * k * t) / n);
        m -= x[t] * Math.sin((2 * Math.PI * k * t) / n);
      }
      expect(re[k]).toBeCloseTo(r, 9);
      expect(im[k]).toBeCloseTo(m, 9);
    }
  });

  it("a 10 Hz sine peaks at 10 Hz and lands in the alpha band", () => {
    const x = Array.from({ length: 1024 }, (_, i) => Math.sin((2 * Math.PI * 10 * i) / 500));
    const p = periodogram(x);
    const peak = p.reduce((a, b) => (b[1] > a[1] ? b : a));
    expect(peak[0]).toBeCloseTo(10, 0);
    const alpha = BANDS.find((b) => b.name === "Alpha")!;
    expect(peak[0] >= alpha.lo && peak[0] < alpha.hi).toBe(true);
  });
});
