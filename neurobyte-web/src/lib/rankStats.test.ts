import { describe, expect, it } from "vitest";
import fs from "node:fs";
import { parseNpy } from "./npy";
import { DEV_SUBJECTS, personStats, rankHistogram, ranksFrom } from "./rankStats";
import { asArrayBuffer, dataFile, HAS_LABELS, HAS_MODELS, modelFile } from "../../test/fixtures";

const load = (p: string) => parseNpy(asArrayBuffer(fs.readFileSync(p)));

describe("rankHistogram (synthetic, always runs)", () => {
  it("puts bucket edges in the right bucket and sums to 100", () => {
    const h = rankHistogram([1, 2, 3, 5, 6, 10, 11, 20, 21, 102]);
    expect(h.map((b) => b.label)).toEqual(["1", "2", "3–5", "6–10", "11–20", "21+"]);
    expect(h.map((b) => b.pct)).toEqual([10, 10, 20, 20, 20, 20]);
    expect(h.reduce((a, b) => a + b.pct, 0)).toBeCloseTo(100, 9);
  });
  it("returns zeros for no recordings", () => {
    expect(rankHistogram([]).every((b) => b.pct === 0)).toBe(true);
  });
});

describe("personStats / ranksFrom (synthetic, always runs)", () => {
  it("computes top-1, top-5 and mean rank as percentages", () => {
    const s = personStats("sub-01", [1, 1, 4, 6]);
    expect(s).toEqual({ sub: "sub-01", top1: 50, top5: 75, meanRank: 3, n: 4 });
  });
  it("reads 1-based ranks from a flat ranked matrix", () => {
    expect(ranksFrom([5, 6, 7, 7, 5, 6], [5, 6], 3)).toEqual([1, 3]);
  });
});

describe.skipIf(!HAS_MODELS || !HAS_LABELS)("C13 dev test reproduces the website analytics", () => {
  const per = Object.fromEntries(
    DEV_SUBJECTS.map((sub) => {
      if (!HAS_MODELS || !HAS_LABELS) return [sub, []];
      const ranked = load(modelFile(`C13/${sub}_ranked.npy`));
      const truth = load(dataFile(`tst_stim/${sub}_tst_stim.npy`));
      return [sub, ranksFrom(ranked.data, truth.data, ranked.shape[1])];
    }),
  ) as Record<string, number[]>;

  it("pooled rank histogram", () => {
    const h = rankHistogram(DEV_SUBJECTS.flatMap((s) => per[s]));
    const want = [36.03, 10.29, 20.83, 16.91, 10.05, 5.88];
    h.forEach((b, i) => expect(Math.abs(b.pct - want[i])).toBeLessThan(0.01));
  });

  it("per-person top-1 / top-5", () => {
    const want: Record<string, [number, number]> = {
      "sub-01": [49.02, 73.53],
      "sub-02": [32.35, 68.63],
      "sub-03": [26.47, 57.84],
      "sub-04": [36.27, 68.63],
    };
    for (const sub of DEV_SUBJECTS) {
      const s = personStats(sub, per[sub]);
      expect(s.n).toBe(102);
      expect(Math.abs(s.top1 - want[sub][0])).toBeLessThan(0.01);
      expect(Math.abs(s.top5 - want[sub][1])).toBeLessThan(0.01);
      console.log(`${sub} mean rank ${s.meanRank.toFixed(2)}`); // recorded, not asserted
    }
  });
});
