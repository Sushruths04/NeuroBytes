import { describe, expect, it } from "vitest";
import fs from "node:fs";
import { parseNpy } from "./npy";
import { accuracy, csls, rankScores, TEST_IDS } from "./rank";
import { MODELS } from "../content/models";
import { asArrayBuffer, HAS_LABELS, HAS_MODELS, HAS_SUBMISSION, modelFile, subFile, dataFile } from "../../test/fixtures";

const load = (p: string) => parseNpy(asArrayBuffer(fs.readFileSync(p)));
const SUBS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"];

describe.skipIf(!HAS_SUBMISSION)("browser ranking reproduces submission2 exactly", () => {
  const cfg = HAS_SUBMISSION ? JSON.parse(fs.readFileSync(subFile("weights/config.json"), "utf8")) : null;
  const { iters, tau } = cfg?.postprocess_params ?? { iters: 3, tau: 0.5 };

  for (const sub of SUBS) {
    it(`${sub}: Sinkhorn + argsort == shipped ranked list`, () => {
      const s = load(subFile(`scores/${sub}_fused_scores.npy`));
      const shipped = load(subFile(`ranked/${sub}_ranked_candidate_list.npy`));
      expect(s.shape).toEqual([102, 102]);
      const r = rankScores(s.data, 102, { iters, tau });
      expect(r.ranked.flat()).toEqual(Array.from(shipped.data));
      for (const row of r.ranked) expect([...row].sort((a, b) => a - b)).toEqual(TEST_IDS);
    });
  }

  it.skipIf(!HAS_LABELS)("pooled accuracy on sub-01..04 equals the README headline (30.64% / 60.54%)", () => {
    let top1 = 0, top5 = 0, n = 0;
    for (const sub of SUBS.slice(0, 4)) {
      const s = load(subFile(`scores/${sub}_fused_scores.npy`));
      const truth = load(dataFile(`tst_stim/${sub}_tst_stim.npy`)).data;
      const a = accuracy(rankScores(s.data, 102, { iters, tau }).ranked, truth);
      top1 += a.top1 * 102; top5 += a.top5 * 102; n += 102;
    }
    expect((top1 / n) * 100).toBeCloseTo(30.64, 2);
    expect((top5 / n) * 100).toBeCloseTo(60.54, 2);
  });
});

describe.skipIf(!HAS_MODELS)("every model: browser post-processing reproduces the team's Python ranking", () => {
  for (const m of MODELS) {
    for (const sub of SUBS) {
      it(`${m.id} ${sub}: ${m.cslsK ? "CSLS k=" + m.cslsK + " + " : ""}Sinkhorn + argsort == team code`, () => {
        const pre = load(modelFile(`${m.id}/${sub}_pre.npy`));
        const want = load(modelFile(`${m.id}/${sub}_ranked.npy`));
        const r = rankScores(pre.data, 102, { iters: 3, tau: 0.5, cslsK: m.cslsK });
        expect(r.ranked.flat()).toEqual(Array.from(want.data));
      });
    }
    it.skipIf(!HAS_LABELS)(`${m.id}: pooled dev accuracy equals the frozen config (${m.frozen.top1}% / ${m.frozen.top5}%)`, () => {
      const ranks: number[] = [];
      for (const sub of SUBS.slice(0, 4)) {
        const pre = load(modelFile(`${m.id}/${sub}_pre.npy`));
        const truth = load(dataFile(`tst_stim/${sub}_tst_stim.npy`)).data;
        ranks.push(...accuracy(rankScores(pre.data, 102, { iters: 3, tau: 0.5, cslsK: m.cslsK }).ranked, truth).ranks);
      }
      const top1 = (ranks.filter((r) => r === 1).length / ranks.length) * 100;
      const top5 = (ranks.filter((r) => r <= 5).length / ranks.length) * 100;
      const mean = ranks.reduce((a, b) => a + b, 0) / ranks.length;
      expect(top1).toBeCloseTo(m.frozen.top1, 1);
      expect(top5).toBeCloseTo(m.frozen.top5, 1);
      expect(mean).toBeCloseTo(m.frozen.meanRank, 1);
    });
  }
});

describe("csls (synthetic, always runs)", () => {
  it("matches the team's definition: 2S - mean top-k of row - mean top-k of column", () => {
    const S = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const out = csls(S, 3, 3, 2);
    // row top-2 means: 2.5, 5.5, 8.5; column top-2 means: 5.5, 6.5, 7.5
    expect(Array.from(out)).toEqual([2 - 2.5 - 5.5, 4 - 2.5 - 6.5, 6 - 2.5 - 7.5, 8 - 5.5 - 5.5, 10 - 5.5 - 6.5, 12 - 5.5 - 7.5, 14 - 8.5 - 5.5, 16 - 8.5 - 6.5, 18 - 8.5 - 7.5]);
  });
});

describe("sinkhorn ranking (synthetic, always runs)", () => {
  it("recovers a planted one-to-one matching and returns permutations", () => {
    const n = 102, m = 102;
    const S = new Float64Array(n * m);
    let seed = 1234;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
    const perm = TEST_IDS.map((_, i) => (i * 37) % 102); // a fixed bijection
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) S[i * m + j] = rnd() + (perm[i] === j ? 3 : 0);
    const r = rankScores(S, n, { iters: 3, tau: 0.5 });
    for (let i = 0; i < n; i++) {
      expect(r.ranked[i][0]).toBe(TEST_IDS[perm[i]]);
      expect([...r.ranked[i]].sort((a, b) => a - b)).toEqual(TEST_IDS);
    }
    for (let j = 0; j < m; j++) {
      let col = 0;
      for (let i = 0; i < n; i++) col += r.probs[i * m + j];
      expect(col).toBeCloseTo(1, 9); // last Sinkhorn step normalises columns
    }
  });
});

describe("accuracy", () => {
  it("computes ranks, median and mean", () => {
    const ranked = [[1, 2, 3], [3, 2, 1], [2, 3, 1]];
    const a = accuracy(ranked, [1, 1, 1]);
    expect(a.ranks).toEqual([1, 3, 3]);
    expect(a.top1).toBeCloseTo(1 / 3);
    expect(a.medianRank).toBe(3);
    expect(a.meanRank).toBeCloseTo(7 / 3);
  });
  it("throws when a label is missing", () => {
    expect(() => accuracy([[1, 2]], [9])).toThrow();
  });
});
