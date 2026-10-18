/**
 * Ranking post-processing, a line-for-line port of submission2/rank.py:
 *
 *   Z = (S - mean_row) / max(std_row, 1e-12)
 *   K = exp(clip(Z / tau, -700, 700))
 *   repeat iters: normalise rows, then columns        (Sinkhorn)
 *   ranked = TEST_IDS[argsort(-K, axis=1)]
 *
 * Sinkhorn exploits that each subject's 102 recordings and 102 videos pair
 * one-to-one. rank.test.ts checks this reproduces the shipped lists exactly.
 */
export const TEST_IDS: number[] = Array.from({ length: 102 }, (_, i) => 1001 + i);

/** S is row-major (n × m). Returns a new row-major Float64Array. */
export function sinkhorn(S: Float64Array, n: number, m: number, iters: number, tau: number): Float64Array {
  const K = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    let mean = 0;
    for (let j = 0; j < m; j++) mean += S[i * m + j];
    mean /= m;
    let v = 0;
    for (let j = 0; j < m; j++) v += (S[i * m + j] - mean) ** 2;
    const sd = Math.max(Math.sqrt(v / m), 1e-12); // numpy std: ddof = 0
    for (let j = 0; j < m; j++) {
      const z = (S[i * m + j] - mean) / sd / tau;
      K[i * m + j] = Math.exp(Math.min(700, Math.max(-700, z)));
    }
  }
  for (let t = 0; t < iters; t++) {
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < m; j++) s += K[i * m + j];
      s = Math.max(s, 1e-300);
      for (let j = 0; j < m; j++) K[i * m + j] /= s;
    }
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += K[i * m + j];
      s = Math.max(s, 1e-300);
      for (let i = 0; i < n; i++) K[i * m + j] /= s;
    }
  }
  return K;
}

/** Candidate column indices of one row, highest score first. */
export function argsortDesc(row: ArrayLike<number>): number[] {
  const idx = Array.from({ length: row.length }, (_, i) => i);
  return idx.sort((a, b) => row[b] - row[a] || a - b);
}

export interface Ranking {
  n: number;
  /** ranked[i] = 102 stimulus IDs for recording i, most likely first. */
  ranked: number[][];
  /** Post-Sinkhorn match probabilities, row-major n × 102. */
  probs: Float64Array;
}

/**
 * CSLS (cross-domain similarity local scaling), port of the team's
 * src/mindread/methods.py: 2S - r_query - r_candidate, where r is the mean of the
 * k largest scores in that row / column. Removes each candidate's average pull ("hubs").
 */
export function csls(S: Float64Array, n: number, m: number, k: number): Float64Array {
  k = Math.min(k, n - 1, m - 1);
  const topMean = (vals: number[]) => {
    vals.sort((a, b) => a - b);
    let s = 0;
    for (let i = vals.length - k; i < vals.length; i++) s += vals[i];
    return s / k;
  };
  const rq = new Float64Array(n);
  const rc = new Float64Array(m);
  for (let i = 0; i < n; i++) rq[i] = topMean(Array.from(S.subarray(i * m, (i + 1) * m)));
  for (let j = 0; j < m; j++) {
    const col: number[] = [];
    for (let i = 0; i < n; i++) col.push(S[i * m + j]);
    rc[j] = topMean(col);
  }
  const out = new Float64Array(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[i * m + j] = 2 * S[i * m + j] - rq[i] - rc[j];
  return out;
}

export interface PostProcess {
  iters: number;
  tau: number;
  /** CSLS neighbourhood size; null/undefined = no CSLS (winfuse alone) */
  cslsK?: number | null;
  /** false = stop after CSLS (the team's "no Sinkhorn" fallback) */
  useSinkhorn?: boolean;
}

/**
 * Post-processing exactly as the team's combo/c1x.py: optional CSLS, then Sinkhorn,
 * then a stable argsort (ties keep candidate order). Without Sinkhorn, rows are
 * ranked on the CSLS scores and softmax-normalised only for display.
 */
export function rankScores(scores: Float64Array, n: number, opts: PostProcess): Ranking {
  const m = TEST_IDS.length;
  const S = opts.cslsK ? csls(scores, n, m, opts.cslsK) : scores;
  const sk = opts.useSinkhorn !== false;
  const order = sk ? sinkhorn(S, n, m, opts.iters, opts.tau) : S;
  const probs = sk ? order : normaliseRows(S, n, m);
  const ranked: number[][] = [];
  for (let i = 0; i < n; i++) {
    ranked.push(argsortDesc(order.subarray(i * m, (i + 1) * m)).map((j) => TEST_IDS[j]));
  }
  return { n, ranked, probs };
}

function normaliseRows(S: Float64Array, n: number, m: number): Float64Array {
  // Softmax per row, only used to display "before bijection" scores.
  const out = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    let mx = -Infinity;
    for (let j = 0; j < m; j++) mx = Math.max(mx, S[i * m + j]);
    let mean = 0;
    for (let j = 0; j < m; j++) mean += S[i * m + j];
    mean /= m;
    let v = 0;
    for (let j = 0; j < m; j++) v += (S[i * m + j] - mean) ** 2;
    const sd = Math.max(Math.sqrt(v / m), 1e-12);
    let s = 0;
    for (let j = 0; j < m; j++) s += out[i * m + j] = Math.exp((S[i * m + j] - mx) / sd / 0.5);
    for (let j = 0; j < m; j++) out[i * m + j] /= s;
  }
  return out;
}

export interface Accuracy {
  top1: number;
  top5: number;
  medianRank: number;
  meanRank: number;
  /** 1-based rank of the true video per recording. */
  ranks: number[];
}

export function accuracy(ranked: number[][], truth: ArrayLike<number>): Accuracy {
  const ranks = ranked.map((row, i) => row.indexOf(truth[i]) + 1);
  if (ranks.some((r) => r <= 0)) throw new Error("a true label is missing from its ranked row");
  const sorted = [...ranks].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return {
    top1: ranks.filter((r) => r === 1).length / ranks.length,
    top5: ranks.filter((r) => r <= 5).length / ranks.length,
    medianRank: sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2,
    meanRank: ranks.reduce((a, b) => a + b, 0) / ranks.length,
    ranks,
  };
}
