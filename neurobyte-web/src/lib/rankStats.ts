/**
 * Pooled dev-test rank statistics for the analysis charts ("Where the true video
 * lands", "Per person"). Only sub-01..04 have public labels; sub-05/06 are never
 * read here.
 */
import type { ModelId } from "../content/models";
import { fetchModelNpy, fetchNpy } from "./api";
import { accuracy } from "./rank";

export const DEV_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04"] as const;

export const RANK_BUCKETS = [
  { label: "1", lo: 1, hi: 1 },
  { label: "2", lo: 2, hi: 2 },
  { label: "3–5", lo: 3, hi: 5 },
  { label: "6–10", lo: 6, hi: 10 },
  { label: "11–20", lo: 11, hi: 20 },
  { label: "21+", lo: 21, hi: Infinity },
] as const;

export interface Bucket {
  label: string;
  pct: number;
}

/** Share (0..100) of recordings whose true video falls in each rank bucket. */
export function rankHistogram(ranks: number[]): Bucket[] {
  const n = ranks.length || 1;
  return RANK_BUCKETS.map((b) => ({
    label: b.label,
    pct: (ranks.filter((r) => r >= b.lo && r <= b.hi).length / n) * 100,
  }));
}

export interface PersonStats {
  sub: string;
  top1: number;
  top5: number;
  meanRank: number;
  n: number;
}

export function personStats(sub: string, ranks: number[]): PersonStats {
  const n = ranks.length;
  return {
    sub,
    top1: n ? (ranks.filter((r) => r === 1).length / n) * 100 : 0,
    top5: n ? (ranks.filter((r) => r <= 5).length / n) * 100 : 0,
    meanRank: n ? ranks.reduce((a, b) => a + b, 0) / n : 0,
    n,
  };
}

/** 1-based true ranks per recording from an int64 n×102 ranked matrix and the labels. */
export function ranksFrom(ranked: ArrayLike<number>, labels: ArrayLike<number>, m = 102): number[] {
  const rows: number[][] = [];
  for (let i = 0; i < labels.length; i++) rows.push(Array.from({ length: m }, (_, j) => ranked[i * m + j]));
  return accuracy(rows, labels).ranks;
}

const cache = new Map<ModelId, Promise<Record<string, number[]>>>();

/**
 * True ranks of the team's ranking for sub-01..04 and one model, fetched once per
 * model (4 label reads per model, all logged by the server).
 */
export function loadDevRanks(model: ModelId): Promise<Record<string, number[]>> {
  let p = cache.get(model);
  if (!p) {
    p = Promise.all(
      DEV_SUBJECTS.map(async (sub) => {
        const [ranked, labels] = await Promise.all([fetchModelNpy(model, sub, "ranked"), fetchNpy("labels", sub)]);
        return [sub, ranksFrom(ranked.data, labels.data, ranked.shape[1] ?? 102)] as const;
      }),
    ).then((pairs) => Object.fromEntries(pairs));
    p.catch(() => cache.delete(model)); // allow a retry after a failure
    cache.set(model, p);
  }
  return p;
}
