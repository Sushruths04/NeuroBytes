import { describe, expect, it } from "vitest";
import { inferenceSummary, verdictOf } from "./inference";

// Synthetic 102-id ranking: a fixed permutation of 1001..1102, best first.
const ids = Array.from({ length: 102 }, (_, i) => 1001 + i);
const row = ids.map((_, i) => ids[(i * 37 + 19) % 102]);
const scores = new Map(row.map((id, k) => [id, 1 - k / 102]));
const prob = (id: number) => scores.get(id)!;

describe("inferenceSummary", () => {
  it("is a permutation fixture", () => {
    expect(new Set(row).size).toBe(102);
  });

  it("selects rank 1 by default, marks the truth at rank 3", () => {
    const truth = row[2];
    const s = inferenceSummary(row, prob, truth, 0);
    expect(s.selectedId).toBe(row[0]);
    expect(s.selectedRank).toBe(1);
    expect(s.selectedScore).toBe(1);
    expect(s.truthRank).toBe(3);
    expect(s.top5.map((c) => c.id)).toEqual(row.slice(0, 5));
    expect(s.top10.map((c) => c.rank)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(s.top10.filter((c) => c.isTruth).map((c) => c.rank)).toEqual([3]);
    expect(s.top5[1].score).toBeCloseTo(1 - 1 / 102);
  });

  it("follows the selected index and clamps it", () => {
    const s = inferenceSummary(row, prob, undefined, 3);
    expect(s.selectedId).toBe(row[3]);
    expect(s.selectedRank).toBe(4);
    expect(s.selectedScore).toBeCloseTo(1 - 3 / 102);
    expect(inferenceSummary(row, prob, undefined, 500).selectedRank).toBe(102);
    expect(inferenceSummary(row, prob, undefined, -2).selectedRank).toBe(1);
  });

  it("unlabelled: no truth rank and no truth marks", () => {
    const s = inferenceSummary(row, prob, undefined, 0);
    expect(s.truthRank).toBeNull();
    expect(s.top10.some((c) => c.isTruth)).toBe(false);
  });

  it("truth outside the ranking gives null", () => {
    expect(inferenceSummary(row, prob, 9999, 0).truthRank).toBeNull();
  });
});

describe("verdictOf", () => {
  it("covers correct / near / miss / private", () => {
    expect(verdictOf(1, false)).toEqual({ kind: "correct", text: "Correct · rank 1" });
    expect(verdictOf(3, false)).toEqual({ kind: "near", text: "True video at rank 3" });
    expect(verdictOf(40, false)).toEqual({ kind: "miss", text: "True video at rank 40" });
    expect(verdictOf(null, false).kind).toBe("private");
    // sub-05/06: never show correctness, even if a rank were somehow passed
    expect(verdictOf(1, true)).toEqual({ kind: "private", text: "scored person · labels private" });
  });
});
