/**
 * Pure helpers for the results ("Inference") layout: which candidate is shown in
 * the player, the Top 5 / Top 10 rows and the verdict badge.
 */

export interface Candidate {
  id: number;
  /** 1-based rank in this view */
  rank: number;
  score: number;
  isTruth: boolean;
}

export interface InferenceSummary {
  selectedId: number;
  selectedRank: number;
  selectedScore: number;
  top5: Candidate[];
  top10: Candidate[];
  /** rank of the watched video, null when labels are unknown */
  truthRank: number | null;
}

/**
 * @param row      ranked candidate ids for one recording, best first
 * @param prob     score of a candidate id in the current view
 * @param truth    watched video id (sub-01..04 only), undefined otherwise
 * @param selected 0-based index into row of the candidate shown in the player
 */
export function inferenceSummary(
  row: readonly number[],
  prob: (id: number) => number,
  truth: number | undefined,
  selected: number,
): InferenceSummary {
  const sel = Math.min(Math.max(0, Math.trunc(selected) || 0), Math.max(0, row.length - 1));
  const cand = (id: number, k: number): Candidate => ({ id, rank: k + 1, score: prob(id), isTruth: truth !== undefined && id === truth });
  const idx = truth !== undefined ? row.indexOf(truth) : -1;
  return {
    selectedId: row[sel],
    selectedRank: sel + 1,
    selectedScore: prob(row[sel]),
    top5: row.slice(0, 5).map(cand),
    top10: row.slice(0, 10).map(cand),
    truthRank: idx >= 0 ? idx + 1 : null,
  };
}

export type VerdictKind = "correct" | "near" | "miss" | "private";

export function verdictOf(truthRank: number | null, unlabelled: boolean): { kind: VerdictKind; text: string } {
  if (unlabelled || truthRank === null) return { kind: "private", text: "scored person · labels private" };
  if (truthRank === 1) return { kind: "correct", text: "Correct · rank 1" };
  return { kind: truthRank <= 5 ? "near" : "miss", text: `True video at rank ${truthRank}` };
}
