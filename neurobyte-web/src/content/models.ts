/**
 * The models the app can rank with. C13 is the team's final pipeline and the default.
 * Formulas, weights and dev numbers come from the team's frozen configs
 * (configs/frozen_C10..C13.json) and submission2/weights/config.json for winfuse.
 * The score matrices themselves are exported by scripts/export_models.py and served
 * by the local data API; the browser runs the post-processing (CSLS + Sinkhorn).
 */
export type ModelId = "C13" | "C10" | "WF";

export interface Window {
  /** ms after video onset */
  from: number;
  to: number;
  label: string;
}

export interface ModelInfo {
  id: ModelId;
  name: string;
  short: string;
  /** what this model adds over the previous one */
  adds: string;
  formula: string;
  scorers: string;
  /** CSLS neighbourhood; null = no CSLS */
  cslsK: number | null;
  /** the one band of the recording the model reads (shaded on the EEG plot) */
  windows: Window[];
  /** frozen dev-test numbers (sub-01..04, 408 recordings), as published by the team */
  frozen: { top1: number; top5: number; meanRank: number };
}

/**
 * One band per model: the span of the recording its scorers read, from the team code.
 * winfuse: 398 of its 509 scorers use 15 latency windows, 50–200 … 400–700 ms
 * (submission2/weights/channels.txt). mindread's 77 scorers reuse those 15 windows
 * (combo/combo.py), T2 reads 50–400 ms (configs/development_tempo_t2feat.json), and the
 * networks (C11–C13) read 0–600 ms (150 samples at 250 Hz, configs/frozen_C13.json).
 * C13 is the final submission (neurobytes repo, configs/frozen_C13.json, 18 Oct 2026):
 * networks h6 and h10, 36.03 / 67.16 / 5.99.
 */
const WINDOW_50_700: Window = { from: 50, to: 700, label: "50–700 ms" };
const WINDOW_0_700: Window = { from: 0, to: 700, label: "0–700 ms" };

export const MODELS: ModelInfo[] = [
  {
    id: "C13",
    name: "C13",
    short: "C13",
    adds: "+ two contrastive networks (h6, h10)",
    formula: "0.85 × C10 blend + 0.075 × network h6 + 0.075 × network h10 → CSLS k=2 → Sinkhorn",
    scorers: "509 time-window decoders + 77 feature decoders + T2 encoder + two contrastive networks (h6, h10)",
    cslsK: 2,
    windows: [WINDOW_0_700],
    frozen: { top1: 36.03, top5: 67.16, meanRank: 5.99 },
  },
  {
    id: "C10",
    name: "C10",
    short: "C10",
    adds: "both pipelines + alignment + Sinkhorn",
    formula: "0.85 × (509/586 winfuse + 77/586 mindread) + 0.15 × T2 → CSLS k=2 → Sinkhorn",
    scorers: "509 winfuse ridges + 77 mindread scorers + T2 encoder",
    cslsK: 2,
    windows: [WINDOW_50_700],
    frozen: { top1: 32.6, top5: 66.67, meanRank: 6.59 },
  },
  {
    id: "WF",
    name: "WF",
    short: "WF",
    adds: "winfuse pipeline on its own",
    formula: "509 ridge scorers, rank fusion + hubness, equal weights → Sinkhorn",
    scorers: "509 ridge scorers (EEG windows × video embeddings)",
    cslsK: null,
    windows: [WINDOW_50_700],
    frozen: { top1: 30.64, top5: 60.54, meanRank: 7.59 },
  },
];

export const DEFAULT_MODEL: ModelId = "C13";
export const MODEL_IDS = MODELS.map((m) => m.id);
export const modelInfo = (id: ModelId) => MODELS.find((m) => m.id === id)!;
export const isModelId = (s: unknown): s is ModelId => typeof s === "string" && (MODEL_IDS as string[]).includes(s);
