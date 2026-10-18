import { fetchManifest, fetchModelNpy, fetchNpy, type Manifest } from "./api";
import { modelInfo, type ModelId } from "../content/models";
import { type Recording } from "./eeg";
import { accuracy, rankScores, TEST_IDS, type Accuracy, type Ranking } from "./rank";

export type StageKind = "live" | "replayed";

export interface Stage {
  id: string;
  title: string;
  detail: string;
  kind: StageKind;
  /** seconds of the guided animation spent on this stage */
  seconds: number;
}

/**
 * The pipeline of the selected model, in order. Every model has the same eight
 * stages, so the animation timing is model-independent; only the text changes.
 * "live" stages run in the browser on the uploaded file. "replayed" stages ran
 * offline in Python (the fitted models are many GB); the browser loads their
 * output, the fused 102 × 102 score matrix before post-processing, and then
 * runs the post-processing itself.
 */
export function stagesFor(model: ModelId): Stage[] {
  const m = modelInfo(model);
  const team = model !== "WF";
  return [
    { id: "read", title: "Read & validate", detail: "Parse .npy, check (trials, 17, 1500), finite values", kind: "live", seconds: 1.2 },
    { id: "match", title: "Identify recording", detail: "SHA-256 fingerprint against the six scored test sets", kind: "live", seconds: 1.2 },
    { id: "preprocess", title: "Preprocess", detail: team ? "Band-pass, epoch 0–3 s, per-person standardisation" : "Notch → 0.5–40 Hz band-pass → 250 Hz → z-score (train stats)", kind: "replayed", seconds: 3 },
    { id: "window", title: "Latency window", detail: `Most scorers read ${m.windows[0].label} after the clip starts`, kind: "replayed", seconds: 3 },
    { id: "scorers", title: team ? "Brain-to-video scorers" : "509 ridge scorers", detail: m.scorers, kind: "replayed", seconds: 3 },
    { id: "fusion", title: "Fusion", detail: m.formula.split(" → ")[0], kind: "replayed", seconds: 3 },
    {
      id: "sinkhorn",
      title: m.cslsK ? "CSLS + Sinkhorn" : "Sinkhorn bijection",
      detail: (m.cslsK ? `CSLS k=${m.cslsK} removes hub pull; then ` : "") + "Sinkhorn: 102 recordings ↔ 102 videos one-to-one (3 iterations, τ = 0.5)",
      kind: "live",
      seconds: 3,
    },
    { id: "rank", title: "Rank candidates", detail: "argsort per recording → 102 × 102 stimulus IDs, most likely first", kind: "live", seconds: 1.5 },
  ];
}

/** Stage layout shared by all models (titles differ, timing does not). */
export const STAGES: Stage[] = stagesFor("C13");

export const TOTAL_SECONDS = STAGES.reduce((a, s) => a + s.seconds, 0);

/** Which stage is active `t` seconds into the animation (STAGES.length = finished). */
export function stageAt(t: number): { index: number; progress: number } {
  let acc = 0;
  for (let i = 0; i < STAGES.length; i++) {
    if (t < acc + STAGES[i].seconds) return { index: i, progress: (t - acc) / STAGES[i].seconds };
    acc += STAGES[i].seconds;
  }
  return { index: STAGES.length, progress: 1 };
}

export interface PipelineResult {
  model: ModelId;
  subject: string;
  hasLabels: boolean;
  /** sub-01..04 have public ground truth; it may still be switched off */
  isDevSubject: boolean;
  n: number;
  /** before the bijection: fused scores only */
  pre: Ranking;
  /** after Sinkhorn: the submitted ranking */
  post: Ranking;
  /** rows identical to the ranking computed by the model's own Python code */
  shipped: { equalRows: number; total: number } | null;
  labels: number[] | null;
  preAcc: Accuracy | null;
  postAcc: Accuracy | null;
  timingsMs: Record<string, number>;
}

export type MatchOutcome =
  | { ok: true; subject: string; manifest: Manifest }
  | { ok: false; reason: string; manifest: Manifest };

export async function matchRecording(rec: Recording, model: ModelId): Promise<MatchOutcome> {
  const manifest = await fetchManifest();
  const hit = manifest.subjects.find((s) => s.sha256 === rec.sha256);
  if (!hit)
    return {
      ok: false,
      manifest,
      reason:
        "This recording is not one of the six test sets the pipeline has scored. Ranking a new recording needs the subject's own ridges fitted on its training data, which runs in Python: python submission2/rank.py --fit --subjects <sub> --data-root <challenge root>.",
    };
  if (!(manifest.models[model] ?? []).includes(hit.id))
    return {
      ok: false,
      manifest,
      reason: `${modelInfo(model).name} has no score matrix for ${hit.id}. Run scripts/export_models.py (see app/README.md), or pick another model.`,
    };
  if (rec.nTrials !== TEST_IDS.length)
    return { ok: false, manifest, reason: `expected ${TEST_IDS.length} trials, file has ${rec.nTrials}.` };
  return { ok: true, subject: hit.id, manifest };
}

export async function runPipeline(subject: string, manifest: Manifest, model: ModelId): Promise<PipelineResult> {
  const info = manifest.subjects.find((s) => s.id === subject)!;
  const cfg = modelInfo(model);
  const timingsMs: Record<string, number> = {};
  const t = performance.now();
  const scores = await fetchModelNpy(model, subject, "pre");
  timingsMs.loadScores = performance.now() - t;
  const [n, m] = scores.shape;
  if (m !== TEST_IDS.length) throw new Error(`score matrix has ${m} columns, expected 102`);

  const { iters, tau } = manifest.config.sinkhorn;
  const opts = { iters, tau, cslsK: cfg.cslsK };
  const pre = rankScores(scores.data, n, { ...opts, useSinkhorn: false });
  const t2 = performance.now();
  const post = rankScores(scores.data, n, opts);
  timingsMs.postprocess = performance.now() - t2;

  let shipped: PipelineResult["shipped"] = null;
  try {
    const r = await fetchModelNpy(model, subject, "ranked");
    let equalRows = 0;
    for (let i = 0; i < n; i++) {
      const row = r.data.subarray(i * 102, (i + 1) * 102);
      if (post.ranked[i].every((v, j) => v === row[j])) equalRows++;
    }
    shipped = { equalRows, total: n };
  } catch {
    /* no reference ranking exported: nothing to compare against */
  }

  let labels: number[] | null = null;
  if (info.hasLabels) labels = Array.from((await fetchNpy("labels", subject)).data);

  return {
    model,
    subject,
    hasLabels: !!labels,
    isDevSubject: ["sub-01", "sub-02", "sub-03", "sub-04"].includes(subject),
    n,
    pre,
    post,
    shipped,
    labels,
    preAcc: labels ? accuracy(pre.ranked, labels) : null,
    postAcc: labels ? accuracy(post.ranked, labels) : null,
    timingsMs,
  };
}
