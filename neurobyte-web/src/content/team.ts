/**
 * Team content from the NeuroDecode site (website/content.json, 18 Oct 2026),
 * copied verbatim except: C13's numbers and the robustness checks come from the final
 * submission repo (neurobytes: configs/frozen_C13.json, RESULTS.md, 18 Oct 2026).
 * All numbers are the official dev test, sub-01..04 (408 recordings).
 * The science cards' HTML is converted to structured text (no innerHTML).
 */
export interface System { name: string; top1: number; top5: number; note: string; final?: boolean }
export interface Experiment { id: string; what: string; val: string; dev: string; verdict: string; kind: "pos" | "neu" | "neg" }

export const SYSTEMS: System[] = [
  {
    "name": "J1",
    "top1": 11.03,
    "top5": 29.66,
    "note": "ridge baseline"
  },
  {
    "name": "T1",
    "top1": 11.52,
    "top5": 35.54,
    "note": "time-resolved"
  },
  {
    "name": "T2",
    "top1": 13.97,
    "top5": 38.72,
    "note": "V-JEPA 2 + S3D"
  },
  {
    "name": "WF1",
    "top1": 23.04,
    "top5": 52.45,
    "note": "winfuse 509"
  },
  {
    "name": "C3",
    "top1": 26.47,
    "top5": 57.35,
    "note": "both pipelines"
  },
  {
    "name": "C5",
    "top1": 30.88,
    "top5": 62.25,
    "note": "+ CSLS"
  },
  {
    "name": "C10",
    "top1": 32.6,
    "top5": 66.67,
    "note": "+ align, Sinkhorn"
  },
  {
    "name": "C11",
    "top1": 34.56,
    "top5": 66.18,
    "note": "+ neural net"
  },
  {
    "name": "C12",
    "top1": 35.05,
    "top5": 67.4,
    "note": "+ 7 heads"
  },
  {
    "name": "C13",
    "top1": 36.03,
    "top5": 67.16,
    "note": "+ 2 neural nets (h6, h10)",
    "final": true
  }
];

export const EXPERIMENTS: Experiment[] = [
  {
    "id": "Robustness",
    "what": "Same-seed rebuild on a fresh GPU; five new seeds; validation on held-out training videos",
    "val": "all six people: top-1 14.95%, top-5 39.67%",
    "dev": "bit-identical; new seeds 34.31 / 67.65",
    "verdict": "robust",
    "kind": "pos"
  },
  {
    "id": "Per-person",
    "what": "Per-person fine-tuning (C14); per-person fusion weights",
    "val": "fine-tune: network 6/6 better; weights unstable",
    "dev": "C14 34.1 vs C13 35.3 (seed noise)",
    "verdict": "tie / rejected",
    "kind": "neu"
  },
  {
    "id": "C12 heads",
    "what": "Network also predicts AlexNet fc6, ResNet-50 layer3, VideoMAE (7 heads)",
    "val": "top-1 16.6→17.1",
    "dev": "34.6→35.0 top-1, 66.2→67.4 top-5",
    "verdict": "kept",
    "kind": "pos"
  },
  {
    "id": "Cognitive heads",
    "what": "Cognitive attributes, human labels, captions as extra network targets",
    "val": "top-5 43.7→44.0, top-1 17.2→16.9",
    "dev": "33.8 / 67.6",
    "verdict": "helps shortlist only",
    "kind": "neu"
  },
  {
    "id": "Ridge TTA",
    "what": "Score every ridge scorer at ±16 ms shifts and average",
    "val": "single +2.8 top-5; fusion top-1 17.3→16.0",
    "dev": "31.4",
    "verdict": "hurts top-1",
    "kind": "neg"
  },
  {
    "id": "X14 neural",
    "what": "Contrastive multi-person CNN (0–600 ms, 4 embedding heads, 5 seeds), weight 0.1 chosen on validation",
    "val": "top-1 15.9→16.6 · top-5 41.8→43.0",
    "dev": "C10 32.6→34.6 top-1",
    "verdict": "kept",
    "kind": "pos"
  },
  {
    "id": "X14 linear",
    "what": "Linear-trunk network (Codex component finding) alone and with conv",
    "val": "alone 8.7 vs conv 6.4; in fusion tie (16.5)",
    "dev": "alone 14.0 / 38.5",
    "verdict": "strong alone, redundant",
    "kind": "neu"
  },
  {
    "id": "Ensembles",
    "what": "15 seeds, window ensembles; re-ranking grid; equal families",
    "val": "±0.15",
    "dev": "within noise",
    "verdict": "no gain",
    "kind": "neg"
  },
  {
    "id": "X11 cognitive",
    "what": "63 CLIP zero-shot brain attributes + scene-change dynamics",
    "val": "top-1 15.9→15.6",
    "dev": "31.1",
    "verdict": "no gain",
    "kind": "neg"
  },
  {
    "id": "X11 human text",
    "what": "Human descriptions/labels of each clip (EMD metadata)",
    "val": "top-1 15.9→15.6",
    "dev": "31.9",
    "verdict": "no gain",
    "kind": "neg"
  },
  {
    "id": "X12 stacking",
    "what": "Family weights learned with a rank-1 loss",
    "val": "LOSO top-1 +0.5",
    "dev": "top-5 +2.2, top-1 −1.0",
    "verdict": "marginal",
    "kind": "neu"
  },
  {
    "id": "X10 consensus",
    "what": "Borrow other people's scores for the same clips",
    "val": "+0.5 (noise)",
    "dev": "32.6→32.1",
    "verdict": "hurts",
    "kind": "neg"
  },
  {
    "id": "X13 late windows",
    "what": "Frame-locked decoding after 800 ms",
    "val": "at chance",
    "dev": "—",
    "verdict": "finding",
    "kind": "neu"
  },
  {
    "id": "X15 CAR",
    "what": "Common-average re-reference (organizer: none applied)",
    "val": "+0.3 top-1, rank worse",
    "dev": "12.5→11.5 (30 scorers)",
    "verdict": "no gain",
    "kind": "neg"
  }
];

/** Mean rank of the true video (of 102) by latency window (ms), from the team's window sweep. Lower is better. */
export const LATENCY: [string, number][] = [["50–200", 35.5], ["50–400", 27.1], ["200–500", 32.9], ["480–830", 45.9], ["910–1260", 48.3], ["1340–1690", 52.2], ["1760–2110", 51.2], ["2190–2540", 51.0], ["2620–2970", 50.1]];

export interface ScienceCard {
  num: string;
  title: string;
  big?: boolean;
  paragraphs: string[];
  /** label/value tiles (the original site's "timeline" boxes) */
  tiles?: [string, string][];
  /** paragraph after the tiles */
  after?: string;
  bullets?: string[];
}

/** The team site's seven science cards (website/content.json "science"), as structured text. */
export const TEAM_SCIENCE: ScienceCard[] = [
  {
    big: true,
    num: "800 ms",
    title: "The decodable signal is the onset response",
    paragraphs: [
      "Decoding is strongest from 50–400 ms after the clip starts, the window of the visual P1/N1 components. Every window after ~800 ms is at chance, even when locked to the frame on screen. The brain's later responses to a 3-second clip are too small for 17 channels to separate 102 videos.",
    ],
  },
  {
    big: true,
    num: "robust",
    title: "Robust, not random",
    paragraphs: ["Every gain we kept had to survive three checks."],
    tiles: [
      ["Rebuild from scratch", "retraining with the same seeds on a fresh GPU gives bit-identical scores for all six people"],
      ["New random seeds", "five new seeds give 34.31 / 67.65 vs 36.03 / 67.16: within one standard error (≈2.4 points)"],
      ["Clean machine", "released weights on a fresh machine, natively and in Docker, reproduce 36.03 / 67.16"],
    ],
  },
  {
    num: "r = 0.16",
    title: "Misses are noise, not hard videos",
    paragraphs: [
      "If some videos were intrinsically hard, the same ones would fail for everyone. They don't: per-video difficulty barely correlates between people. The true video is usually close (rank 2–5), and single-recording noise decides the last step.",
    ],
  },
  {
    num: "V-JEPA 2",
    title: "Deep video models already speak the brain's language",
    paragraphs: [
      "Self-supervised video models (V-JEPA 2 giant) and supervised CNN layers (AlexNet, ResNet) match the early visual response best. Interpretable attributes such as faces, emotion and actions add nothing on top: the embeddings already contain them.",
    ],
  },
  {
    num: "6 brains",
    title: "Learning across people works only with alignment",
    paragraphs: [
      "The 17 channels mix brain sources differently in every head. A per-person spatial map (ridge alignment for the scorers, a learned spatial layer in the neural network) lets each person borrow the others' recordings.",
    ],
  },
  {
    num: "CSLS",
    title: "Hubs and balance",
    paragraphs: [
      "Some videos look similar to everything and win too often (hubs). CSLS removes each candidate's average pull; Sinkhorn makes every video win about once across the 102 recordings. Together they added about 6 points of top-1.",
    ],
  },
  {
    num: "next",
    title: "What would move the needle",
    paragraphs: [],
    bullets: [
      "More repeats per training clip (cleaner targets)",
      "Larger contrastive models pretrained on public EEG",
      "Per-person spatial filters from the 127-channel original montage",
    ],
  },
];

/** The team site's ERP timeline (first science card): what each latency band carries. */
export const ERP_TIMELINE: [string, string][] = [
  ["60–120 ms", "C1/P1 · contrast, motion"],
  ["150–200 ms", "N1/N170 · objects, faces"],
  ["200–300 ms", "P2/EPN · salience, emotion"],
  ["300–500 ms", "P300 · novelty"],
  [">800 ms", "no decodable identity"],
];
