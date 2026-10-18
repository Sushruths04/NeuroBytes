/**
 * Plain-language version of submission2/WHY_IT_WORKS.md. Every number and
 * citation here comes from that document (or submission2/README.md for the
 * progression). Links are built only from identifiers the document gives
 * (DOI, PMID, PMC, arXiv). Where it gives none, we link a Google Scholar
 * search for the title instead of guessing a URL.
 */

export interface Citation {
  id: string;
  authors: string;
  title: string;
  venue: string;
  /** what the paper is cited for */
  role: string;
  url: string;
  /** "verified" = checked against full text; "abstract" = read at abstract level only */
  checked?: "verified" | "abstract";
}

const scholar = (q: string) => `https://scholar.google.com/scholar?q=${encodeURIComponent(q)}`;
const pmid = (n: string) => `https://pubmed.ncbi.nlm.nih.gov/${n}/`;
const pmc = (n: string) => `https://www.ncbi.nlm.nih.gov/pmc/articles/${n}/`;
const arxiv = (n: string) => `https://arxiv.org/abs/${n}`;
const doi = (d: string) => `https://doi.org/${d}`;

export const CITATIONS: Citation[] = [
  { id: "rossion2003", authors: "Rossion, B. et al.", title: "Early lateralization and orientation tuning for face, word and object processing in the visual cortex", venue: "NeuroImage (2003)", role: "Object categories are distinguishable at 130–170 ms, peaking at the N170.", url: pmid("14642472") },
  { id: "dirusso", authors: "Di Russo, F. et al.", title: "Cortical sources of the early components of the visual evoked potential", venue: "Human Brain Mapping", role: "Where the C1, P1 and N1 components come from.", url: pmc("PMC6871868") },
  { id: "luck", authors: "Luck, S. J. et al.", title: "Event-related brain potentials in the study of visual selective attention", venue: "", role: "P1 at 80–130 ms, N1 at 120–200 ms.", url: pmid("9448241") },
  { id: "cichy2016", authors: "Cichy, R. M., Khosla, A., Pantazis, D., Torralba, A., Oliva, A.", title: "Comparison of deep neural networks to spatio-temporal cortical dynamics of human visual object recognition reveals hierarchical correspondence", venue: "Scientific Reports 6:27755 (2016)", role: "Earlier network layers match earlier brain responses; deeper layers match later ones.", url: doi("10.1038/srep27755"), checked: "verified" },
  { id: "eegdnn2020", authors: "", title: "Time-resolved correspondences between deep neural network layers and EEG measurements in object processing", venue: "(2020)", role: "The same layer-to-latency ordering, measured with EEG.", url: pmid("32388211") },
  { id: "radovanovic2010", authors: "Radovanović, M., Nanopoulos, A., Ivanović, M.", title: "Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data", venue: "JMLR (2010)", role: "In high dimensions, a few points become everyone's nearest neighbour.", url: scholar("Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data") },
  { id: "hubclip2025", authors: "", title: "Balancing Hub Centrality in Cross-Modal Retrieval", venue: "arXiv:2503.10526 (2025)", role: "The same hub problem inside CLIP-style retrieval.", url: arxiv("2503.10526"), checked: "abstract" },
  { id: "hubling", authors: "", title: "Hubness, Not Anisotropy, Drives Cross-Lingual Retrieval Asymmetry in Multilingual Embedding Models", venue: "arXiv:2605.26575", role: "Recommends hubness-corrected similarity (CSLS) over plain cosine.", url: arxiv("2605.26575"), checked: "abstract" },
  { id: "cuturi2013", authors: "Cuturi, M.", title: "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", venue: "NeurIPS (2013)", role: "Sinkhorn row/column scaling as the core of entropic optimal transport.", url: scholar("Cuturi Sinkhorn Distances NeurIPS 2013") },
  { id: "sinkeff", authors: "", title: "On the Efficiency of Sinkhorn-Knopp for Entropically Regularized Optimal Transport", venue: "arXiv:2604.03787", role: "Convergence of the Sinkhorn–Knopp iteration.", url: arxiv("2604.03787"), checked: "abstract" },
  { id: "cormack2009", authors: "Cormack, G. V., Clarke, C. L. A., Büttcher, S.", title: "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods", venue: "SIGIR (2009)", role: "Combine rankings, not raw scores, so no calibration is needed.", url: scholar("Reciprocal rank fusion outperforms Condorcet and individual rank learning methods") },
  { id: "haxby2020", authors: "Haxby, J. V. et al.", title: "Hyperalignment: Modeling shared information encoded in idiosyncratic cortical topographies", venue: "eLife (2020)", role: "Brains share information but encode it in individual layouts; align function, not anatomy.", url: pmc("PMC7266639") },
  { id: "hybridhyper", authors: "", title: "Hybrid hyperalignment", venue: "", role: "A variant of functional alignment across subjects.", url: pmc("PMC8273921") },
  { id: "crosssub2026", authors: "", title: "Structured Visual Target Learning for Cross-Subject EEG-to-Image Retrieval", venue: "arXiv:2609.36971 (2026)", role: "Cross-subject transfer is an active problem in this exact task.", url: arxiv("2609.36971"), checked: "abstract" },
  { id: "song2024", authors: "Song, Y., Liu, B., Li, X., Shi, N., Wang, Y., Gao, X.", title: "Decoding Natural Images from EEG for Object Recognition (NICE)", venue: "ICLR 2024, arXiv:2308.13234", role: "Benchmark: 15.6% top-1 / 42.8% top-5, 200-way zero-shot.", url: arxiv("2308.13234"), checked: "verified" },
  { id: "gifford2022", authors: "Gifford, A. T. et al.", title: "A large and rich EEG dataset for modeling human visual object recognition (THINGS-EEG2)", venue: "NeuroImage 264:119754 (2022)", role: "The image-EEG benchmark dataset.", url: pmc("PMC9771828") },
  { id: "neuroclip", authors: "", title: "NeuroCLIP: Brain-Inspired Prompt Tuning for EEG-to-Image Multimodal Contrastive Learning", venue: "arXiv:2511.09250", role: "Contrastive training: 63.2% top-1, 200-way.", url: arxiv("2511.09250"), checked: "abstract" },
];

export interface Finding {
  id: string;
  title: string;
  /** one sentence a non-specialist can follow */
  plain: string;
  measured: string[];
  why: string;
  takeaway: string;
  cites: string[];
}

/** The four design choices that produced the result, largest first. */
export const KEY_CHOICES: Finding[] = [
  {
    id: "windows",
    title: "Look at the right moment: 50–400 ms after the video starts",
    plain: "Each recording is 3 seconds long, but the part that tells you which video was playing comes in the first fraction of a second. So we only use that part.",
    measured: [
      "The best windowed scorer (50–400 ms) reaches validation MRR 0.2016; the best whole-epoch scorer only 0.1184.",
      "Adding windows moved pooled top-5 from 39.95% to 47.30% in one step, more than every other choice combined.",
      "The later the window, the worse it does: 50–400 ms 0.2016, 200–400 ms 0.1380, 400–700 ms 0.0776 (last of 15).",
    ],
    why: "This is exactly where visual neuroscience says object information lives. The early visual responses are P1 (about 80–130 ms) and N1 (about 120–200 ms), and object categories can be told apart from 130–170 ms, peaking at the N170. After about 400 ms the response to the stimulus has mostly faded. What remains is ongoing brain rhythm and preparation to move, which says nothing about which video is on screen.",
    takeaway: "The biggest win came from knowing when to look, not from a bigger video model. Timing is the one thing EEG measures better than any other method. Separately, the mindread pipeline, which shares no code with this one, found the same interval: encoding peaks around 225 ms and is near zero after 400 ms.",
    cites: ["rossion2003", "dirusso", "luck"],
  },
  {
    id: "bijection",
    title: "Use the structure of the task: each video plays exactly once",
    plain: "Every subject saw each of the 102 test videos once. If two recordings both claim the same video, at least one of them is wrong, so we adjust the scores until each video is shared out roughly once.",
    measured: ["Sinkhorn normalisation of the 102 × 102 score matrix: top-5 53.43% → 59.56% (+6.13 pp), the largest gain after windows."],
    why: "Sinkhorn–Knopp scaling alternately normalises rows and columns until the matrix is doubly stochastic. It is the core computation of entropic optimal transport. Row sums of one mean each recording spreads its belief over the videos; column sums of one mean each video can be matched about once. Applying the constraint softly keeps a full ranking, and top-5 needs one. A hard one-to-one assignment (Hungarian) throws the ranking away, and it failed.",
    takeaway: "Nothing about the EEG or the video features changed. We simply stopped treating 102 linked trials as 102 independent problems.",
    cites: ["cuturi2013", "sinkeff"],
  },
  {
    id: "hubness",
    title: "Cancel “hub” videos that look like a match for everything",
    plain: "Some videos score well against almost every recording, simply because of where they sit in the feature space. We subtract each video's average score so it has to earn a high rank.",
    measured: [
      "Subtracting each candidate's mean score across queries: +1.7 pp top-5.",
      "Ranks are combined by reciprocal rank fusion. On its own that is slightly worse than z-scoring (50.98% vs 51.23%); with hubness correction added it is the best option (53.43%).",
    ],
    why: "Hubness is a known effect of high dimensions: a few points become nearest neighbours of almost everything, and retrieval gets worse. Our feature spaces have 512–4096 dimensions and the queries are noisy, so this is a textbook case. The same problem and fix (CSLS) appear in cross-lingual and CLIP-style retrieval. Reciprocal rank fusion lets 509 scorers with very different score scales be averaged without calibrating each one.",
    takeaway: "Each part tested alone looked not worth keeping; they only pay off together. Post-processing choices should be evaluated jointly.",
    cites: ["radovanovic2010", "hubclip2025", "hubling", "cormack2009"],
  },
  {
    id: "alignment",
    title: "Borrow data from other people, after translating it",
    plain: "Every head mixes the brain's signals into 17 electrodes differently, so you can't just add another person's recordings. We learn a translation from each other subject into this subject's own space, using the 695 videos everyone saw, and only then pool the data.",
    measured: [
      "+1.54 pp on validation, +0.98 pp on the dev test; every subject improved.",
      "Control: pooling the other subjects without the translation drops validation from 36.28% to 29.23% (−7.05 pp).",
    ],
    why: "This is functional alignment, the idea behind hyperalignment in fMRI. People share the information but encode it in individual layouts, so you align by function rather than anatomy. Our −7.05 pp control is an EEG version of that premise: raw donor recordings act as label noise. A simple per-donor ridge is enough here because the target, the video embedding, is the same for everyone. Only the EEG side needs translating.",
    takeaway: "The control matters as much as the gain. It shows that the translation, not the extra rows, is doing the work.",
    cites: ["haxby2020", "hybridhyper", "crosssub2026"],
  },
];

export const SUPPORTING: Finding[] = [
  {
    id: "layers",
    title: "Mid-level CNN layers beat large transformers",
    plain: "The video features that match EEG best come from the middle of older image networks (ResNet-50 layer 3, AlexNet fc6/fc7), not from the final output of the biggest modern models.",
    measured: ["Top validation scorers: resnet50 layer3 (MRR 0.2016), alexnet fc6 (0.1984), alexnet fc7, ahead of DINOv2, CLIP and VideoMAE pooled outputs.", "Treating each layer as its own representation: +2.45 pp over pooled outputs."],
    why: "Brain and network line up stage by stage. Early layers match early responses and deeper layers match later ones (Cichy et al. 2016 with MEG; the same ordering has been shown in EEG). We decode 50–400 ms, which is where mid-level visual representations live, so mid-level layers are the right partner.",
    takeaway: "Model size is the wrong axis. What matters is the level of abstraction that matches the time window you can decode.",
    cites: ["cichy2016", "eegdnn2020"],
  },
];

/** submission2/README.md: measured progression of pooled top-5 (sub-01..04). */
export const PROGRESSION: { step: string; top5: number }[] = [
  { step: "Start", top5: 32.84 },
  { step: "", top5: 38.24 },
  { step: "", top5: 39.95 },
  { step: "Windows", top5: 47.3 },
  { step: "", top5: 50.74 },
  { step: "Hubness", top5: 52.45 },
  { step: "Selection", top5: 53.43 },
  { step: "Sinkhorn", top5: 59.56 },
  { step: "Alignment", top5: 60.54 },
];

export const BENCHMARKS = [
  // C13 numbers: the team's configs/frozen_C13.json (also src/content/models.ts); the rest from submission2/README.md
  { system: "C13 (Latest Pipeline)", task: "102-way, 3-s video, 17-channel EEG", top1: "36.03%", top5: "67.16%", highlight: true },
  { system: "Winfuse (standalone)", task: "102-way, same data", top1: "30.6%", top5: "60.5%" },
  { system: "NICE (Song et al., ICLR 2024)", task: "200-way zero-shot, THINGS-EEG2 images", taskCite: "gifford2022", top1: "15.6%", top5: "42.8%", cite: "song2024" },
  { system: "Reference baseline", task: "102-way, same data", top1: "1.7%", top5: "9.3%" },
  { system: "Chance", task: "102-way", top1: "0.98%", top5: "4.90%" },
];
