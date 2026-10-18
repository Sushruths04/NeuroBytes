# Why this pipeline works: the domain knowledge behind each design choice

The challenge allocates **30% to qualitative criteria** — technical soundness and novelty
(15%), engineering quality and reproducibility (10%), presentation including failure
analysis (5%) — and states that it is "particularly interested in approaches that not
only are accurate but also provide scientifically meaningful insights into the
similarities and differences between how biological and artificial systems process visual
stimuli."

This document is the answer to "why does this work, and what does it tell us about brains
versus models". Every design choice is paired with (a) the number we measured and (b) the
prior literature that predicts or explains it. Where our result **contradicts** what the
literature would lead you to expect, that is marked explicitly — those are the more
interesting entries.

**Scope.** This covers the `winfuse` side of the pipeline — the part I can source from my
own measurements. It is written to sit alongside `science.md` (the mindread-side findings, in the team repo), which records
the `mindread` measured findings but currently carries **no citations**. The two are meant
to be read together: `science.md` says *what we found*, this says *why it was findable*.
The `mindread`-side feature choices (SigLIP, V-JEPA 2, S3D, CLAP, the low-level time
courses, the TRF work) are **not** covered here and are marked as gaps in §13 — they need
an owner who can source them.

Reference numbers: `winfuse` alone reaches top-1 30.64%, top-5 60.54% on sub-01..04 (408
trials, 102 candidates; chance 0.98% / 4.90%). The repo's current frozen model is
**C5.blend_csls** at 30.9% / 62.3%, which blends `winfuse` with `mindread` features — so
the combined result is better than either side alone, and §12 argues that is not an
accident.

---

## 1. Latency windows: the single largest factor

**What we did.** Instead of feeding the whole 3-second epoch to the regression, we built
15 feature sets, each the preprocessed trace cropped to a window after video onset
(50–200, 50–400, 80–320, 100–300, … 400–700 ms).

**What we measured.** The best windowed scorer (`win_50_400 × resnet50 layer3`) reaches
validation MRR **0.2016**; the best whole-epoch scorer reaches **0.1184**. Adding the
window dimension moved pooled top-5 from 39.95% to **47.30%** in one step — more than
every other design choice combined. Within the sweep, performance falls monotonically as
the window moves later: 50–400 ms scores 0.2016, 200–400 ms scores 0.1380, and 400–700 ms
is the weakest of all 15 at **0.0776**.

**Why this is expected, and what it confirms.** The timing is not arbitrary — it lands
exactly on the interval that the visual ERP literature identifies as carrying
category-level information. Object categories are reliably distinguishable from
occipito-temporal recordings "as early as 130–170 ms… peaking at the level of the N170
component" (Rossion et al.), and the early visual components C1, P1 (≈80–130 ms) and N1
(≈120–200 ms) are the classical stimulus-locked responses (Luck et al.; Di Russo et al.).
Our 50–400 ms optimum brackets P1, N1 and the immediately following interval.

The decline after ~400 ms is equally expected: stimulus-locked evoked activity has largely
resolved, and what remains in a 3-second clip is ongoing oscillatory activity plus
anticipatory and motor-preparation signals that are not specific to *which* video is
playing. Including it adds variance without adding signal, so the ridge's effective
signal-to-noise ratio falls.

**The insight worth stating.** Our strongest single feature is not a better video model —
it is knowing *when to look*. A pipeline that treats the epoch as an undifferentiated
feature vector discards the one thing EEG is uniquely good at: millisecond timing. This is
a case where cognitive-neuroscience domain knowledge, not machine-learning capacity, was
the binding constraint.

**Independent corroboration within this project.** `science.md` §1 reaches the same
conclusion from a completely different method — a time map of held-out encoding r² in 50 ms
bins — and puts the peak at **~225 ms**, with low-level features peaking at 175 ms and
"near zero after 400 ms". My window sweep, which never looked at encoding r², selects
50–400 ms as optimal and ranks 400–700 ms last of 15. Two pipelines sharing no code, using
different metrics, converge on the same interval. That is the strongest internal evidence
we have that the effect is real rather than a modelling artefact.

---

## 2. Mid-level CNN layers beat large transformers

**What we measured.** The strongest stimulus representations are **not** the final pooled
embeddings of the largest models. Ranked by validation MRR, the top scorers use
`resnet50@layer_layer3` (0.2016), `alexnet_layers@layer_fc6` (0.1984) and
`alexnet_layers@layer_fc7` — mid-to-late convolutional and first fully-connected layers of
two comparatively old CNNs — ahead of DINOv2, CLIP and VideoMAE pooled outputs. Treating
individual layers as separate representations was worth +2.45 pp over pooled outputs
alone.

**Why this is expected.** This is the hierarchical-correspondence result, and it is one of
the better-replicated findings in the field. Cichy, Khosla, Pantazis, Torralba and Oliva
(*Scientific Reports* 6:27755, 2016) compared an 8-layer DNN against millisecond-resolved
MEG and found "an ordered relationship between the stages of processing in a DNN and the
time course with which object representations emerge in the human brain" — early layers
match early responses, deeper layers match later ones. Greene & Hansen and subsequent
time-resolved EEG work (Nonaka et al.; "Time-resolved correspondences between deep neural
network layers and EEG measurements in object processing", 2020) report the same ordering
in EEG specifically: layer depth and the latency of peak correlation co-vary.

Our pipeline is dominated by **early-to-mid latency windows**, so hierarchical
correspondence predicts that **early-to-mid network layers** should match best. They do.
The two findings are the same finding viewed from two sides: we are decoding the 50–400 ms
interval, which is where mid-level visual representations live, so mid-level CNN features
are the right partner for it.

**The insight worth stating.** Model scale is the wrong axis here. A 2012-era AlexNet
fully-connected layer beats DINOv2's pooled output because it sits at the right level of
abstraction for the latency we can actually decode. If we had only used final-layer
embeddings from the largest available models — the obvious default — we would have left a
substantial part of the result on the table.

---

## 3. Why spatial covariance features failed (a negative result with a clean mechanism)

**What we measured.** Channel covariance computed inside the best windows and projected to
the Riemannian tangent space lands **at or below chance**: 3.43% top-5 against 4.90%
chance, and fusing it into the ensemble costs 5.26 pp. In an earlier phase, 17 EEG feature
extractors including `cov_tangent`, `cov_vec`, `xdawn_evoked`, `spatial_pca` and
`ica_components` were benchmarked and none beat chance on stimulus identity.

**Why this is expected on reflection.** Riemannian covariance methods are among the
strongest approaches in BCI — Barachant et al.'s tangent-space framework and its
descendants have won multiple BCI competitions — but the paradigms where they dominate are
**oscillatory**: motor imagery (sensorimotor mu/beta rhythms) and SSVEP, where the
discriminative information genuinely lives in the spatial covariance of band-limited
power. A covariance matrix is a second-order summary that **integrates over time** and
therefore discards the waveform.

Our signal is an **evoked response**: a specific deflection with a specific shape at a
specific latency. Covariance throws away exactly the dimension that carries it. The eNeuro
study "Representational Content of Oscillatory Brain Activity during Object Recognition"
(2021) makes the complementary point from the data side — oscillatory power components
correlate mainly with low-level V1-like representations rather than with the
category-level structure we need.

**The insight worth stating.** "Use the method that wins BCI competitions" is bad advice
without asking what *kind* of signal the paradigm produces. Oscillatory-paradigm tooling
and evoked-paradigm tooling are not interchangeable, and the covariance family is firmly
in the first camp.

---

## 4. Hubness correction and the Sinkhorn bijection step: +7.8 pp combined

**What we did.** Two post-hoc transforms of the 102×102 score matrix. First, subtract each
*candidate's* mean score across queries (hubness correction, +1.7 pp). Then iterate row and
column normalisation toward a doubly stochastic matrix (Sinkhorn, a further **+6.13 pp**,
53.43% → 59.56%).

**Why this is expected.** Hubness is a well-characterised consequence of
high-dimensionality: Radovanović, Nanopoulos and Ivanović ("Hubs in Space: Popular Nearest
Neighbors in High-Dimensional Data", *JMLR* 2010) showed that in high-dimensional spaces a
small number of points become disproportionately frequent nearest neighbours of everything
else, which systematically degrades retrieval. The standard remedy in cross-lingual word
retrieval is CSLS (Conneau, Lample et al.), which subtracts a local neighbourhood mean; the
same problem and remedy have since been documented inside CLIP-style cross-modal retrieval
("Balancing Hub Centrality in Cross-Modal Retrieval", 2025) and in multilingual embedding
retrieval, where CSLS is now recommended as the default metric over plain cosine.

Our setting is a textbook instance: we are retrieving among 102 candidates in embedding
spaces of 512–4096 dimensions, from a noisy query. Some candidate videos sit in dense
regions and score well for *every* recording. Removing the column mean cancels that
query-independent bias.

Sinkhorn is the principled completion of the same idea. Sinkhorn–Knopp matrix scaling
alternately normalises rows and columns until the matrix is doubly stochastic, and it is
the computational core of entropically regularised optimal transport (Cuturi, 2013). Here
it has a direct task interpretation: the challenge description states each test file holds
"one recording per test video", so the correct answer across a subject's block is a
**bijection** — 102 recordings matched one-to-one with 102 videos. Row sums of one say
"each recording distributes its belief over videos"; column sums of one say "each video is
available to be matched about once". Iterating imposes the constraint softly, which is what
we need because top-5 requires a full ranking rather than a hard assignment.

**Where our result diverges from a sibling result, and why.** The `mindread` pipeline tested
CSLS at k=10 and found it within noise; the full assignment solution (Hungarian) was
rejected there and also failed for us. But Sinkhorn gave us the largest single gain in the
whole project. Our explanation: CSLS subtracts a mean over the k nearest neighbours, and
k=10 out of 102 candidates averages over ~10% of the pool, smoothing away much of the
per-candidate bias it is meant to remove. Sinkhorn removes the bias globally and
iteratively instead. Consistently, `mindread` later found CSLS with **k=2** — far more
local, hence sharper — did help. Hard assignment fails for a different reason: it discards
the ranking that the metric actually scores.

**The insight worth stating.** The largest post-hoc gain came from exploiting the
*structure of the task* rather than improving the model. Nothing about the EEG or the video
features changed; we simply stopped pretending the 102 trials were independent problems when
the task description says they are not.

---

## 5. Rank-space fusion rather than score averaging

**What we did.** Each of 509 scorers contributes `1/(60 + rank)` rather than its raw score,
and the ensemble is a plain average.

**Why.** This is Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009), which is
standard in information retrieval precisely because it needs no score calibration across
systems. Our scorers have wildly different score distributions — a cosine against a
4096-dimensional AlexNet layer is not comparable to a reliability-weighted correlation in
EEG space — and rank transformation makes them combinable without per-scorer calibration.

**A measured subtlety.** Reciprocal rank *alone* is slightly **worse** than plain
z-scoring (50.98% vs 51.23%); it only becomes the best option once hubness correction is
applied on top (53.43%). Either component tested in isolation would have looked not worth
keeping. This is a concrete argument for evaluating post-processing choices jointly rather
than as an ablation ladder.

---

## 6. Cross-subject alignment: borrowing data across brains

**What we did.** For each target subject, map every donor subject's EEG features into the
target's own feature space with a ridge fitted on the videos both saw, then pool the mapped
rows before fitting. **695 training videos were viewed by all six subjects**, which is what
makes the map estimable.

**What we measured.** +1.54 pp on validation and +0.98 pp on the dev test, with every
subject improving. Critically, the **control**: pooling donors' EEG *without* the map
collapses validation from 36.28% to **29.23%**.

**Why this is expected.** This is functional alignment, and the fMRI literature has
established both halves of our result. Haxby et al.'s **hyperalignment** (and the
*eLife* 2020 formulation, "Hyperalignment: Modeling shared information encoded in
idiosyncratic cortical topographies") is built on exactly the premise our control
demonstrates: shared information is present across brains but encoded in *idiosyncratic*
topographies, so it "captures shared information by projecting pattern vectors… into a
common, high-dimensional information space, rather than by aligning topographies in a
canonical anatomical space". Chen et al.'s Shared Response Model is the low-rank variant.
The standard finding is that functional alignment enables cross-subject pooling where
anatomical alignment alone does not.

Our −7.05 pp unaligned control is a clean EEG-side replication of that premise: 17 scalp
channels are a different mixture of underlying sources in every head, so donor rows added
raw act as label noise. The direction of the benefit also matches the literature's
expectation that data-poor subjects gain most — in our 9-scorer arm, the weakest subject
gained +2.19 pp while the strongest *lost* 0.88 pp.

Cross-subject transfer is an active problem in exactly our task: "Structured Visual Target
Learning for Cross-Subject EEG-to-Image Retrieval" (2026) addresses it from the target side,
and NeuroCLIP reports inter-subject gains, which suggests this axis has more to give than we
extracted.

**Why a simple ridge sufficed instead of full hyperalignment.** Our regression predicts the
*video embedding*, which is identical for every subject. Only the EEG side carries
subject-specific geometry, so a per-donor linear map into the target's space is enough; there
is no need for a jointly-estimated common space. That is a small simplification worth noting
— the asymmetry of the problem removes the need for the harder algorithm.

---

## 7. Banded ridge: the literature's recommendation that did not transfer

**What the literature says.** When combining feature spaces of unknown relative scale, the
right thing is a separate regularisation hyperparameter per space — **banded ridge
regression** (Nunez-Elizalde et al. 2019; Dupré la Tour et al., "Feature-space selection with
banded ridge regression", *NeuroImage* 2022). It is the standard in voxelwise encoding work
and is explicitly recommended for layer-wise DNN features, where it "effectively removes
non-predictive or redundant layers and disentangles the contributions of each layer". The
sibling `mindread` pipeline found banded ridge among its better configurations.

**What we measured.** It **lost** to the matched control in both blockings: EEG-side 32.34%
vs 33.00%, stimulus-side 34.83% vs 36.15%.

**Why, and this is the interesting part.** The comparison is not banded-versus-unbanded in
the abstract — it is *banded ridge over k blocks* versus *k separate scorers fused in rank
space*. Concatenating blocks into one regression produces a single predicted embedding, in
which a weak block dilutes the shared prediction. Keeping the blocks as separate scorers
lets each contribute only its own *ranking*, after which hubness correction and averaging
down-weight the poor ones automatically. Per-block shrinkage can reweight blocks but cannot
put each one back into its own rank space.

**The insight worth stating.** Banded ridge is the right answer when the fusion happens
inside the regression. Once fusion moves to rank space — which the hubness and bijection
results above make clearly worthwhile — the premise changes and the recommendation
inverts. The literature is not wrong; it is answering a different question.

---

## 8. Why EEG augmentation hurt

**What we measured.** Gaussian input noise cost 2.90 pp; channel dropout cost 4.35 pp.

**Why this is expected.** Adding isotropic Gaussian noise to the inputs of a **linear**
model is mathematically equivalent to L2 (Tikhonov) regularisation — a classical result
(Bishop, 1995). We already select the ridge penalty by 5-fold cross-validation over 15
values, so noise augmentation cannot add anything; it can only re-derive a penalty we are
already optimising, through a worse-conditioned route. The evidence is visible in the
diagnostics: with noise augmentation, cross-validation selected a **5× smaller** alpha
(658 vs 3162), because the injected noise was doing the regularising.

There is a second, data-specific reason the direction is wrong. We measured that **test
recordings have 62–68% of the standard deviation of training recordings**, which is the
signature of the test recordings being averaged over more stimulus repetitions. The
augmentation literature points the same way: "Trial Averaging for Deep EEG Classification"
(2023) reports that averaging trials dramatically improves classification and that "simple
trial averaging outperformed a popular related algorithm, Mixup". The useful operation for
this dataset is **de**-noising the training data toward the test distribution, not adding
noise to it.

**The insight worth stating.** Augmentation is a reasonable default for a deep network with
a free capacity budget. For a cross-validated linear model it is provably redundant, and
when the test set is *cleaner* than the training set it is actively counterproductive.

---

## 9. Where we sit against published work

Direct comparison is impossible — different dataset, different candidate-set size, images
versus 3-second video — but it gives a sanity check on the order of magnitude.

| system | task | top-1 | top-5 |
|---|---|---|---|
| **this pipeline** | 102-way, 3-s video, 17-channel EEG | **30.6%** | **60.5%** |
| NICE (Song et al., ICLR 2024) | 200-way zero-shot, THINGS-EEG2 images | 15.6% | 42.8% |
| organisers' reference baseline | 102-way, same data | 1.7% | 9.3% |
| chance (102-way) | | 0.98% | 4.90% |

The 200-way task is harder than 102-way, so these are not equivalent, and more recent
THINGS-EEG2 systems report considerably higher numbers (NeuroCLIP, 63.2% top-1 in 200-way)
using contrastive training that we did not attempt. The honest reading: our result is in the
right region for the field, achieved with ridge regression rather than a trained encoder,
and roughly **18× chance on top-1**.

What stands out methodologically is how much of our result came from classical
signal-analysis and retrieval reasoning rather than from model capacity: windows (+15.4 pp),
bijection (+6.1 pp), hubness (+1.7 pp) and cross-subject alignment (+1.0 pp) together
dominate every change we made to the feature extractors.

---

## 10. What is genuinely novel here, stated conservatively

Little of the machinery is new; each component is established in some field. What we have
not seen combined before:

1. **A systematic latency-window × network-layer grid as the feature basis**, which
   operationalises hierarchical correspondence (§2) as a *feature-engineering* strategy
   rather than only as an analysis result. The literature uses layer–latency correspondence
   to *describe* brain-model similarity; we use it to *build* a decoder.
2. **Sinkhorn applied to the candidate-set bijection in EEG retrieval.** Hubness correction
   and CSLS are known in cross-modal retrieval; exploiting the one-to-one structure of a
   trial block with full doubly-stochastic scaling, and measuring +6.13 pp, we have not seen
   reported for EEG stimulus identification.
3. **A cross-subject alignment control on the EEG side** that quantifies the cost of
   skipping alignment (−7.05 pp) alongside the benefit of doing it (+1.54 pp). The
   hyperalignment premise is usually asserted; here it is measured in both directions.
4. **A negative-result set with mechanisms** — banded ridge, augmentation, covariance
   features, per-subject selection, Hungarian assignment — each with an explanation for
   *why* it failed in this paradigm, which is the part that transfers to the next dataset.

---

## 11. Honest divergences from the literature

Flagged because they are the entries a reviewer should push on:

- **Banded ridge should have helped and did not** (§7). Our explanation is the rank-space
  fusion interaction, but we did not test the stronger version — banded ridge *inside* each
  scorer with rank fusion preserved across scorers. That may still work.
- **Encoding should contribute and does not.** 18 of our 509 scorers predict EEG from
  features, the direction most of the encoding-model literature uses (Naselaris et al.;
  Nishimoto et al.; Huth et al.). They are measurably complementary — enc-vs-dec score
  correlation 0.14–0.22, against 0.80 between two decoding scorers — yet removing them
  barely changes the result. Two explanations were tested and falsified (under-weighting;
  train/test amplitude mismatch). **This remains unresolved** and is the clearest gap
  between our result and what the field would predict.
- **Contrastive training is the field's current direction and we did not try it.** NICE and
  NeuroCLIP both train EEG encoders with InfoNCE against frozen image embeddings. We used
  ridge regression throughout. Given that the published contrastive systems beat the
  published ridge-style systems on THINGS-EEG2, this is the most likely place where our
  approach is leaving performance behind.
- **One subject resists everything.** sub-03 sits at 48.0% top-5 against 61.8–69.6% for the
  others and is diagnosed as intrinsically low-SNR (mean CV encoding r 0.027 against
  0.044–0.050; only 8% of embedding dimensions reach r>0.1 against 31–44%). Four general
  per-subject remedies all cost accuracy. The literature on individual differences in EEG
  decodability would predict such variation; we can characterise it but not fix it.

---

## 12. Three tensions with `science.md` that need resolving

Reading `science.md` against my results, three things disagree. All three are worth
resolving before the report, because each one changes what we would claim.

### 12a. Cross-subject alignment: "adds nothing" vs +1.54 pp

`science.md` §4 records, under things that did not work: *"Pooling other subjects: without
alignment it follows an inverted U (best weight 0.5–0.85). With subject-to-subject
alignment (PCA + ridge maps) it adds nothing on top of the encoder+decoder fusion (V1 rank
29.5–29.7 vs 29.7)."*

I measure the opposite: **+1.54 pp validation, +0.98 pp dev test, every subject improving**,
with a control showing unaligned pooling costs −7.05 pp.

Candidate explanations, in the order I would test them:

1. **Different metric and surface.** You measured V1 mean rank on training-video CV; I
   measured top-5 on *bijective* validation pools (102 trials with 102 distinct videos,
   candidate set exactly those videos). Mean rank is dominated by the tail; top-5 by the
   head. An improvement concentrated in the head can be invisible in mean rank.
2. **Different baseline strength.** Yours was measured *on top of encoder+decoder fusion*;
   mine on a decoding-only ensemble. If the encoder already captures the shared
   across-subject component, alignment has nothing left to add for you — which would be a
   genuinely interesting result rather than a null.
3. **Different map.** Yours is PCA + ridge; mine is a plain per-donor ridge in the original
   feature space, fitted on the videos both saw, with no PCA step. If your PCA was fitted on
   the donor and the target separately, the components may not correspond.
4. **Scale.** My gain *grows* with the number of scorers fused: +0.87 pp at 9, +1.30 at 24,
   +1.54 at 509. Measured on a small arm it looks like noise. I nearly discarded it myself.

Note that 1 and 2 would both mean nobody is wrong — we measured different quantities.

### 12b. Which subject is hardest: sub-06 or sub-03?

`science.md` §2 gives mean held-out encoder r: sub-01 0.174, sub-02 0.126, sub-03 0.135,
sub-04 0.091, sub-05 0.145, **sub-06 0.046** — so by that measure sub-06 is hardest and
sub-04 is second-hardest, with sub-03 mid-pack.

In my retrieval results **sub-03 is clearly the worst** (48.0% top-5 against 61.8–69.6%),
and sub-04 is fine. My own encoding diagnostic agrees with my retrieval result, not yours:
sub-03's mean CV encoding r is 0.027 against 0.044–0.050 for the others, with only 8% of
embedding dimensions reaching r>0.1 against 31–44%.

These cannot both be right as stated, so one of the following holds: the two encoder r
values are computed over different channel sets, time windows or feature spaces; or
predictability of the *EEG* (your r) and discriminability *between candidate videos* (my
top-5) genuinely come apart. The second would be a real finding worth a paragraph in the
report — a subject can have a reproducible response that is nonetheless not
stimulus-specific.

**This matters practically**: sub-06 is scored and sub-03 is not. If sub-06 is the hard one
by the measure that predicts leaderboard performance, our expectations should be lower than
my dev numbers suggest.

### 12c. The encoding direction: "largest single gain" vs ~zero contribution

`science.md` §5: *"The decoder direction is the largest single gain… improves V1 rank by
about 2 and V2 top-5 by about 4–5 points. The two directions make different errors."*

I agree decoding is the stronger direction and that the two make different errors — I
measure enc-vs-dec score correlation 0.14–0.22 against 0.80 between two decoding scorers,
which is exactly "different errors". But in my ensemble the 18 encoding scorers carry
**0.2% of the weight** and removing them barely changes the result. I tried two explanations
(under-weighting, train/test amplitude mismatch) and falsified both.

So we agree the directions are complementary and disagree on whether that complementarity
is *convertible*. Since your fusion extracts value from it and mine does not, the difference
is probably in **how** the two are combined — yours learns non-negative weights per scorer
on calibration splits, mine uses rank-space fusion with equal weights. That is a concrete,
testable difference and the most likely place where the C3/C5 blend is getting its edge
over either side alone.

---

## 13. Gaps — what this document does not cover

Marked explicitly so an owner can pick them up. I cannot source these honestly because they
are not my measurements.

| gap | what is needed |
|---|---|
| **SigLIP, V-JEPA 2 (giant L30, large), S3D, CLAP** | Why these specific models and layers? V-JEPA 2 L30 was your best single feature — is there a published reason a self-supervised *video* model at that depth should match EEG better than image models? |
| **The low-level time courses** | Luminance / contrast / motion / audio-envelope features carry the largest unique variance at 125–225 ms per `science.md` §1. That is a strong, citable claim about early visual processing and deserves grounding. |
| **TRF / slow-wave branch** | `docs/knowledge/research_notes/trf_stimulus_tracking.md` exists; §3's two-timescale model needs the same measurement-to-literature joining this document does for the winfuse side. |
| **The duplicate-grouping decision** | Near-duplicate videos are grouped in your CV. Worth a note on why that matters for an unbiased estimate. |
| **Contrastive training** | Neither side tried it, and it is the field's current direction (§11). Worth stating as a deliberate scope decision rather than an omission. |

## References

Verified sources, in the order they appear above.

**Visual ERP timing (§1)**
- Rossion, B. et al. Early lateralization and orientation tuning for face, word and object
  processing in the visual cortex. *NeuroImage* (2003). PMID 14642472. — object categories
  distinguishable at 130–170 ms, peaking at N170.
- Di Russo, F. et al. Cortical sources of the early components of the visual evoked
  potential. *Human Brain Mapping*. PMC6871868. — C1, P1, N1 sources.
- Luck, S. J. et al. Event-related brain potentials in the study of visual selective
  attention. PMID 9448241. — P1 80–130 ms, N1 120–200 ms.

**DNN–brain hierarchical correspondence (§2)**
- Cichy, R. M., Khosla, A., Pantazis, D., Torralba, A., Oliva, A. Comparison of deep neural
  networks to spatio-temporal cortical dynamics of human visual object recognition reveals
  hierarchical correspondence. *Scientific Reports* 6:27755 (2016). doi:10.1038/srep27755,
  PMID 27282108.
- Time-resolved correspondences between deep neural network layers and EEG measurements in
  object processing (2020). PMID 32388211.

**Oscillatory versus evoked representational content (§3)**
- Representational Content of Oscillatory Brain Activity during Object Recognition:
  Contrasting Cortical and Deep Neural Network Hierarchies. *eNeuro* 8(3) (2021).
- Barachant, A. et al. Riemannian geometry applied to BCI classification. — the
  tangent-space framework, developed for motor imagery / ERP / SSEP.

**Hubness and retrieval normalisation (§4)**
- Radovanović, M., Nanopoulos, A., Ivanović, M. Hubs in Space: Popular Nearest Neighbors in
  High-Dimensional Data. *JMLR* (2010).
- Balancing Hub Centrality in Cross-Modal Retrieval (2025). arXiv:2503.10526. — the hubness
  problem inside CLIP-style retrieval.
- Hubness, Not Anisotropy, Drives Cross-Lingual Retrieval Asymmetry in Multilingual
  Embedding Models. arXiv:2605.26575. — recommends CSLS over cosine as default.
- Sinkhorn–Knopp matrix scaling / entropically regularised optimal transport; see
  Cuturi, M. Sinkhorn Distances (NeurIPS 2013) and On the Efficiency of Sinkhorn-Knopp for
  Entropically Regularized Optimal Transport, arXiv:2604.03787.

**Rank fusion (§5)**
- Cormack, G. V., Clarke, C. L. A., Büttcher, S. Reciprocal rank fusion outperforms Condorcet
  and individual rank learning methods. *SIGIR* (2009).

**Functional alignment across subjects (§6)**
- Haxby, J. V. et al. Hyperalignment: Modeling shared information encoded in idiosyncratic
  cortical topographies. *eLife* (2020). PMC7266639.
- Hybrid hyperalignment. PMID 33762217 / PMC8273921.
- Structured Visual Target Learning for Cross-Subject EEG-to-Image Retrieval (2026).
  arXiv:2609.36971.

**Banded ridge (§7)**
- Dupré la Tour, T., Eickenberg, M., Nunez-Elizalde, A. O., Gallant, J. L. Feature-space
  selection with banded ridge regression. *NeuroImage* (2022). PMC9807218.
- Nunez-Elizalde, A. O. et al. Voxelwise encoding models with non-spherical multivariate
  normal priors. *NeuroImage* (2019).

**Augmentation and trial averaging (§8)**
- Bishop, C. M. Training with noise is equivalent to Tikhonov regularization. *Neural
  Computation* 7(1) (1995).
- Trial Averaging for Deep EEG Classification (2023). bioRxiv 2023.02.09.527905. — averaging
  helps substantially; outperformed Mixup.

**Benchmarks (§9)**
- Song, Y., Liu, B., Li, X., Shi, N., Wang, Y., Gao, X. Decoding Natural Images from EEG for
  Object Recognition. *ICLR* 2024. arXiv:2308.13234. — NICE; 15.6% top-1 / 42.8% top-5,
  200-way zero-shot.
- Gifford, A. T. et al. A large and rich EEG dataset for modeling human visual object
  recognition. *NeuroImage* 264:119754 (2022). PMC9771828. — THINGS-EEG2.
- NeuroCLIP: Brain-Inspired Prompt Tuning for EEG-to-Image Multimodal Contrastive Learning.
  arXiv:2511.09250. — 63.2% top-1, 200-way.

**A note on sourcing.** Claims attributed to specific papers above were checked against the
abstract or full text. The two load-bearing citations — Cichy et al. 2016 for hierarchical
correspondence and Song et al. 2024 for the benchmark comparison — were verified directly.
Entries cited for general context (e.g. the 2025–2026 arXiv preprints) were read only at
abstract level and are marked by their role: they establish that a problem is active, not
that a specific number is correct.
