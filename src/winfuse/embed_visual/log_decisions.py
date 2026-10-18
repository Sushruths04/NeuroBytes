"""Log Agent V-A's material design decisions. Idempotent; safe to re-run.

    .venv-embed/bin/python -m winfuse.embed_visual.log_decisions
"""
from __future__ import annotations

from winfuse.common.decisions import log_decision

DECISIONS = [
    dict(
        title="Model roster for visual embeddings",
        choice="Seven models spanning four inductive biases: AlexNet (all layers) and "
               "ResNet-50 (ImageNet-supervised CNNs), CLIP ViT-B/32 (language-aligned "
               "ViT), DINOv2 small+base (self-supervised ViT), VideoMAE-base and "
               "X-CLIP-base-patch32 (temporal video models). V-JEPA skipped.",
        alternatives=[
            "AlexNet only (match baseline): rejected, the whole point of Phase 2 is a "
            "richer stimulus space than the baseline's single conv layer.",
            "Only CLIP/DINOv2 (semantic ViTs): rejected, we also want the CNN ladder for a "
            "depth sweep and video models for temporal content.",
            "Add V-JEPA: skipped — see its own decision (16 GB / availability).",
        ],
        rationale="Each family tests a different hypothesis about what in the stimulus the "
                  "EEG tracks: low/mid vision (CNN layers), semantics (CLIP), "
                  "self-supervised structure (DINOv2), motion/time (VideoMAE, X-CLIP). "
                  "Spanning them lets V-C attribute any EEG alignment to a capability "
                  "rather than to one arbitrary network.",
        consequences="Seven models x 1102 videos x (pooled+timeresolved+layers) is the bulk "
                     "of the compute budget; mitigated by MPS and resume-safe caching.",
    ),
    dict(
        title="Frame sampling: use the frozen shared sampler, never a local one",
        choice="All visual models consume src.common.frames variants `pooled` (8 frames at "
               "bin centres) and `timeresolved` (12 frames on a 0.25 s grid). No model "
               "rolls its own sampling.",
        alternatives=[
            "Per-model native frame counts sampled independently: rejected, it would "
            "confound model differences with frame-set differences.",
            "Uniform linspace index sampling (baseline variant): rejected for the proper "
            "models because fps is non-constant (15.0-30.5), so index spacing drifts from "
            "time spacing by up to 87 ms — material for the latency analysis.",
        ],
        rationale="Identical frames across models is a precondition for comparing them; the "
                  "sampler addresses frames by timestamp via the measured effective fps.",
        consequences="Video models that want more frames must resample from these fixed sets "
                     "(see the 16-frame mapping decision).",
    ),
    dict(
        title="268->224 resize is a DOWNsample; use each model's native processor",
        choice="Feed 268x268 RGB uint8 frames to each model's own preprocessing (torchvision "
               "weight.transforms() or the HF image processor), which resizes to 224 "
               "(bilinear/bicubic + antialias) and applies the model's normalisation.",
        alternatives=[
            "Hand-write one resize for all models: rejected, each model was trained with its "
            "own resize/crop/normalise and must see matching inputs.",
            "Upsample to the model's expected size: not applicable — 268>224, so this is a "
            "downsample, and forcing an upscale would invent detail.",
        ],
        rationale="Matching each model's training-time preprocessing is what makes the "
                  "pretrained features valid; the shared 268 input guarantees every model "
                  "starts from the same pixels.",
        consequences="Minor per-model differences in resize interpolation are inherent to "
                     "using pretrained models and are documented per manifest.",
    ),
    dict(
        title="AlexNet: expose all layers (conv1-5, fc6, fc7); top-level feature = fc7",
        choice="`alexnet_layers` taps the ReLU after each conv (features indices 2,5,7,9,12) "
               "global-average-pooled, plus fc6/fc7 (classifier indices 2,5). The pooled and "
               "timeseries outputs use fc7.",
        alternatives=[
            "Only features.3 like the baseline: rejected, baseline_alexnet already owns that; "
            "the proper version must enable a depth sweep.",
            "Keep conv spatial maps un-pooled: rejected, would blow dimensionality to tens of "
            "thousands and break the per-column comparison; global-average-pool is the "
            "standard CNN feature readout.",
        ],
        rationale="A layer sweep tests whether EEG aligns with early (edges/texture) or late "
                  "(object) AlexNet representations, which the single baseline layer cannot.",
        consequences="fc6/fc7 are 4096-d; effective dim reported via PCA keeps the usable "
                     "size honest.",
    ),
    dict(
        title="DINOv2 pooling: expose BOTH CLS and mean-patch per block",
        choice="For DINOv2 small and base, cache per-block CLS and per-block mean-of-patch "
               "tokens, and set the pooled/timeseries output to concat(last-layer CLS, "
               "last-layer mean-patch).",
        alternatives=[
            "CLS only: rejected, DINOv2's patch tokens carry strong dense/layout structure "
            "that CLS alone discards.",
            "mean-patch only: rejected, loses the global summary CLS provides.",
        ],
        rationale="DINOv2 is self-supervised; we do not know a priori which readout aligns "
                  "with EEG, so we expose both and let V-C decide empirically.",
        consequences="Doubles the DINOv2 layer count (24 layer arrays each); cheap on disk.",
    ),
    dict(
        title="CLIP readout: projected image embedding + per-block CLS",
        choice="`clip_vit_b32` pooled output is the 512-d projected image embedding "
               "(visual_projection of the post-LN CLS), the vector in CLIP's shared "
               "image/text space. Per-block CLS states (768-d x 12) are exposed as layers.",
        alternatives=[
            "Raw last_hidden_state mean: rejected, the projected CLS is the semantically "
            "meaningful CLIP vector and what downstream retrieval conventionally uses.",
            "Mean-patch like DINOv2: rejected, CLIP is trained to concentrate the summary in "
            "CLS via the projection, so CLS is the natural readout.",
        ],
        rationale="The projected embedding is CLIP's intended output; per-block CLS enables a "
                  "ViT depth sweep analogous to AlexNet's.",
        consequences="Pooled dim is only 512 (post-projection) — compact and semantic.",
    ),
    dict(
        title="VideoMAE 16-frame mapping: nearest-neighbour time resample, no pixel blending",
        choice="VideoMAE-base expects 16 frames. Map our 8 (pooled) or window frames onto 16 "
               "by idx=round(linspace(0, n_in-1, 16)) — repeating real decoded frames.",
        alternatives=[
            "Pixel-space interpolation between frames: rejected, it synthesises motion-blurred "
            "frames the model never trained on and that the EEG never saw.",
            "Zero/black pad to 16: rejected, injects fake dark frames that corrupt the "
            "temporal statistics VideoMAE models.",
        ],
        rationale="Repeating real frames preserves true temporal order and content; the model "
                  "simply sees some frames twice, which is benign for a masked-autoencoder "
                  "video representation.",
        consequences="Effective temporal resolution of the pooled VideoMAE vector is 8 unique "
                     "frames presented as 16; documented in the manifest.",
    ),
    dict(
        title="X-CLIP uses 8 frames natively; manual MIT path around a transformers bug",
        choice="X-CLIP-base-patch32's MIT position embedding is fixed at 8 frames, so we feed "
               "8 (1:1 with the pooled variant). The installed transformers' "
               "get_video_features assumes the MIT returns a dataclass but it returns a "
               "tuple, so we replicate the path (vision_model -> visual_projection -> MIT "
               "pooler) and read mit_output[1].",
        alternatives=[
            "Resample X-CLIP to 16 like VideoMAE: rejected, it raises a shape error — the "
            "model only accepts 8.",
            "Call get_video_features directly: rejected, it crashes on this transformers "
            "version (AttributeError on pooler_output).",
        ],
        rationale="8 frames is the model's contract; the manual path yields the exact same "
                  "MIT-integrated video embedding without the broken helper.",
        consequences="Pooled X-CLIP output is the 512-d MIT video embedding; a pre-MIT "
                     "per-frame CLS mean is also exposed as a layer for comparison.",
    ),
    dict(
        title="Video-model timeseries: sliding 6-frame windows over the 0.25 s grid",
        choice="Because a video model emits one vector per clip, the timeseries for VideoMAE "
               "and X-CLIP is built by sliding a 6-frame window centred on each of the 12 "
               "timeresolved grid points (edge-padded by repetition), resampled to the "
               "model's frame count, giving (12, D).",
        alternatives=[
            "Single clip only, no timeseries for video models: rejected, V-C's latency "
            "analysis needs a per-time readout from every model.",
            "1-frame windows: rejected, defeats the purpose of a temporal model.",
        ],
        rationale="A short centred window gives each grid point a genuine motion-aware vector "
                  "while keeping alignment to the EEG's 0.25 s windows.",
        consequences="Each timeseries point summarises ~1.25 s of context, not an instant; "
                     "stated in the module docstring and manifest.",
    ),
    dict(
        title="Device: MPS by default with the 0.99 cosine gate; CPU fallback on failure",
        choice="Run every model on MPS, but first verify_device_equivalence on 20 videos "
               "(per-video cosine>=0.99 MPS vs CPU). Record the verdict in the manifest. On "
               "failure, fall back to CPU and log it. Wrap extraction in MpsFallbackRecorder.",
        alternatives=[
            "CPU everywhere: rejected, user relaxed the constraint and MPS is 1.7-7.6x "
            "faster here.",
            "MPS with no gate: rejected, the frozen contract keeps the correctness gate even "
            "though the permission gate is lifted.",
        ],
        rationale="MPS is permitted but reported embeddings must be numerically faithful; the "
                  "gate bounds any float32-GPU drift at cosine<0.01.",
        consequences="All seven models passed at cosine_min=1.00000 on the 20-video check, so "
                     "none fell back for numerical reasons.",
    ),
    dict(
        title="Caching, batching and resume safety",
        choice="One model at a time, frames processed per video (<=12 frames/batch to the "
               "GPU), each video written atomically to cache/embeddings/<model>/_parts via "
               "embed_cache.save_part. ec.pending drives resume; a crash loses at most one "
               "video. float32 on disk.",
        alternatives=[
            "In-memory accumulation then one write: rejected, a crash would lose the whole "
            "run and risk >10 GB RAM.",
            "Large multi-video GPU batches: rejected, 16 GB budget and the per-video cache "
            "granularity is what makes resume cheap.",
        ],
        rationale="Per-video atomic parts give both the <10 GB memory bound and free resume.",
        consequences="Many small files during a run; consolidate() stacks them into one array "
                     "per variant at the end.",
    ),
    dict(
        title="V-JEPA skipped",
        choice="V-JEPA is not included in this phase's roster.",
        alternatives=[
            "Include V-JEPA-base: deferred — the two video models already cover the temporal "
            "hypothesis, and V-JEPA's larger memory footprint and extra extraction time are "
            "not justified before V-C shows video models matter at all.",
        ],
        rationale="The brief marks V-JEPA optional ('only if it fits 16 GB'); VideoMAE and "
                  "X-CLIP already test the temporal-model hypothesis within budget.",
        consequences="If V-C finds video models beat the AlexNet baseline, V-JEPA becomes a "
                     "worthwhile follow-up; recorded so the omission is deliberate.",
    ),
    dict(
        title="Effective dimensionality fitted on TRAIN videos only",
        choice="Report PCA components for 95% variance via ec.effective_dim with the train "
               "mask (IDs 1..1000), never fitting on the 102 test videos.",
        alternatives=[
            "Fit PCA on all 1102: rejected, that fits a (reported) object on test videos, "
            "which the no-leakage constraint forbids for anything fitted.",
        ],
        rationale="Even though test videos carry no labels, fitting the dimensionality "
                  "estimate only on train keeps the reported statistic leakage-clean and "
                  "consistent with Phase 1's rule.",
        consequences="Effective dim is a train-set property; test embeddings are projected "
                     "with the same train-fitted basis downstream by V-C if needed.",
    ),
]


def main() -> int:
    for d in DECISIONS:
        log_decision(agent="V-A", tags=["phase2", "embed_visual"], **d)
    print(f"logged {len(DECISIONS)} decisions to decisions/V-A.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
