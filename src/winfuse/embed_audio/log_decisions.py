"""Log Agent V-B's material design decisions. Idempotent; safe to re-run.

    .venv-embed/bin/python -m winfuse.embed_audio.log_decisions
"""
from __future__ import annotations

from winfuse.common.decisions import log_decision
from . import constants as K

DECS = [
    dict(
        title="Audio model selection",
        choice="Run wav2vec2-base (speech), Whisper-tiny AND Whisper-base encoders "
               "(speech, two capacities), plus BOTH general-sound models CLAP "
               "(laion/clap-htsat-unfused) and AST (AudioSet). All loaded and embedded "
               "20 videos cleanly on MPS in 16 GB, so none had to be dropped.",
        alternatives=["pick only one general-sound model to save time/RAM: unnecessary, "
                      "both fit; CLAP gives a shared text-audio space, AST gives AudioSet "
                      "class structure — they capture different things",
                      "skip Whisper-base, keep only tiny: base is cheap enough and lets "
                      "the latency/capacity question be asked"],
        rationale="The stimuli are heterogeneous (speech, music, ambience). A speech model "
                  "and two general-sound models cover the space; running both Whisper sizes "
                  "costs little and probes representation capacity.",
        consequences="Five neural audio models to maintain and checksum. More cache, but "
                     "each is small (<1 GB).",
    ),
    dict(
        title="wav2vec2 layer-wise extraction",
        choice="output_hidden_states=True exposes all 13 hidden states (embeddings + 12 "
               "transformer layers); each is mean-pooled over time into layers['L00'..'L12']. "
               "pooled = mean of the last hidden state; timeseries = last hidden state "
               "sub-sampled to the 12-step 0.25 s grid.",
        alternatives=["keep only the last layer: loses the known fact that mid layers of "
                      "wav2vec2 carry the most transferable phonetic/acoustic information",
                      "store full per-layer sequences: ~13x the cache for little gain since "
                      "V-C pools anyway"],
        rationale="Layer-wise mean-pooling lets V-C test which depth aligns best with EEG "
                  "without re-running the model, at modest cache cost.",
        consequences="15 cache variants for wav2vec2 (pooled, timeseries, 13 layers).",
    ),
    dict(
        title="Whisper encoder-only, no decoding",
        choice="Instantiate WhisperModel and keep only .encoder; run it on the model's own "
               "log-mel features. Mean-pool over the encoder time axis for pooled; "
               "sub-sample to the 0.25 s grid for timeseries. The decoder is never built.",
        alternatives=["full Whisper + transcription then embed text: adds an LM, is slow, "
                      "and conflates audio with Whisper's language prior",
                      "use a fixed encoder layer other than the last: last is the standard "
                      "representation; layer sweep deferred to wav2vec2 where it is cheap"],
        rationale="We want an acoustic representation, not a transcript. The encoder's last "
                  "hidden state is the natural clip embedding.",
        consequences="No transcript is available; that is intentional and avoids a text LM.",
    ),
    dict(
        title="General-sound models: CLAP and AST both kept; CLAP resampled to 48 kHz",
        choice="CLAP audio tower via get_audio_features (512-d shared-space vector, pooled "
               "only). CLAP's feature extractor requires 48 kHz and rejects 16 kHz, so the "
               "shared 16 kHz waveform is resampled up to 48 kHz with librosa before CLAP; "
               "this is a pure resample of the identical shared waveform. AST uses the shared "
               "16 kHz waveform directly and emits pooled + time-resolved.",
        alternatives=["feed CLAP the 16 kHz waveform: hard error from its feature extractor",
                      "re-decode the video at 48 kHz for CLAP: would bypass the frozen shared "
                      "extractor and risk CLAP seeing different content than other models"],
        rationale="Resampling the shared waveform keeps every model on the same audio content "
                  "while satisfying CLAP's trained sample rate.",
        consequences="CLAP has one extra resample step; its input is band-limited to 8 kHz "
                     "(the 16 kHz Nyquist) even though fed at 48 kHz, which is a documented "
                     "limitation of reusing the shared 16 kHz waveform.",
    ),
    dict(
        title="Handcrafted feature set (much richer than Phase 1's 4 scalars)",
        choice="hc_audio (217-d pooled): RMS, mel summary (per-band mean/std, n_mels=40), "
               "MFCC + delta + delta-delta mean/std (n_mfcc=20), spectral centroid/bandwidth/"
               "rolloff/flatness/flux, ZCR, onset strength, tempo. hc_visual (37-d pooled): "
               "luminance, contrast, Canny edge density, Gabor energy, HSV (12-bin hue hist, "
               "saturation, value), Farneback optic-flow magnitude + 8-bin direction histogram, "
               "frame-difference motion energy, scene-cut count. hc_av = hc_audio ++ hc_visual. "
               "All with a 12-step 0.25 s time-resolved variant too.",
        alternatives=["keep Phase 1's 4 scalars: that representation was the diagnosed "
                      "bottleneck",
                      "one giant feature vector instead of three named sets: harder for V-C "
                      "to attribute signal to audio vs visual"],
        rationale="Phase 1's bottleneck was explicitly the crude 4-descriptor stimulus "
                  "representation; a broad, interpretable hand-built set directly tests whether "
                  "richer-but-still-handcrafted features help.",
        consequences="hc_* are interpretable and CPU-cheap but shallow vs the neural models; "
                     "the report compares them head-to-head with Phase 1's 4 scalars.",
    ),
    dict(
        title="Mel/MFCC/STFT parameters tuned to the 0.25 s grid",
        choice=f"n_fft={K.N_FFT} (~128 ms), hop_length={K.HOP_LENGTH} (0.125 s), "
               f"n_mels={K.N_MELS}, n_mfcc={K.N_MFCC}, at 16 kHz. Hop 0.125 s gives exactly "
               "2 STFT frames per 0.25 s bin, which average cleanly onto the 12-step grid.",
        alternatives=["librosa default hop 512 (32 ms): finer but misaligned with the 0.25 s "
                      "EEG grid and more frames to pool",
                      "n_fft 512: too short to resolve low-frequency structure in 3 s clips"],
        rationale="The 0.25 s grid is fixed by src.common.frames and the EEG windows; choosing "
                  "the hop as an integer divisor of 0.25 s makes the time-resolved pooling exact.",
        consequences="Time resolution of audio handcrafted features is 0.125 s before the "
                     "0.25 s regrid; fine for this task.",
    ),
    dict(
        title="Optic-flow parameters",
        choice="cv2.calcOpticalFlowFarneback(pyr_scale=0.5, levels=3, winsize=15, "
               "iterations=3, poly_n=5, poly_sigma=1.2) between consecutive timeresolved "
               "frames; magnitude mean and an 8-bin, magnitude-weighted direction histogram.",
        alternatives=["sparse Lucas-Kanade on corners: sparse, misses global motion in "
                      "textured scenes",
                      "deep optical flow (RAFT): a neural net, outside the handcrafted scope "
                      "and heavier"],
        rationale="Farneback is dense, deterministic, CPU-cheap and the OpenCV-standard choice; "
                  "direction histogram captures motion orientation that Phase 1's scalar "
                  "motion-energy discarded.",
        consequences="Flow is computed on 12 frames at 0.25 s spacing, so sub-0.25 s motion is "
                     "not resolved; adequate for a pooled descriptor.",
    ),
    dict(
        title="How silent / no-audio videos are embedded",
        choice=f"The {len(K.SILENT_IDS)} videos with no usable audio (34 no stream + 9 silent) "
               f"receive the shared extractor's zero waveform and are embedded normally. Every "
               f"audio model therefore returns a finite, constant-ish vector for them (verified "
               f"non-NaN). Their IDs are written to every audio manifest under silent_ids, with "
               f"silent_ids_test={K.SILENT_IDS_TEST} flagged because 2 "
               f"({K.NO_AUDIO_TEST_IDS}) are no-audio-stream test videos.",
        alternatives=["drop silent videos from the audio cache: breaks the 1..1102 row "
                      "alignment every other model and V-C rely on",
                      "emit NaN for silent videos: the cache writer rejects non-finite values "
                      "by contract"],
        rationale="Keeping full 1102-row coverage with a documented zero-waveform fallback is "
                  "the only option compatible with the frozen cache contract and with V-C's "
                  "row alignment.",
        consequences="Silent videos contribute no audio information; audio metrics are diluted "
                     "and this must be reported with every audio result. V-C can recompute "
                     "excluding silent_ids.",
    ),
    dict(
        title="MPS device-equivalence gate before every neural full run",
        choice="For each neural model, verify_device_equivalence runs the first 20 videos on "
               "CPU and MPS and requires per-video cosine >= 0.99 before the full extraction. "
               "All five passed with cos_min=1.00000. The verdict goes in each manifest; on "
               "failure the code falls back to CPU and logs a decision. Handcrafted descriptors "
               "are CPU numpy/librosa/cv2 and skip the check (recorded as applicable=false).",
        alternatives=["trust MPS without checking: violates the retained numerical-equivalence "
                      "gate for anything feeding reported statistics",
                      "run everything on CPU: slower for no benefit since the gate passes"],
        rationale="MPS is permitted but the equivalence gate is a correctness gate; passing it "
                  "lets us use MPS safely and record proof.",
        consequences="A ~20-video CPU+MPS probe per model up front; negligible vs the full run.",
    ),
    dict(
        title="Joint audio-visual via cache concatenation, not a joint network",
        choice="joint_av concatenates the best available visual pooled embedding (prefer a "
               "V-A model: dinov2_base > clip_vit_b32 > videomae_base > xclip > dinov2_small, "
               "else our hc_visual) with the best audio pooled (wav2vec2_base else hc_audio), "
               "each z-scored on training videos (id<=1000) only, visual columns first. Built "
               "from caches at the end of a full run; if a source is missing it logs and skips "
               "rather than blocking.",
        alternatives=["train a true audio-visual fusion model (ImageBind): not available/"
                      "verified pre-cutoff in this env and would not fit cleanly in 16 GB with "
                      "the rest; logged as not done",
                      "reimplement a V-A visual model here: out of scope and duplicative"],
        rationale="A z-scored concatenation is a strong, honest joint baseline that needs no "
                  "new training and respects the no-test-leakage rule by fitting the scaler on "
                  "train only. V-A runs in parallel so we must tolerate a missing visual cache.",
        consequences="joint_av depends on which caches exist when it runs; the manifest records "
                     "the exact sources, order and scaling so it is reproducible.",
    ),
    dict(
        title="ImageBind-style audio-visual model not run",
        choice="No ImageBind/true AV-fusion model is run. None is in the pinned, "
               "pre-2026-10-16 model list (cache/model_availability.json), and loading an "
               "unpinned one would break the provenance constraint and strain 16 GB alongside "
               "the five audio models.",
        alternatives=["pull ImageBind from its original repo: not a pinned HF revision here, "
                      "fails the recorded-revision/checksum constraint"],
        rationale="The joint requirement is satisfied by the z-scored concatenation; a bespoke "
                  "AV model is explicitly optional and gated on genuine availability.",
        consequences="No single-network AV embedding; joint_av (concatenation) stands in.",
    ),
]


def main() -> int:
    for d in DECS:
        log_decision(agent="V-B", tags=["phase2", "audio", "V-B"], **d)
    print(f"logged {len(DECS)} V-B decisions -> decisions/V-B.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
