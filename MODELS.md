# Pretrained models used by C13 (exact names and versions)

The challenge requires the exact name and version of every public pretrained model; their weights are not redistributed. All models are used **frozen** to embed the stimulus videos (frames or audio). The EEG networks we train ourselves are released as weights (release `v1.0`).

## Our feature bank (`outputs/mindread_features/*.npz`; each file stores `model` and `revision`)

| Feature file | Model | Version / revision | Used in C13 by |
|---|---|---|---|
| `vjepa2_g_l30` | `facebook/vjepa2-vitg-fpc64-256` (layer 30) | `875c192b7b704b87d1e1d99345769632dd5f739a` | ours77, T2, networks |
| `vjepa2_l_l12_18_24` | `facebook/vjepa2-vitl-fpc64-256` (layers 12, 18, 24) | HF `main` @ 2026-10-17 | ours77 |
| `s3d_k400` | `torchvision.models.video.s3d` | weights `KINETICS400_V1` (torchvision 0.19.1) | ours77, T2 |
| `clap_audio` | `laion/larger_clap_general` | `ada0c23a3…` (full hash in file) | ours77, T2 |
| `siglip_b16` | `google/siglip-base-patch16-224` | HF `main` | ours77 |
| `clip_l14` | OpenCLIP `ViT-L-14` | pretrained tag `openai` (open_clip 2.29.0) | networks |
| `dinov2_b_l4_8_12` | `facebook/dinov2-base` (layers 4, 8, 12) | HF `main` | networks, T2 |
| `lowlevel_flat` | own luminance/contrast/motion features (no pretrained model) | v1 | T2 |

## winfuse embedding banks (`cache/embeddings/*/manifest.json`; 509 aligned scorers)

| Bank | Model | Revision |
|---|---|---|
| `alexnet_layers` | `torchvision.models.alexnet`, `AlexNet_Weights.IMAGENET1K_V1` | torchvision 0.29.1 |
| `resnet50` | `torchvision.models.resnet50`, `ResNet50_Weights.IMAGENET1K_V2` | torchvision 0.29.1 |
| `clip_vit_b32` | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f…` |
| `dinov2_base` | `facebook/dinov2-base` | `f9e44c814b77203eaa57a6bdbbd535…` |
| `dinov2_small` | `facebook/dinov2-small` | `ed25f3a31f01632728cabb09d1542f…` |
| `videomae_base` | `MCG-NJU/videomae-base` | `dc740ceda42fce44faed2ea03c6d44…` |
| `xclip_base` | `microsoft/xclip-base-patch32` | `a2e27a78a2b5d802e894b8a1ef14f3…` |
| `clap_audio` | `laion/clap-htsat-unfused` | `8fa0f1c6d0433df6e97c127f64b2a1…` |
| `wav2vec2_base` | `facebook/wav2vec2-base-960h` | `22aad52d435eb6dbaf354bdad9b0da…` |
| `whisper_base` / `whisper_tiny` | `openai/whisper-base` / `openai/whisper-tiny` | `e37978b9…` / `169d4a43…` |
| `ast_audioset` | `MIT/ast-finetuned-audioset-10-10-0.4593` | revision not recorded in the original manifest |
| `hc_visual`, `hc_audio`, `hc_av`, `joint_av`, `phase1_descriptors` | hand-crafted (OpenCV 4.10, librosa 0.10.2) / concatenations; no pretrained model | handcrafted-v1 |

Recorded full hashes are in each bank's `manifest.json` (shipped in release `v1.0`). Some revisions above remain unpinned or incomplete; the release is not a complete version audit. The AST revision was not recorded.

## Trained by us (weights released)

- **Contrastive EEG networks h6 and h10** (5 seeds each): `combo/neural_clip.py`, release `v1.0`.
- **Ridge scorers (winfuse 509, ours77) and T2:** deterministic closed-form fits, rebuilt by the pipeline; aligned winfuse can take about an hour per person (see `PIPELINE.md`).
