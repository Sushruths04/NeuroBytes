# Video captions (display only)

`captions.json` holds one generated English sentence for each of the 102 test videos (IDs 1001–1102), as `{"<id>": "<caption>"}`. The web UI (`neurobyte-web/`) shows them next to the predicted and top-ranked videos so the results are readable without opening every clip.

**The captions are not used by C13.** No code in `combo/`, `src/` or `configs/` reads them, and they play no part in any score or ranking.

## How they were made

We generated them ourselves from the official challenge videos. No external annotations, no other dataset and no labels were used.

| | |
|---|---|
| Model | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), revision `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Input | the official mp4 of each clip (`stimuli_test/<id>.mp4`): 8 evenly spaced frames, resized to 448 px wide |
| Prompt | "Describe this 3-second video clip in one sentence: the scene, the main people or objects, and what they are doing." |
| Decoding | greedy (`do_sample=False`), at most 48 new tokens |
| Run | 17 Oct 2026, one Modal L4 GPU; torch 2.6.0, transformers 4.53.2 |
| Script | [`modal_audio_caption.py`](modal_audio_caption.py) (copied unchanged from the team's working repo; its paths refer to that layout) |

The script captioned all 1102 official clips (training and test) in one pass and also computed CLAP audio features; only the 102 test-video captions are kept here, because the web UI shows only test videos. The captions are model output and can be wrong or imprecise.

## Using them

The web UI finds this file automatically (natively and in the `neurobytes-web` Docker image). A `captions.json` placed in the challenge data folder, or `NEUROBYTES_CAPTIONS=/path/to/captions.json`, takes precedence.
