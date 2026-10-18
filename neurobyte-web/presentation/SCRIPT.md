# NeuroBytes — 4-minute talk script

About 540 words, roughly 4 minutes at a steady pace (~135 words/min). Timings are cumulative.

## Slide 1 — See what the brain watched (0:00–0:40)

Hi, we're team NeuroBytes: Sushruth Shivaraju and Ayush Shetty. Our question was simple: can you tell which video someone watched, just from their brain activity? We take three seconds of EEG from 17 electrodes over the visual cortex and rank 102 candidate videos. The model never sees the video at test time, only the brain signal. Our final system, C13, puts the right video first 36% of the time and in the top five 67% of the time. Chance is under one percent, so that's about 37 times better than guessing.

## Slide 2 — Three seconds, five steps (0:40–1:35)

Here's how it works, in five steps. First, record: three seconds of EEG while a clip plays. Second, window: we only look at the first 700 milliseconds after the clip starts; more on why in a moment. Third, decode: we map the EEG onto features from deep video models like V-JEPA, CLIP and DINOv2, plus audio. Fourth, fuse: about 650 separate scorers each produce a ranking, and we combine them by rank, so they never need to be calibrated against each other. Fifth, balance: we remove "hub" videos that look like a match for everything, and use Sinkhorn so each video is picked roughly once. Every setting was chosen on held-out training videos. The test set was only used to measure the result.

## Slide 3 — From 11% to 35% top-1 (1:35–2:15)

This chart is the whole journey. Our first ridge baseline, J1, got 11% top-1. The winfuse pipeline, with 509 latency-window scorers, more than doubled that to 23%. Combining both pipelines took us to 26 and a half. Correcting for hubness pushed top-five past 62%. Cross-person alignment and Sinkhorn balancing reached 66.7% top-five. Finally, two contrastive neural networks trained across all six people gave C13: 36% top-one, 67% top-five. Every step stayed in only if it improved validation.

## Slide 4 — The signal is the onset response (2:15–3:05)

The most important finding is about timing. Here we decode from one time window at a time, and a lower bar means the right video ranks higher. The best window is 50 to 400 milliseconds after the clip starts. That is exactly where visual neuroscience places the P1 and N1 responses that encode objects and faces. After about 800 milliseconds, every window sits at chance, the line across the top. So most of a three-second recording says nothing about which video is on screen. Our biggest gain didn't come from a bigger model. It came from knowing when to look.

## Slide 5 — What the brain signal can and cannot tell us (3:05–4:00)

So why does it work? Four lessons. One: the decodable signal is the onset response. Two: the structure of the task matters. Removing hubs with CSLS added more than four points of top-one, and Sinkhorn balancing helped further, without touching the EEG at all. Three: you can learn from other people's brains, but only after aligning them. Every head mixes the signals differently, and pooling without alignment actually hurts. Four: it's robust. Rebuilding from scratch is bit-identical, and new random seeds stay within one standard error. The misses are mostly noise, not hard videos; the right answer is usually in the top five. With more repeats per clip and larger pretrained EEG models, we think this can go further. Thank you.
