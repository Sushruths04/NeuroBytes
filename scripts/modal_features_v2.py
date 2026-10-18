"""T2 feature bank v2 on Modal (17 Oct 2026, user-approved GPU use): layer-wise features for a layer sweep.

    modal run scripts/modal_features_v2.py --limit 3        # smoke test, nothing saved
    modal run scripts/modal_features_v2.py                  # all 1102 official clips, 4 GPU jobs in parallel
Each output ../outputs/mindread_features/layers_<name>.npz holds ids, layers (N, L, D) = per-layer mean tokens,
averaged over frames (or time for audio), plus model, revision and layer indices. Bands are derived locally.
The volume `mindread-videos` is deleted at the end; verify with `modal volume list` and `modal app list`.
"""
import glob, io, json, os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'challenge_official', 'eeg-mind-reading-challenge'))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'outputs', 'mindread_features'))
JOBS = {
    'dinov2_l':  dict(model='facebook/dinov2-large', kind='image', layers=[4, 8, 12, 16, 20, 24]),
    'dinov2_g':  dict(model='facebook/dinov2-giant', kind='image', layers=[8, 16, 24, 32, 40]),
    'w2vbert':   dict(model='facebook/w2v-bert-2.0', kind='audio', layers=[4, 8, 12, 16, 20, 24]),
    'vjepa2_g':  dict(model='facebook/vjepa2-vitg-fpc64-256', kind='video', layers=[10, 20, 30, 40]),
}
app = modal.App('mindread-features-v2')
vol = modal.Volume.from_name('mindread-videos', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.6.0', 'torchvision==0.21.0', 'transformers==4.53.2', 'accelerate==1.8.1',
                      'av==13.1.0', 'numpy<2.1', 'pillow'))


@app.function(image=image, gpu='L4', volumes={'/videos': vol}, timeout=5400)
def extract(name: str, limit: int = 0):
    import av, numpy as np, torch
    from huggingface_hub import HfApi
    job = JOBS[name]; rev = HfApi().model_info(job['model']).sha
    paths = sorted(glob.glob('/videos/*.mp4'))[: limit or None]
    ids, feats = [], []
    if job['kind'] == 'image':
        from transformers import AutoImageProcessor, AutoModel
        p = AutoImageProcessor.from_pretrained(job['model'], revision=rev)
        m = AutoModel.from_pretrained(job['model'], revision=rev, torch_dtype=torch.float16).eval().cuda()
    elif job['kind'] == 'video':
        from transformers import AutoModel, AutoVideoProcessor
        p = AutoVideoProcessor.from_pretrained(job['model'], revision=rev)
        m = AutoModel.from_pretrained(job['model'], revision=rev, torch_dtype=torch.float16).eval().cuda()
    else:
        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
        p = AutoFeatureExtractor.from_pretrained(job['model'], revision=rev)
        m = Wav2Vec2BertModel.from_pretrained(job['model'], revision=rev, torch_dtype=torch.float16).eval().cuda()
    for i, path in enumerate(paths):
        c = av.open(path, metadata_errors='ignore')
        with torch.no_grad():
            if job['kind'] in ('image', 'video'):
                F = [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
                if job['kind'] == 'image':
                    F = [F[j] for j in np.linspace(0, len(F) - 1, 8).astype(int)]            # same 8 frames as v1
                    x = p(images=F, return_tensors='pt')['pixel_values'].to('cuda', torch.float16)
                    hs = m(pixel_values=x, output_hidden_states=True).hidden_states
                    z = torch.stack([hs[l].mean(1).mean(0) for l in job['layers']])         # (L, D): tokens, then frames
                else:
                    F = np.stack([F[j] for j in np.linspace(0, len(F) - 1, 64).astype(int)])
                    x = p(list(F), return_tensors='pt')['pixel_values_videos'].to('cuda', torch.float16)
                    hs = m(pixel_values_videos=x, output_hidden_states=True, skip_predictor=True).hidden_states
                    z = torch.stack([hs[l].mean(1)[0] for l in job['layers']])
            else:
                a = None
                if c.streams.audio:
                    rs = av.AudioResampler(format='flt', layout='mono', rate=16000)
                    ch = [r.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for r in rs.resample(fr)]
                    ch += [r.to_ndarray().reshape(-1) for r in rs.resample(None)]
                    a = np.concatenate(ch) if ch else None
                a = a if a is not None and len(a) > 1600 else np.zeros(48000, np.float32)    # silent clip -> silence
                x = p(a, sampling_rate=16000, return_tensors='pt').to('cuda')
                x['input_features'] = x['input_features'].half()
                hs = m(**x, output_hidden_states=True).hidden_states
                z = torch.stack([hs[l].mean(1)[0] for l in job['layers']])
        c.close()
        feats.append(z.float().cpu().numpy()); ids.append(int(os.path.basename(path).split('.')[0]))
        if i % 100 == 0 or limit:
            print(name, i, feats[-1].shape, flush=True)
    buf = io.BytesIO()
    np.savez(buf, ids=np.array(ids), layers=np.stack(feats).astype(np.float32), layer_index=np.array(job['layers']),
             model=job['model'], revision=f"{rev}; per-layer mean tokens; image: 8 frames mean; video: 64 frames; audio: 16 kHz time mean")
    return name, buf.getvalue()


@app.local_entrypoint()
def main(limit: int = 0, only: str = ''):
    names = [n for n in JOBS if not only or n in only.split(',')]
    paths = sorted(glob.glob(os.path.join(ROOT, 'stimuli_train', '*.mp4')) + glob.glob(os.path.join(ROOT, 'stimuli_test', '*.mp4')))
    with vol.batch_upload(force=True) as b:
        for p in paths:
            b.put_file(p, '/' + os.path.basename(p))
    try:
        for res in extract.starmap([(n, limit) for n in names], return_exceptions=True):
            if isinstance(res, Exception):
                print('FAILED', repr(res)[:300]); continue
            name, data = res
            if limit:
                print('smoke ok', name, len(data)); continue
            open(os.path.join(OUT, f'layers_{name}.npz'), 'wb').write(data); print('saved', name)
    finally:
        import subprocess
        subprocess.run(['modal', 'volume', 'delete', 'mindread-videos', '--yes'], check=False)
