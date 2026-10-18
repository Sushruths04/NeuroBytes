"""Audio (CLAP) features of the official clips on Modal.

    modal run scripts/modal_audio.py --limit 4     # smoke test, nothing saved
    modal run scripts/modal_audio.py               # all 1102 official clips
Output (../outputs/mindread_features/):
    clap_audio.npz     ids, feats (N, 512)  CLAP audio projection; silent clips get the embedding of silence
The volume `mindread-videos` is deleted at the end; verify with `modal volume list` and `modal app list`.
"""
import glob, io, json, os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'challenge_official', 'eeg-mind-reading-challenge'))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'outputs', 'mindread_features'))
CLAP = 'laion/larger_clap_general'

app = modal.App('mindread-audio')
vol = modal.Volume.from_name('mindread-videos', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.6.0', 'transformers==4.53.2', 'av==13.1.0', 'numpy<2.1'))


@app.function(image=image, gpu='L4', volumes={'/videos': vol}, timeout=3600)
def extract(limit: int = 0):
    import av, numpy as np, torch
    from huggingface_hub import HfApi
    from transformers import ClapModel, ClapProcessor

    revs = {CLAP: HfApi().model_info(CLAP).sha}
    print('revisions', revs, flush=True)
    paths = sorted(glob.glob('/videos/*.mp4'))
    if limit:
        paths = paths[:limit]

    def read(p):
        c = av.open(p, metadata_errors='ignore'); audio = None
        if c.streams.audio:
            rs = av.AudioResampler(format='flt', layout='mono', rate=48000)
            chunks = [r.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for r in rs.resample(fr)]
            chunks += [r.to_ndarray().reshape(-1) for r in rs.resample(None)]
            audio = np.concatenate(chunks).astype(np.float32) if chunks else None
        c.close()
        return audio

    cp = ClapProcessor.from_pretrained(CLAP, revision=revs[CLAP])
    cm = ClapModel.from_pretrained(CLAP, revision=revs[CLAP]).eval().cuda()
    ids, clap = [], []
    for i, p in enumerate(paths):
        vid = int(os.path.basename(p).split('.')[0])
        audio = read(p)
        a = audio if audio is not None and len(audio) else np.zeros(48000 * 3, np.float32)
        x = cp(audios=[a], sampling_rate=48000, return_tensors='pt').to('cuda')
        with torch.no_grad():
            clap.append(cm.get_audio_features(**x)[0].float().cpu().numpy())
        ids.append(vid)
        if i % 50 == 0 or limit:
            print(i, vid, 'audio' if audio is not None else 'silent', flush=True)
    buf = io.BytesIO()
    np.savez(buf, ids=np.array(ids), feats=np.stack(clap).astype(np.float32), model=CLAP, revision=json.dumps(revs))
    return buf.getvalue()


@app.local_entrypoint()
def main(limit: int = 0):
    paths = sorted(glob.glob(os.path.join(ROOT, 'stimuli_train', '*.mp4')) + glob.glob(os.path.join(ROOT, 'stimuli_test', '*.mp4')))
    print('uploading', len(paths), 'clips')
    with vol.batch_upload(force=True) as b:
        for p in paths:
            b.put_file(p, '/' + os.path.basename(p))
    try:
        res = extract.remote(limit)
        if limit:
            return
        os.makedirs(OUT, exist_ok=True)
        open(os.path.join(OUT, 'clap_audio.npz'), 'wb').write(res)
        print('saved to', OUT)
    finally:
        import subprocess
        subprocess.run(['modal', 'volume', 'delete', 'mindread-videos', '--yes'], check=False)
        print('volume mindread-videos delete requested; verify with `modal volume list`')
