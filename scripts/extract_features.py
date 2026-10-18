"""Extract frozen-network features from the OFFICIAL videos on the local GPU (videos never leave the laptop).

    .venv/bin/python scripts/extract_features.py alexnet_f3 siglip_b16
Output: outputs/mindread_features/<name>.npz (see mindread.features). Exact checkpoints are listed in
configs/models.lock.yaml (and stored in each npz).
"""
import glob, os, sys, time

import av
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mindread import features as fs  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'challenge_official', 'eeg-mind-reading-challenge'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMNET_MEAN, IMNET_STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def videos():
    return sorted(glob.glob(os.path.join(ROOT, 'stimuli_train', '*.mp4')) + glob.glob(os.path.join(ROOT, 'stimuli_test', '*.mp4')))


def frames(path, n=8):
    c = av.open(path, metadata_errors='ignore')
    F = [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
    c.close()
    return np.stack([F[i] for i in fs.frame_picks(len(F), n)])  # (n, H, W, 3) uint8


def to_tensor(F, size):
    x = torch.from_numpy(F).permute(0, 3, 1, 2).float() / 255.0
    return torch.nn.functional.interpolate(x, size=(size, size), mode='bilinear', antialias=True, align_corners=False)


class AlexNetF3:
    """Organizer baseline: torchvision AlexNet (ImageNet) module features.3 (conv2, pre-ReLU), mean over 8 frames."""
    model, revision = 'torchvision/alexnet', 'IMAGENET1K_V1'

    def __init__(self):
        import torchvision
        self.m = torchvision.models.alexnet(weights='IMAGENET1K_V1').features[:4].eval().to(DEV)

    @torch.no_grad()
    def __call__(self, F):
        x = (to_tensor(F, 224) - IMNET_MEAN) / IMNET_STD
        z = self.m(x.to(DEV)).flatten(1).cpu().numpy()
        return z.mean(0), None


class SigLIP:
    model, revision = 'google/siglip-base-patch16-224', 'main'

    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel
        self.p = AutoImageProcessor.from_pretrained(self.model)
        self.m = AutoModel.from_pretrained(self.model, torch_dtype=(torch.float16 if DEV == 'cuda' else torch.float32)).eval().to(DEV)

    @torch.no_grad()
    def __call__(self, F):
        x = self.p(images=list(F), return_tensors='pt')['pixel_values'].to(DEV, next(self.m.parameters()).dtype)
        z = self.m.get_image_features(pixel_values=x)
        if not torch.is_tensor(z):  # transformers >= 5 returns an output object
            z = z.pooler_output
        z = z.float().cpu().numpy()
        return z.mean(0), z


class OpenCLIP:
    def __init__(self, arch, tag):
        import open_clip
        self.model, self.revision = f'open_clip/{arch}', tag
        self.m, _, self.pre = open_clip.create_model_and_transforms(arch, pretrained=tag)
        self.m = self.m.eval().to(DEV)
        self.mean = torch.tensor(self.m.visual.image_mean or (0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
        self.std = torch.tensor(self.m.visual.image_std or (0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, F):
        x = (to_tensor(F, 224) - self.mean) / self.std
        z = self.m.encode_image(x.to(DEV)).float().cpu().numpy()
        return z.mean(0), z


class HFHidden:
    """Mean-pooled tokens of selected hidden layers of a HF vision model (frame-wise), concatenated."""
    def __init__(self, name, layers, kind='dinov2'):
        from transformers import AutoImageProcessor, AutoModel
        self.model, self.revision, self.layers, self.kind = name, 'main', layers, kind
        self.p = AutoImageProcessor.from_pretrained(name)
        m = AutoModel.from_pretrained(name)
        self.m = (m.vision_model if kind == 'siglip' else m).eval().to(DEV)

    @torch.no_grad()
    def __call__(self, F):
        x = self.p(images=list(F), return_tensors='pt')['pixel_values'].to(DEV)
        hs = self.m(pixel_values=x, output_hidden_states=True).hidden_states
        z = torch.cat([hs[l].mean(1) for l in self.layers], 1).float().cpu().numpy()
        return z.mean(0), z


class VideoMAE:
    model, revision = 'MCG-NJU/videomae-base', 'main'

    def __init__(self):
        from transformers import AutoImageProcessor, VideoMAEModel
        self.p = AutoImageProcessor.from_pretrained(self.model); self.m = VideoMAEModel.from_pretrained(self.model).eval().to(DEV)
        self.n_frames = 16

    @torch.no_grad()
    def __call__(self, F):
        x = self.p(list(F), return_tensors='pt')['pixel_values'].to(DEV)
        hs = self.m(pixel_values=x, output_hidden_states=True).hidden_states
        z = torch.cat([hs[l].mean(1) for l in (6, 9, 12)], 1).float().cpu().numpy()[0]
        return z, None


class S3D:
    """torchvision S3D (Kinetics-400): mean-pooled features of 3 depths of the backbone, 16 frames."""
    model, revision = 'torchvision/s3d', 'KINETICS400_V1'

    def __init__(self):
        import torchvision
        self.m = torchvision.models.video.s3d(weights='KINETICS400_V1').eval().to(DEV)
        self.n_frames = 16
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1); self.std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def __call__(self, F):
        x = to_tensor(F, 224).permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, T, H, W)
        x = ((x - self.mean) / self.std).to(DEV)
        outs = []
        for i, blk in enumerate(self.m.features):
            x = blk(x)
            if i in (6, 11, 15):
                outs.append(x.mean((2, 3, 4)))
        return torch.cat(outs, 1).float().cpu().numpy()[0], None


REG = {'alexnet_f3': AlexNetF3, 'siglip_b16': SigLIP,
       'clip_b32': lambda: OpenCLIP('ViT-B-32', 'openai'), 'clip_l14': lambda: OpenCLIP('ViT-L-14', 'openai'),
       'dinov2_b_l4_8_12': lambda: HFHidden('facebook/dinov2-base', (4, 8, 12)),
       'siglip_b16_l4_8_12': lambda: HFHidden('google/siglip-base-patch16-224', (4, 8, 12), kind='siglip'),
       'videomae_b_l6_9_12': VideoMAE, 's3d_k400': S3D}


def main(names):
    paths = videos()
    for name in names:
        ex = REG[name](); ids, M, T = [], [], []; t0 = time.time()
        for i, p in enumerate(paths):
            m, t = ex(frames(p, getattr(ex, 'n_frames', 8)))
            ids.append(int(os.path.basename(p)[:4])); M.append(m)
            if t is not None:
                T.append(t)
            if i % 200 == 0:
                print(name, i, f'{time.time() - t0:.0f}s', flush=True)
        out = fs.save(name, ids, np.stack(M), ex.model, ex.revision, np.stack(T) if T else None)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print('saved', out, np.stack(M).shape, flush=True)
        del ex; torch.cuda.empty_cache()


if __name__ == '__main__':
    main(sys.argv[1:])
