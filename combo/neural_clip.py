"""Contrastive multi-subject EEG encoder as one more scorer family (NC1).

EEG 0-800 ms (winfuse preprocessing, 17 ch x 200 samples) -> per-subject 1x1 spatial mix (17 -> 32) -> shared temporal
conv net -> one projection head per target embedding (CLIP-L/14, V-JEPA 2 giant L30, DINOv2-B, human-text CLIP).
Loss: symmetric InfoNCE against the batch, learnable temperature, per head. Training pools the TRAINING recordings of
all six people. For a development subject's VAL matrix, that subject's VAL videos are removed from every person's
training rows (as in combo.build_pooled), so validation mimics unseen test videos. Test videos are never trained on.
Fixed epochs, no checkpoint selection on validation. Ensemble over seeds.

    python combo/neural_clip.py val   sub-01         # VAL + dev-test matrices for one dev subject
    python combo/neural_clip.py final                 # test matrices for all six (trained on all training rows)
Writes outputs/neural_clip_20261018/<sub>__NC1_<target>.pkl and <sub>__NC1_all.pkl in the lab format.
"""
import os, pickle, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('COMBO_CACHE', os.path.join(os.environ.get('TMPDIR', '/tmp'), 'nc_combo_cache'))
import combo  # noqa: E402
if os.environ.get('NC_SPLIT'):                           # alternative winfuse TRAIN/VAL split seed (robustness checks)
    combo.P.SPLIT_SEED = int(os.environ['NC_SPLIT'])

ALL6 = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
DEV = ALL6[:4]
TARGETS = os.environ.get('NC_TARGETS', 'clip_l14,vjepa2_g_l30,dinov2_b_l4_8_12').split(',')
OUT = os.path.join(combo.ND, 'outputs', os.environ.get('NC_OUT', 'neural_clip_20261018'))
TAG = os.environ.get('NC_TAG', 'NC1')
T_SAMPLES = int(os.environ.get('NC_T', 200))     # 200 = 0-800 ms at 250 Hz
HID = int(os.environ.get('NC_HID', 256))
WD = float(os.environ.get('NC_WD', 0.05))
AVG_P = float(os.environ.get('NC_AVG', 0))       # prob. of averaging a sample with same-video recordings of other people
AVG_K = int(os.environ.get('NC_AVG_K', 2))
ARCH = os.environ.get('NC_ARCH', 'conv')
TTA = os.environ.get('NC_TTA', '0') == '1'          # average scores over small time shifts
FT = int(os.environ.get('NC_FT', 0))                # per-person fine-tuning epochs after pooled training
ADAPT = os.environ.get('NC_ADAPT', '0') == '1'      # per-person residual adapter on the trunk output
OWN = os.environ.get('NC_OWN', '0') == '1'          # train on the target person's rows only (no pooling)
SAVE = os.environ.get('NC_SAVE', '')                  # dir: save per-seed weights + candidate embeddings (final mode)
EPOCHS = int(os.environ.get('NC_EPOCHS', 40))
SEEDS = [int(s) for s in os.environ.get('NC_SEEDS', '0,1,2').split(',')]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def targets(ids):
    out = {}
    for t in TARGETS:
        if t.startswith('wf:'):                                    # winfuse stimulus bank, e.g. wf:resnet50@layer_layer3
            split = 'test' if int(ids[0]) > 1000 else 'train'
            X = np.asarray(combo.stim(t[3:], list(ids), split), dtype=np.float32)
            if X.shape[1] > 2048:                                  # fixed random projection keeps heads small
                rng = np.random.default_rng(0); X = X @ (rng.standard_normal((X.shape[1], 1024)).astype(np.float32) / 32)
            out[t] = X
            continue
        z = np.load(os.path.join(combo.OURS, f'{t}.npz')); tab = {int(i): v for i, v in zip(z['ids'], z['feats'])}
        out[t] = np.stack([tab[int(i)] for i in ids]).astype(np.float32)
    return out


class Net(nn.Module):
    def __init__(self, dims, n_sub=6, ch=17, mid=32, hid=256):
        super().__init__()
        self.spatial = nn.Parameter(torch.eye(ch, mid).repeat(n_sub, 1, 1) + 0.01 * torch.randn(n_sub, ch, mid))
        self.temporal = nn.Sequential(
            nn.Conv1d(mid, 64, 9, padding=4), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.3),
            nn.Conv1d(64, 64, 9, padding=4, stride=2), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.3),
            nn.Conv1d(64, 64, 7, padding=3, stride=2), nn.BatchNorm1d(64), nn.GELU())
        if ARCH == 'linear':                    # spatial mix -> flatten -> linear; no conv, no nonlinearity
            self.temporal = nn.AvgPool1d(2)
            self.trunk = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.LazyLinear(hid))
        else:
            self.trunk = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.LazyLinear(hid), nn.GELU(), nn.Dropout(0.3))
        self.heads = nn.ModuleDict({k: nn.Linear(hid, d) for k, d in dims.items()})
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        if ADAPT:
            self.adapt = nn.Parameter(torch.zeros(n_sub, hid, hid))

    def mix(self, x, s):
        return torch.einsum('bct,bcm->bmt', x, self.spatial[s])

    def head(self, z, s=None):
        h = self.trunk(self.temporal(z))
        if ADAPT and s is not None:                 # per-person residual adapter, starts at identity
            h = h + torch.einsum('bh,bhk->bk', h, self.adapt[s])
        return {k: Fn.normalize(m(h), dim=-1) for k, m in self.heads.items()}

    def forward(self, x, s):
        return self.head(self.mix(x, s), s)


def prep(X):
    X = np.asarray(X[:, :, :T_SAMPLES], dtype=np.float32)
    return X / (X.std(axis=(1, 2), keepdims=True) + 1e-6)


def train(rows, seed):
    """rows: list of (X (n,17,T), subject index, target dict, video ids)."""
    torch.manual_seed(seed); np.random.seed(seed)
    V = np.concatenate([np.asarray(r[3]) for r in rows])
    same = {}
    for j, v in enumerate(V):
        same.setdefault(int(v), []).append(j)
    X = torch.tensor(np.concatenate([r[0] for r in rows]))
    S = torch.tensor(np.concatenate([np.full(len(r[0]), r[1]) for r in rows]), dtype=torch.long)
    Y = {t: torch.tensor(np.concatenate([r[2][t] for r in rows])) for t in TARGETS}
    mu = {t: Y[t].mean(0, keepdim=True) for t in TARGETS}
    Y = {t: Fn.normalize(Y[t] - mu[t], dim=-1) for t in TARGETS}
    net = Net({t: Y[t].shape[1] for t in TARGETS}, hid=HID).to(DEVICE)
    net(X[:2].to(DEVICE), S[:2].to(DEVICE))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=WD)
    n, bs = len(X), 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * ((n + bs - 1) // bs))
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            def aug(x):
                x = torch.roll(x, np.random.randint(-5, 6), dims=2) + 0.1 * torch.randn_like(x)   # +-20 ms jitter
                return x * (torch.rand(x.shape[0], x.shape[1], 1, device=DEVICE) > 0.1)          # channel dropout
            z = net.mix(aug(X[b].to(DEVICE)), S[b].to(DEVICE))
            if AVG_P > 0:                       # multi-person averaging in the shared space (mimics repeat-averaged test)
                src, dst = [], []
                for q, j in enumerate(b.tolist()):
                    if np.random.rand() < AVG_P:
                        cand = [m for m in same[int(V[j])] if S[m] != S[j]]
                        for m in np.random.permutation(cand)[:AVG_K]:
                            src.append(int(m)); dst.append(q)
                if src:
                    zp = net.mix(aug(X[src].to(DEVICE)), S[src].to(DEVICE))
                    acc = z.clone(); cnt = torch.ones(len(b), device=DEVICE)
                    acc = acc.index_add(0, torch.tensor(dst, device=DEVICE), zp)
                    cnt = cnt.index_add(0, torch.tensor(dst, device=DEVICE), torch.ones(len(dst), device=DEVICE))
                    z = acc / cnt[:, None, None]
            out = net.head(z, S[b].to(DEVICE))
            scale = net.logit_scale.exp().clamp(max=100)
            loss = 0
            for t in TARGETS:
                y = Y[t][b].to(DEVICE); L = scale * out[t] @ y.T
                lbl = torch.arange(len(b), device=DEVICE)
                loss = loss + (Fn.cross_entropy(L, lbl) + Fn.cross_entropy(L.T, lbl)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net, mu


def finetune(net, mu, X, s, Y, epochs=FT):
    """Copy of the pooled net, trained a few epochs on one person's rows only (lower lr)."""
    import copy
    net = copy.deepcopy(net); net.train()
    X = torch.tensor(X); Yn = {t: Fn.normalize(torch.tensor(Y[t]) - mu[t], dim=-1) for t in TARGETS}
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=WD)
    n, bs = len(X), 128
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            x = torch.roll(X[b].to(DEVICE), np.random.randint(-5, 6), dims=2)
            out = net(x + 0.1 * torch.randn_like(x), torch.full((len(b),), s, dtype=torch.long, device=DEVICE))
            scale = net.logit_scale.exp().clamp(max=100); lbl = torch.arange(len(b), device=DEVICE); loss = 0
            for t in TARGETS:
                L = scale * out[t] @ Yn[t][b].to(DEVICE).T
                loss = loss + (Fn.cross_entropy(L, lbl) + Fn.cross_entropy(L.T, lbl)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def score(net, mu, X, s, cand):
    xs = torch.tensor(X).to(DEVICE); sv = torch.full((len(X),), s, dtype=torch.long, device=DEVICE)
    if TTA:
        outs = [net(torch.roll(xs, k, dims=2), sv) for k in (-4, -2, 0, 2, 4)]
        out = {t: Fn.normalize(sum(o[t] for o in outs), dim=-1) for t in TARGETS}
    else:
        out = net(xs, sv)
    res = {}
    for t in TARGETS:
        c = Fn.normalize(torch.tensor(cand[t]) - mu[t], dim=-1).to(DEVICE)
        res[t] = (out[t] @ c.T).cpu().numpy()
    return res


def load_all():
    from mindread.data import load_test
    D = {}
    for k, s in enumerate(ALL6):
        Xtr, ids = combo.pp.preprocessed_train(s); ids = list(ids)
        if s in DEV:
            Xte = combo.pp.preprocessed_test(s, reason='neural NC1 (dev subject)')
        else:
            Xte = combo.pp.get_preprocessor(s).transform(load_test(combo.DATA, int(s[-1]), final_inference=True))
        D[s] = dict(k=k, X=prep(Xtr), ids=ids, Y=targets(ids), Xte=prep(Xte))
    return D


def save(sub, name, mats):
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(mats, open(os.path.join(OUT, f'{sub}__{TAG}_{name}.pkl'), 'wb'))


def run_val(D, sub):
    t0 = time.time()
    d = D[sub]; itr, iva = combo.P.split_ids(d['ids'])
    vids = {d['ids'][i] for i in iva}
    rows = []
    for s in ([sub] if OWN else ALL6):
        keep = [i for i, v in enumerate(D[s]['ids']) if v not in vids]
        rows.append((D[s]['X'][keep], D[s]['k'], {t: D[s]['Y'][t][keep] for t in TARGETS}, [D[s]['ids'][i] for i in keep]))
    Yva = {t: d['Y'][t][iva] for t in TARGETS}
    Ytest = targets(combo.P.TEST_IDS)
    truth = np.array([combo.P.TEST_IDS.index(v) for v in combo.cio.load_test_labels(sub, reason='neural NC1')]) if sub in DEV else None
    acc_v = {t: 0 for t in TARGETS}; acc_t = {t: 0 for t in TARGETS}
    for seed in SEEDS:
        net, mu = train(rows, seed)
        if FT:
            r = rows[0] if OWN else rows[ALL6.index(sub)]
            net = finetune(net, mu, r[0], d['k'], r[2])
        v = score(net, mu, d['X'][iva], d['k'], Yva)
        te = score(net, mu, d['Xte'], d['k'], Ytest)
        for t in TARGETS:
            acc_v[t] = acc_v[t] + v[t] / len(SEEDS); acc_t[t] = acc_t[t] + te[t] / len(SEEDS)
    for t in TARGETS:
        save(sub, t, {'val': acc_v[t], 'test': acc_t[t], 'truth': truth})
    save(sub, 'all', {'val': sum(acc_v.values()) / len(TARGETS), 'test': sum(acc_t.values()) / len(TARGETS), 'truth': truth})
    print('NC1 val', sub, round(time.time() - t0), 's', flush=True)


def run_final(D):
    t0 = time.time()
    rows = [(D[s]['X'], D[s]['k'], D[s]['Y'], D[s]['ids']) for s in ALL6]
    Ytest = targets(combo.P.TEST_IDS)
    acc = {s: {t: 0 for t in TARGETS} for s in ALL6}
    if SAVE:
        os.makedirs(SAVE, exist_ok=True)
        allids = list(range(1, 1001)) + list(combo.P.TEST_IDS)
        tr_ids = [i for i in allids if i <= 1000]
        cand = {t: np.concatenate([targets(tr_ids)[t], Ytest[t]]) for t in TARGETS}
        torch.save({'ids': allids, 'cand': {t: torch.tensor(v).half() for t, v in cand.items()}, 'targets': TARGETS,
                    'subjects': ALL6, 'config': dict(T=T_SAMPLES, HID=HID, ARCH=ARCH, ADAPT=ADAPT, FT=FT, EPOCHS=EPOCHS, WD=WD, seeds=SEEDS)},
                   os.path.join(SAVE, f'{TAG}_candidates.pt'))
    for seed in SEEDS:
        net, mu = train(rows, seed)
        if SAVE:
            torch.save({'state_dict': net.state_dict(), 'mu': {t: v.cpu() for t, v in mu.items()},
                        'dims': {t: int(Ytest[t].shape[1]) for t in TARGETS}}, os.path.join(SAVE, f'{TAG}_seed{seed}.pt'))
        for s in ALL6:
            m = finetune(net, mu, D[s]['X'], D[s]['k'], D[s]['Y']) if FT else net
            te = score(m, mu, D[s]['Xte'], D[s]['k'], Ytest)
            for t in TARGETS:
                acc[s][t] = acc[s][t] + te[t] / len(SEEDS)
    for s in ALL6:
        for t in TARGETS:
            save(s, f'final_{t}', {'test': acc[s][t]})
        save(s, 'final_all', {'test': sum(acc[s].values()) / len(TARGETS)})
    print('NC1 final', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    D = load_all()
    if sys.argv[1] == 'val':
        for sub in sys.argv[2:] or DEV:
            run_val(D, sub)
    else:
        run_final(D)
