"""Composable identification methods for the D/E phases.

A method is fitted per subject on training videos only and returns a score matrix
(queries x candidates). `bench.run_method` evaluates it on V1 and V2 exactly like the encoder runs.

Methods
- Encode:      one or more feature bands -> EEG (plain ridge on concatenated z-scored bands, or himalaya banded ridge)
- latency routing (N9):  separate band weights per time window (one banded ridge per window)
- shrinkage (N10):       training targets pulled toward the other subjects' response to the same training video
- reliability selection: keep the channel x time outputs with the best inner-CV prediction
- csls (N11):            hubness correction of the score matrix (label-free, uses the query set jointly)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .data import assert_training_ids
from .encoder import corr_scores


def zfit(F):
    sc = StandardScaler().fit(F)
    return sc, sc.transform(F)


class Encode:
    """Video features (list of bands) -> EEG target; scores = Pearson(observed, predicted)."""

    def __init__(self, bands, banded=False, windows=None, n_out_keep=None, shrink=None, alphas=None, n_iter=30):
        self.bands, self.banded, self.windows = bands, banded, windows       # windows: list of slices over the time axis
        self.n_out_keep, self.shrink, self.n_iter = n_out_keep, shrink, n_iter
        self.alphas = np.logspace(-1, 6, 15) if alphas is None else alphas

    # -- helpers
    def _X(self, tables, ids, fit=False):
        Xs = []
        for k, (name, tab) in enumerate(zip(self.bands, tables)):
            F = np.stack([tab[int(i)] for i in ids]).reshape(len(ids), -1)
            if fit:
                self.sc[k] = StandardScaler().fit(F)
            Xs.append(self.sc[k].transform(F) / np.sqrt(F.shape[1]))   # equal total variance per band
        return Xs

    def _ridge(self, Xs, Y):
        if self.banded and len(Xs) > 1:
            from himalaya.ridge import GroupRidgeCV
            from himalaya.backend import set_backend
            set_backend('numpy', on_error='warn')
            X = np.concatenate(Xs, 1); groups = np.concatenate([np.full(x.shape[1], g) for g, x in enumerate(Xs)])
            m = GroupRidgeCV(groups=groups, solver_params=dict(n_iter=self.n_iter, alphas=self.alphas, progress_bar=False,
                                                              n_targets_batch=2000), random_state=0)
            m.fit(X.astype(np.float32), Y.astype(np.float32))
            return ('banded', m)
        m = RidgeCV(alphas=self.alphas).fit(np.concatenate(Xs, 1), Y)
        return ('ridge', m)

    @staticmethod
    def _pred(model, Xs):
        kind, m = model
        P = m.predict(np.concatenate(Xs, 1).astype(np.float32) if kind == 'banded' else np.concatenate(Xs, 1))
        return np.asarray(P, dtype=np.float64)

    # -- API
    def fit(self, ids, Y, tables, others=None):
        """ids: training video IDs; Y: (n, d) own targets; others: {video_id: [targets of other subjects]} (training videos only)."""
        assert_training_ids(ids)
        if self.shrink is not None and others:
            w = self.shrink
            Y = np.stack([w * y + (1 - w) * np.mean(others[int(i)], 0) if int(i) in others else y for i, y in zip(ids, Y)])
        self.sc = [None] * len(self.bands)
        Xs = self._X(tables, ids, fit=True)
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-9
        Yz = (Y - self.mu) / self.sd
        self.keep = None
        if self.n_out_keep:   # inner split on the training fold only: rank outputs by prediction accuracy
            rng = np.random.default_rng(0); idx = rng.permutation(len(ids)); a, b = idx[: int(.8 * len(idx))], idx[int(.8 * len(idx)):]
            mdl = self._ridge([x[a] for x in Xs], Yz[a]); P = self._pred(mdl, [x[b] for x in Xs])
            Pc, Oc = P - P.mean(0), Yz[b] - Yz[b].mean(0)
            r = (Pc * Oc).sum(0) / np.sqrt((Pc ** 2).sum(0) * (Oc ** 2).sum(0) + 1e-12)
            self.keep = np.sort(np.argsort(-r)[: self.n_out_keep])
        if self.windows:
            self.models = [self._ridge(Xs, Yz[:, w]) for w in self.windows]
        else:
            self.models = [self._ridge(Xs, Yz)]
        return self

    def predict(self, tables, ids):
        Xs = self._X(tables, ids)
        if self.windows:
            P = np.zeros((len(ids), len(self.mu)))
            for w, m in zip(self.windows, self.models):
                P[:, w] = self._pred(m, Xs)
        else:
            P = self._pred(self.models[0], Xs)
        return P

    def scores(self, Yq, tables, cand_ids):
        Yq = (Yq - self.mu) / self.sd
        P = self.predict(tables, cand_ids)
        if self.keep is not None:
            Yq, P = Yq[:, self.keep], P[:, self.keep]
        return corr_scores(Yq, P)


def csls(S, k=10):
    """Cross-domain similarity local scaling: 2S - r_query - r_candidate (mean of k nearest)."""
    k = min(k, S.shape[0] - 1, S.shape[1] - 1)
    r_q = np.sort(S, 1)[:, -k:].mean(1, keepdims=True)
    r_c = np.sort(S, 0)[-k:, :].mean(0, keepdims=True)
    return 2 * S - r_q - r_c


def time_windows(n_ch, n_t, edges):
    """Index arrays over a flattened (ch x t) target for consecutive time windows defined by sample edges."""
    grid = np.arange(n_ch * n_t).reshape(n_ch, n_t)
    return [grid[:, a:b].ravel() for a, b in zip(edges[:-1], edges[1:])]


class TRF:
    """Temporal response function: slow EEG (17 ch x T at 10 Hz over 0.5-3 s) predicted from stimulus time courses
    (lowlevel_official feats_t, 100 Hz) through one kernel over lags 0..max_lag shared across time and videos."""

    def __init__(self, ts_table, t0_ms=500, t1_ms=3000, fs=10, max_lag_ms=600, lag_step_ms=50, alphas=None):
        self.ts, self.t0, self.t1, self.fs = ts_table, t0_ms, t1_ms, fs
        self.lags = np.arange(0, max_lag_ms + 1, lag_step_ms) // 10           # in 100 Hz samples
        self.times = np.arange(t0_ms, t1_ms, 1000 // fs) // 10                 # target sample positions at 100 Hz
        self.alphas = np.logspace(0, 6, 13) if alphas is None else alphas

    def _design(self, ids):
        T = np.stack([self.ts[int(i)] for i in ids])                           # (n, k, 300)
        T = (T - self.m) / self.s
        # A negative lagged position is before stimulus onset, not the end of
        # the clip (NumPy would silently wrap a negative index). Out-of-clip
        # values contribute zero in the standardized stimulus coordinates.
        positions = self.times[:, None] - self.lags[None, :]
        valid = (positions >= 0) & (positions < T.shape[-1])
        X = T[:, :, np.clip(positions, 0, T.shape[-1] - 1)]
        X = np.where(valid[None, None, :, :], X, 0.0)                         # (n, k, nt, nlag)
        return X.transpose(0, 2, 1, 3).reshape(len(ids) * len(self.times), -1)  # rows = (video, time)

    def fit(self, ids, Yslow):
        assert_training_ids(ids)
        T = np.stack([self.ts[int(i)] for i in ids]); self.m = T.mean((0, 2), keepdims=True)[0]; self.s = T.std((0, 2), keepdims=True)[0] + 1e-9
        n, nt = len(ids), len(self.times)
        Y = Yslow.reshape(n, 17, nt).transpose(0, 2, 1).reshape(n * nt, 17)
        self.r = RidgeCV(alphas=self.alphas).fit(self._design(ids), Y)
        return self

    def predict(self, ids):
        nt = len(self.times)
        return self.r.predict(self._design(ids)).reshape(len(ids), nt, 17).transpose(0, 2, 1).reshape(len(ids), -1)


class TwoTimescale:
    """N1: onset encoder (bands -> 50-1000 ms) + sustained TRF (time courses -> <4 Hz 0.5-3 s); z-scored score fusion."""

    def __init__(self, bands, ts_table, w_slow=0.5, n_onset=17 * 95, onset_kw=None):
        self.onset = Encode(bands, **(onset_kw or {})); self.trf = TRF(ts_table); self.w, self.n_on = w_slow, n_onset

    def fit(self, ids, Y, tables, others=None):
        self.onset.fit(ids, Y[:, :self.n_on], tables, others)
        self.trf.fit(ids, Y[:, self.n_on:])
        return self

    @staticmethod
    def _z(S):
        return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)

    def scores(self, Yq, tables, cand_ids):
        S1 = self.onset.scores(Yq[:, :self.n_on], tables, cand_ids)
        if self.w == 0:
            return S1
        S2 = corr_scores(Yq[:, self.n_on:], self.trf.predict(cand_ids))
        self.last_slow = S2
        return self._z(S1) + self.w * self._z(S2) if self.w < 100 else S2


class Decode:
    """EEG -> video embedding (ridge); score = Pearson(predicted embedding, candidate embedding)."""

    def __init__(self, bands, alphas=None, band_scale=False):
        self.bands = bands; self.alphas = np.logspace(0, 7, 15) if alphas is None else alphas
        # band_scale: z-score each band and divide by sqrt(dim), as Encode does, so wide bands do not dominate
        self.band_scale = band_scale

    def _F(self, tables, ids, fit=False):
        Fs = [np.stack([t[int(i)] for i in ids]).reshape(len(ids), -1) for t in tables]
        if not self.band_scale:
            F = np.concatenate(Fs, 1)
            if fit:
                self.fsc = StandardScaler().fit(F)
            return self.fsc.transform(F)
        if fit:
            self.fsc = [StandardScaler().fit(F) for F in Fs]
        return np.concatenate([sc.transform(F) / np.sqrt(F.shape[1]) for sc, F in zip(self.fsc, Fs)], 1)

    def fit(self, ids, Y, tables, others=None):
        assert_training_ids(ids)
        self.ysc = StandardScaler().fit(Y)
        self.r = RidgeCV(alphas=self.alphas).fit(self.ysc.transform(Y), self._F(tables, ids, fit=True))
        return self

    def scores(self, Yq, tables, cand_ids):
        return corr_scores(self.r.predict(self.ysc.transform(Yq)), self._F(tables, cand_ids))


class Fuse:
    """N7: z-scored (per query) sum of encoder and decoder scores."""

    def __init__(self, enc, dec, w=0.5):
        self.enc, self.dec, self.w = enc, dec, w

    def fit(self, ids, Y, tables, others=None):
        self.enc.fit(ids, Y, tables, others); self.dec.fit(ids, Y, tables, others)
        return self

    def scores(self, Yq, tables, cand_ids):
        z = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)
        a = self.enc.scores(Yq, tables, cand_ids)
        if self.w >= 100:
            return self.dec.scores(Yq, tables, cand_ids)
        return z(a) + self.w * z(self.dec.scores(Yq, tables, cand_ids))


class SeparateStreams:
    """Independent bidirectional ridge scorers with externally fixed fusion.

    ``tables_by_band`` maps feature names to the existing video-ID tables.
    Each extra stream may use a subset of flattened EEG target columns. The
    baseline always uses the complete target and its original Fuse geometry.
    This class never selects weights: a development harness must learn them
    from inner-held-out predictions, without the outer validation responses.
    """

    def __init__(self, baseline_bands, stream_bands, weights=None,
                 decoder_weight=2.0, stream_columns=None, decoder_band_scale=False):
        if 'baseline' in stream_bands:
            raise ValueError("'baseline' is reserved for the frozen scorer")
        self.baseline_bands = list(baseline_bands)
        self.stream_bands = {name: list(bands) for name, bands in stream_bands.items()}
        self.decoder_weight = decoder_weight
        self.decoder_band_scale = decoder_band_scale
        self.stream_columns = dict(stream_columns or {})
        if set(self.stream_columns) - set(self.stream_bands):
            raise ValueError('stream_columns contains an unknown stream')
        self.weights = {'baseline': 1.0, **{name: 0.0 for name in self.stream_bands}}
        if weights is not None:
            if set(weights) - set(self.weights):
                raise ValueError('weights contains an unknown stream')
            self.weights.update(weights)
        self._check_weights()

    def _check_weights(self):
        expected = {'baseline', *self.stream_bands}
        if set(self.weights) != expected:
            raise ValueError('weights must name every scorer exactly once')
        values = np.asarray(list(self.weights.values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.any(values > 0):
            raise ValueError('fusion weights must be finite, nonnegative and not all zero')

    def fit(self, ids, Y, tables_by_band, others=None):
        assert_training_ids(ids)
        self.n_target_columns = Y.shape[1]
        self.columns = {'baseline': np.arange(self.n_target_columns)}
        for name in self.stream_bands:
            cols = np.asarray(self.stream_columns.get(name, self.columns['baseline']))
            if (cols.ndim != 1 or not len(cols) or not np.issubdtype(cols.dtype, np.integer)
                    or np.any(cols < 0) or np.any(cols >= self.n_target_columns)
                    or len(np.unique(cols)) != len(cols)):
                raise ValueError(f'invalid EEG target columns for stream {name!r}')
            self.columns[name] = cols.copy()
        self.band_sets = {'baseline': self.baseline_bands, **self.stream_bands}
        self.models = {}
        for name, bands in self.band_sets.items():
            if not bands:
                raise ValueError(f'stream {name!r} has no feature bands')
            model = Fuse(Encode(bands), Decode(bands, band_scale=self.decoder_band_scale), w=self.decoder_weight)
            # No cross-subject shrinkage is used by these scorers. Keep the
            # API argument for compatibility without importing other subjects.
            stream_target = Y if name == 'baseline' else Y[:, self.columns[name]]
            self.models[name] = model.fit(ids, stream_target,
                                          [tables_by_band[b] for b in bands])
        return self

    def score_components(self, Yq, tables_by_band, cand_ids):
        if Yq.shape[1] != self.n_target_columns:
            raise ValueError('query target width does not match the fitted target')
        return {name: self.models[name].scores(Yq if name == 'baseline' else Yq[:, self.columns[name]],
                                               [tables_by_band[b] for b in bands], cand_ids)
                for name, bands in self.band_sets.items()}

    def score_direction_components(self, Yq, tables_by_band, cand_ids):
        """Raw directions for caching across later candidate-subset draws.

        Fuse's row standardization depends on its candidate pool. Cache these
        raw correlations, slice the requested pool, and only then standardize
        each direction and combine as ``z(encoder) + w * z(decoder)``.
        """
        if Yq.shape[1] != self.n_target_columns:
            raise ValueError('query target width does not match the fitted target')
        out = {}
        for name, bands in self.band_sets.items():
            y = Yq if name == 'baseline' else Yq[:, self.columns[name]]
            tables = [tables_by_band[b] for b in bands]
            model = self.models[name]
            out[name] = {'encoder': model.enc.scores(y, tables, cand_ids),
                         'decoder': model.dec.scores(y, tables, cand_ids)}
        return out

    def scores(self, Yq, tables_by_band, cand_ids):
        self._check_weights()
        components = self.score_components(Yq, tables_by_band, cand_ids)
        # Keep the default baseline numerically identical to frozen J1.
        if self.weights['baseline'] == 1.0 and all(self.weights[n] == 0 for n in self.stream_bands):
            return components['baseline']
        out = np.zeros_like(components['baseline'])
        for name, score in components.items():
            z = (score - score.mean(1, keepdims=True)) / (score.std(1, keepdims=True) + 1e-12)
            out += self.weights[name] * z
        return out


class EncodeRouted:
    """Band-level latency routing (from the time map): each time segment is predicted by its own band subset.

    routes: list of (t_start, t_end, bands) in 100 Hz samples within the target window; every segment uses all 17 channels.
    """

    def __init__(self, all_bands, routes, n_ch=17, n_t=95):
        self.all_bands, self.n_ch, self.n_t = all_bands, n_ch, n_t
        grid = np.arange(n_ch * n_t).reshape(n_ch, n_t)
        self.routes = [(grid[:, a:b].ravel(), [all_bands.index(x) for x in bands]) for a, b, bands in routes]
        self.encs = [Encode([all_bands[i] for i in idx]) for _, idx in self.routes]

    def fit(self, ids, Y, tables, others=None):
        for (cols, idx), e in zip(self.routes, self.encs):
            e.fit(ids, Y[:, cols], [tables[i] for i in idx], None)
        return self

    def scores(self, Yq, tables, cand_ids):
        P = np.zeros((len(cand_ids), Yq.shape[1])); Z = np.zeros_like(Yq, dtype=float)
        for (cols, idx), e in zip(self.routes, self.encs):
            P[:, cols] = e.predict([tables[i] for i in idx], cand_ids)
            Z[:, cols] = (Yq[:, cols] - e.mu) / e.sd
        return corr_scores(Z, P)
