import { useMemo, useState } from "react";
import { EXPERIMENTS, SYSTEMS } from "../content/team";
import { modelInfo, type ModelId } from "../content/models";
import { channelName } from "../lib/montage";
import { N_CHANNELS, type Recording } from "../lib/eeg";
import { BANDS, spectrum } from "../lib/spectrum";
import { useTheme } from "../lib/themeMode";

/** Results section of the home page (was the separate Analytics page). */
export function AnalyticsSections({ model }: { model: ModelId }) {
  return (
    <section data-brain-neurons className="grid min-w-0 grid-cols-1 gap-6" aria-labelledby="analytics-title">
      {/* the heading stays in the left column so the spine passes beside it */}
      <div className="lg:grid lg:grid-cols-[minmax(0,1fr)_var(--spine-gutter)_minmax(0,1fr)]">
        <div>
        <p className="eyebrow">Results</p>
        <h2 id="analytics-title" className="mt-4 text-4xl sm:text-5xl">From 11% to <span className="grad-word">35% top-1</span></h2>
        <p className="mt-3 max-w-2xl text-base font-light text-subtle">
          Official dev test, sub-01..04 (408 recordings, 102 candidates each). Numbers from the team's frozen configs.
        </p>
        </div>
      </div>
      <Progression highlight={model} />
      <ExperimentsTable />
    </section>
  );
}

/** #17: every system from the first baseline to C13, top-5 and top-1 bars. */
function Progression({ highlight }: { highlight: ModelId }) {
  const { t } = useTheme();
  const W = 980, H = 300, L = 46, B = 236, T = 24, bw = (W - L - 10) / SYSTEMS.length;
  const Y = (v: number) => B - ((B - T) * v) / 80;
  const hi = modelInfo(highlight).id === "WF" ? "WF1" : highlight;
  return (
    <section className="card p-5" aria-labelledby="prog-title">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="prog-title" className="card-title">From baseline to final system</h2>
        <p className="flex items-center gap-3 text-[11px] text-subtle">
          <span className="inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-iris" /> top-5</span>
          <span className="inline-flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-pine" /> top-1</span>
          <span>selected model outlined</span>
        </p>
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 w-full min-w-[640px]" role="img"
          aria-label={`Top-5 rose from ${SYSTEMS[0].top5}% (${SYSTEMS[0].name}) to ${SYSTEMS.at(-1)!.top5}% (${SYSTEMS.at(-1)!.name}).`}>
          {[0, 20, 40, 60, 80].map((v) => (
            <g key={v}>
              <line x1={L} x2={W - 10} y1={Y(v)} y2={Y(v)} stroke={t["highlight-med"]} strokeDasharray="2 4" />
              <text x={L - 8} y={Y(v) + 4} textAnchor="end" fontSize="11" fill={t.subtle}>{v}%</text>
            </g>
          ))}
          {SYSTEMS.map((s, i) => {
            const x = L + i * bw + bw * 0.16, w = bw * 0.32, sel = s.name === hi;
            return (
              <g key={s.name}>
                {sel && <rect x={x - 6} y={T - 14} width={w * 2 + 16} height={B - T + 52} rx="8" fill="none" stroke={t.gold} strokeWidth="1.5" />}
                <rect x={x} y={Y(s.top5)} width={w} height={B - Y(s.top5)} rx="4" fill={t.iris} opacity={s.final || sel ? 1 : 0.5} />
                <rect x={x + w + 4} y={Y(s.top1)} width={w} height={B - Y(s.top1)} rx="4" fill={t.pine} opacity={s.final || sel ? 1 : 0.6} />
                <text x={x + w / 2} y={Y(s.top5) - 5} textAnchor="middle" fontSize="10" fill={t.text} fontFamily="ui-monospace, monospace">{s.top5.toFixed(1)}</text>
                <text x={x + w * 1.5 + 4} y={Y(s.top1) - 5} textAnchor="middle" fontSize="10" fill={t.text} fontFamily="ui-monospace, monospace">{s.top1.toFixed(1)}</text>
                <text x={x + w + 2} y={B + 17} textAnchor="middle" fontSize="12" fontWeight={s.final ? 700 : 500} fill={t.text}>{s.name}</text>
                <text x={x + w + 2} y={B + 32} textAnchor="middle" fontSize="9.5" fill={t.subtle}>{s.note}</text>
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}

interface SpectrumProps {
  rec: Recording | null;
  subject: string | null;
  /** when given, the panel follows this recording and these channels (analysis page) */
  trial?: number;
  channels?: number[];
}

/** #16: power spectrum and band power of a recording, per electrode. */
export function SpectrumPanel({ rec, subject, trial: trialProp, channels }: SpectrumProps) {
  const { t } = useTheme();
  const [ownTrial, setTrial] = useState(0);
  const [mode, setMode] = useState<"psd" | "band">("psd");
  const [ownChans, setChans] = useState<number[]>(() => Array.from({ length: N_CHANNELS }, (_, i) => i));
  const controlled = trialProp !== undefined;
  const trial = trialProp ?? ownTrial;
  const chans = channels ?? ownChans;
  const sp = useMemo(() => (rec ? spectrum(rec, Math.min(trial, rec.nTrials - 1), chans) : null), [rec, trial, chans]);
  const bandColor = [t.pine, t.iris, t.foam, t.gold, t.love];

  if (!rec || !sp)
    return (
      <section className="card p-5">
        <h2 className="card-title">Frequency analysis</h2>
        <p className="mt-2 text-sm text-subtle">
          Choose a person to see the power spectrum of their recordings.
        </p>
      </section>
    );

  const W = 560, H = 240, L = 46, B = 204, R = W - 12, T = 24;
  const db = sp.psd.map(([, p]) => 10 * Math.log10(p + 1e-12));
  const lo = Math.min(...db), hi = Math.max(...db);
  const X = (f: number) => L + ((R - L) * f) / 45;
  const Yp = (v: number) => B - ((B - T) * (v - lo)) / (hi - lo || 1);
  const maxBand = Math.max(...sp.bands);

  return (
    <section className="card p-5" aria-labelledby="psd-title">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="psd-title" className="card-title">
          Frequency analysis · {subject ?? rec.fileName} · recording {trial + 1}
        </h2>
        <div className="flex flex-wrap items-center gap-2">
          {!controlled && (
            <input type="range" min={0} max={rec.nTrials - 1} value={trial} onChange={(e) => setTrial(Number(e.target.value))}
              aria-label="Recording" className="w-40 accent-iris" />
          )}
          <div role="radiogroup" aria-label="Spectrum view" className="flex rounded-xl border border-highlight-med p-0.5 text-xs">
            {(["psd", "band"] as const).map((m) => (
              <button key={m} type="button" role="radio" aria-checked={mode === m} onClick={() => setMode(m)}
                className={`rounded-lg px-3 py-1 ${mode === m ? "bg-iris text-on-iris" : "text-subtle hover:text-text"}`}>
                {m === "psd" ? "PSD" : "Band power"}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-3 grid gap-5 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
          aria-label={`Relative band power: ${BANDS.map((b, i) => `${b.name} ${(sp.bands[i] * 100).toFixed(0)}%`).join(", ")}`}>
          {mode === "psd" ? (
            <>
              {BANDS.map((b, i) => (
                <rect key={b.name} x={X(b.lo)} y={T} width={X(b.hi) - X(b.lo)} height={B - T} fill={bandColor[i]} opacity="0.08" />
              ))}
              <path d={sp.psd.map(([f], i) => `${i ? "L" : "M"}${X(f).toFixed(1)},${Yp(db[i]).toFixed(1)}`).join(" ")} fill="none" stroke={t.iris} strokeWidth="2" />
              {[0, 10, 20, 30, 40].map((f) => (
                <text key={f} x={X(f)} y={B + 16} textAnchor="middle" fontSize="11" fill={t.subtle}>{f}</text>
              ))}
              <text x={(L + R) / 2} y={H - 4} textAnchor="middle" fontSize="11" fill={t.subtle}>Frequency (Hz)</text>
              <text x={12} y={(T + B) / 2} transform={`rotate(-90 12 ${(T + B) / 2})`} textAnchor="middle" fontSize="11" fill={t.subtle}>Power (dB)</text>
            </>
          ) : (
            BANDS.map((b, i) => {
              const bw = (R - L) / BANDS.length, h = ((B - T) * sp.bands[i]) / (maxBand || 1);
              return (
                <g key={b.name}>
                  <rect x={L + i * bw + 12} y={B - h} width={bw - 24} height={h} rx="6" fill={bandColor[i]} />
                  <text x={L + i * bw + bw / 2} y={B - h - 6} textAnchor="middle" fontSize="11" fill={t.text} fontFamily="ui-monospace, monospace">
                    {(sp.bands[i] * 100).toFixed(0)}%
                  </text>
                  <text x={L + i * bw + bw / 2} y={B + 16} textAnchor="middle" fontSize="11" fill={t.subtle}>{b.name}</text>
                </g>
              );
            })
          )}
        </svg>
        <div>
          <ul className="grid gap-2 text-xs">
            {BANDS.map((b, i) => (
              <li key={b.name} className="grid grid-cols-[110px_minmax(0,1fr)_40px] items-center gap-2">
                <span className="text-subtle">{b.name} ({b.lo}–{b.hi} Hz)</span>
                <span className="h-2 overflow-hidden rounded-full bg-highlight-low">
                  <span className="block h-full rounded-full" style={{ width: `${(sp.bands[i] * 100).toFixed(1)}%`, background: bandColor[i] }} />
                </span>
                <span className="text-right font-mono text-text">{(sp.bands[i] * 100).toFixed(0)}%</span>
              </li>
            ))}
          </ul>
          {!controlled && (
          <fieldset className="mt-4">
            <legend className="text-xs text-subtle">Channels averaged</legend>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {Array.from({ length: N_CHANNELS }, (_, c) => (
                <label key={c} className="chip cursor-pointer bg-highlight-low font-mono text-subtle">
                  <input type="checkbox" className="accent-iris" checked={chans.includes(c)}
                    onChange={() => setChans((cur) => (cur.includes(c) ? cur.filter((x) => x !== c) : [...cur, c].sort((a, b) => a - b)))} />
                  {channelName(c)}
                </label>
              ))}
            </div>
          </fieldset>
          )}
          <p className="mt-3 text-[11px] text-subtle">
            Hann-windowed periodogram of the first 2.05 s (1024 samples at 500 Hz) of the recording as stored. Channels are
            named by electrode (all over the back of the head: parietal and occipital).
          </p>
        </div>
      </div>
    </section>
  );
}

/** Client-facing wording for the experiments that are part of the final pipeline. */
const KEPT: Record<string, { title: string; what: string }> = {
  Robustness: { title: "Robust gains", what: "Rebuilt bit-identically on a fresh GPU; new seeds stay within one standard error" },
  "C12 heads": { title: "Richer network targets", what: "Network also predicts AlexNet, ResNet-50 and VideoMAE features" },
  "X14 neural": { title: "Cross-person neural network", what: "Contrastive network on 0–600 ms, trained across all six people" },
};

/** #20: the experiments that made it into the final pipeline, in the same card style as the chart. */
function ExperimentsTable() {
  const kept = EXPERIMENTS.filter((e) => e.kind === "pos" && KEPT[e.id]);
  return (
    <section className="card p-5" aria-labelledby="exp-title">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="exp-title" className="card-title">What went into the final pipeline</h2>
        <p className="text-[11px] text-subtle">dev test, sub-01..04</p>
      </div>
      <ul className="mt-4 grid gap-3 md:grid-cols-3">
        {kept.map((e) => (
          <li key={e.id} className="rounded-2xl border border-highlight-med/70 px-4 py-3">
            <p className="text-sm font-medium text-text">{KEPT[e.id].title}</p>
            <p className="mt-1 text-xs leading-relaxed text-subtle">{KEPT[e.id].what}</p>
            <p className="mt-2 font-mono text-xs text-iris">{e.dev}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}
