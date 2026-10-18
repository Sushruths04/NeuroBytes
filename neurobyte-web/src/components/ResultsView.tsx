import { useEffect, useMemo, useState } from "react";
import { loadView, saveView } from "../lib/session";
import { writeNpyInt64 } from "../lib/npy";
import { channelName } from "../lib/montage";
import { N_CHANNELS, type Recording } from "../lib/eeg";
import { type PipelineResult } from "../lib/pipeline";
import { modelInfo } from "../content/models";
import { EegCanvas } from "./EegCanvas";
import { SpectrumPanel } from "./AnalyticsView";
import { VideoCard } from "./VideoCard";
import { videoUrl } from "../lib/api";

interface Props {
  rec: Recording;
  result: PipelineResult;
  onReset: () => void;
  /** switch to another person's test recording */
  onPickSubject: (sub: string) => void;
  /** person being loaded in the background, if any */
  switching?: string | null;
  switchError?: string | null;
}

const SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"];
const pct = (x: number) => `${(x * 100).toFixed(2)}%`;
const ALL_CHANNELS = Array.from({ length: N_CHANNELS }, (_, i) => i);

export function ResultsView({ rec, result, onReset, onPickSubject, switching = null, switchError = null }: Props) {
  const info = modelInfo(result.model);
  // Restore the selected trial/view after a reload.
  const [trial, setTrial] = useState(() => Math.min(loadView().trial, result.n - 1));
  const [view, setView] = useState<"post" | "pre">(() => loadView().view);
  const [channels, setChannels] = useState<number[]>(ALL_CHANNELS);
  useEffect(() => saveView({ trial, view }), [trial, view]);
  const ranking = view === "post" ? result.post : result.pre;
  const acc = view === "post" ? result.postAcc : result.preAcc;
  const row = ranking.ranked[trial];
  const truth = result.labels?.[trial];
  const truthRank = truth !== undefined ? row.indexOf(truth) + 1 : null;
  const ranks = useMemo(() => result.postAcc?.ranks ?? null, [result]);
  const prob = (id: number) => ranking.probs[trial * 102 + (id - 1001)];
  const go = (i: number) => setTrial(((i % result.n) + result.n) % result.n);

  function download() {
    const flat = result.post.ranked.flat();
    const blob = new Blob([writeNpyInt64(flat, [result.n, 102])], { type: "application/octet-stream" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${result.subject}_ranked_candidate_list.npy`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const toggleChannel = (c: number) =>
    setChannels((cur) => (cur.includes(c) ? cur.filter((x) => x !== c) : [...cur, c].sort((a, b) => a - b)));

  return (
    <div className="grid gap-5">
      {/* person tabs + recording navigation */}
      <section className="card flex flex-wrap items-center justify-between gap-3 p-3" aria-label="Recording selection">
        <div role="tablist" aria-label="Person" className="flex flex-wrap gap-1">
          {SUBJECTS.map((s) => (
            <button
              key={s}
              type="button"
              role="tab"
              aria-selected={s === result.subject}
              aria-busy={s === switching}
              onClick={() => s !== result.subject && s !== switching && onPickSubject(s)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors duration-200 ${
                s === result.subject
                  ? "bg-iris text-on-iris"
                  : s === switching
                    ? "bg-iris/20 text-text"
                    : "text-subtle hover:bg-overlay hover:text-text"
              }`}
            >
              {s === switching && (
                <span aria-hidden className="mr-1.5 inline-block h-2 w-2 animate-spin rounded-full border border-current border-t-transparent align-middle" />
              )}
              {s}
              {(s === "sub-05" || s === "sub-06") && <span className="ml-1 text-[10px] opacity-80">no labels</span>}
            </button>
          ))}
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <button type="button" className="btn-ghost px-2.5 py-1 text-xs" onClick={() => go(trial - 1)} aria-label="Previous recording">‹</button>
          <select
            value={trial}
            onChange={(e) => setTrial(Number(e.target.value))}
            aria-label="Test recording"
            className="min-w-0 max-w-[12rem] rounded-lg border border-highlight-high bg-surface px-2 py-1 font-mono text-xs text-text sm:max-w-none"
          >
            {result.post.ranked.map((_, i) => (
              <option key={i} value={i}>
                Recording {String(i + 1).padStart(3, "0")}
                {ranks ? ` · true rank ${ranks[i]}` : ""}
              </option>
            ))}
          </select>
          <button type="button" className="btn-ghost px-2.5 py-1 text-xs" onClick={() => go(trial + 1)} aria-label="Next recording">›</button>
          <button type="button" className="btn-ghost px-2.5 py-1 text-xs" onClick={() => go(Math.floor(Math.random() * result.n))}>Random</button>
        </div>
      </section>

      <p role="status" aria-live="polite" className="sr-only">
        {switching ? `Loading ${switching}…` : `Showing ${result.subject}`}
      </p>
      {switchError && (
        <p role="alert" className="rounded-2xl border border-love/60 bg-love/15 px-3 py-2 text-xs text-text">
          Couldn't switch person: {switchError}
        </p>
      )}

      <section
        className={`grid gap-3 transition-opacity duration-300 sm:grid-cols-2 lg:grid-cols-5 ${switching ? "opacity-60" : ""}`}
        aria-label="Summary"
      >
        <Stat label="Model" value={info.short} sub={info.adds} />
        <Stat
          label="Top-1"
          value={result.postAcc ? pct(result.postAcc.top1) : "—"}
          sub={result.postAcc ? `before Sinkhorn ${pct(result.preAcc!.top1)} · chance 0.98%` : "no labels for this person"}
          tone="foam"
        />
        <Stat
          label="Top-5"
          value={result.postAcc ? pct(result.postAcc.top5) : "—"}
          sub={result.postAcc ? `before Sinkhorn ${pct(result.preAcc!.top5)} · chance 4.90%` : "no labels for this person"}
          tone="iris"
        />
        <Stat
          label="Mean rank"
          value={result.postAcc ? result.postAcc.meanRank.toFixed(2) : "—"}
          sub={result.postAcc ? `median ${result.postAcc.medianRank} of 102` : "—"}
          tone="gold"
        />
        <Stat
          label="Verified ranking"
          value={result.shipped ? `${result.shipped.equalRows}/${result.shipped.total}` : "—"}
          sub="rows identical to the Python pipeline's ranking"
          tone={result.shipped && result.shipped.equalRows === result.shipped.total ? "foam" : "love"}
        />
      </section>

      <div className="grid gap-5 lg:grid-cols-[240px_minmax(0,1fr)]">
        <section className="card flex max-h-[1100px] flex-col p-3" aria-labelledby="trials-title">
          <h2 id="trials-title" className="card-title px-1 pb-2">Recordings</h2>
          <ul className="-mx-1 flex-1 space-y-1 overflow-y-auto px-1" aria-label="Recordings">
            {result.post.ranked.map((r, i) => {
              const rank = ranks?.[i];
              return (
                <li key={i}>
                  <button
                    type="button"
                    aria-pressed={i === trial}
                    aria-label={`Recording ${i + 1}, top match video ${r[0]}${rank !== undefined ? `, true video ranked ${rank}` : ""}`}
                    onClick={() => setTrial(i)}
                    className={`flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left text-xs transition ${
                      i === trial ? "bg-overlay text-text" : "text-subtle hover:bg-overlay/50"
                    }`}
                  >
                    <span className="font-mono">#{String(i + 1).padStart(3, "0")}</span>
                    <span className="font-mono">→ {r[0]}</span>
                    {rank !== undefined ? <RankBadge rank={rank} /> : <span className="w-10" />}
                  </button>
                </li>
              );
            })}
          </ul>
        </section>

        <div className="grid min-w-0 gap-5">
          <section className="card p-4">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="card-title">
                  Recording <span className="font-mono text-foam">{trial + 1}</span> · EEG
                </h2>
                <Verdict rank={truthRank} unlabelled={!result.isDevSubject} />
              </div>
              <div className="flex gap-2">
                <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={download}>
                  Export ranked .npy
                </button>
                <button type="button" className="btn-primary px-3 py-1.5 text-xs" onClick={onReset}>
                  Choose another person
                </button>
              </div>
            </div>
            <EegCanvas rec={rec} trial={trial} cursor={null} height={260} windows={info.windows} channels={channels} />
            <p className="mt-1 text-[11px] text-subtle">
              Shaded: the part of each recording {info.short} reads ({info.windows[0].label} after the clip starts).
            </p>
            <fieldset className="mt-3">
              <legend className="text-xs text-subtle">Channels</legend>
              <div className="mt-1 flex flex-wrap gap-1.5">
                <label className="chip cursor-pointer bg-overlay text-text">
                  <input
                    type="checkbox"
                    className="accent-iris"
                    checked={channels.length === N_CHANNELS}
                    onChange={(e) => setChannels(e.target.checked ? ALL_CHANNELS : [])}
                  />
                  all
                </label>
                {ALL_CHANNELS.map((c) => (
                  <label key={c} className="chip cursor-pointer bg-highlight-low font-mono text-subtle">
                    <input type="checkbox" className="accent-iris" checked={channels.includes(c)} onChange={() => toggleChannel(c)} />
                    {channelName(c)}
                  </label>
                ))}
              </div>
            </fieldset>
          </section>

          <section className="card p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h2 className="card-title">Top-5 candidates</h2>
              <div role="radiogroup" aria-label="Ranking stage" className="flex rounded-xl border border-highlight-med p-0.5 text-xs">
                {(["post", "pre"] as const).map((v) => (
                  <button
                    key={v}
                    type="button"
                    role="radio"
                    aria-checked={view === v}
                    onClick={() => setView(v)}
                    className={`rounded-lg px-3 py-1 ${view === v ? "bg-iris text-on-iris" : "text-subtle hover:text-text"}`}
                  >
                    {v === "post" ? "After Sinkhorn (submitted)" : info.cslsK ? "Before Sinkhorn (CSLS only)" : "Fused scores only"}
                  </button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-5">
              {row.slice(0, 5).map((id, k) => (
                <VideoCard
                  key={`${view}-${id}`}
                  id={id}
                  rank={k + 1}
                  score={prob(id)}
                  scoreLabel={view === "post" ? "p" : "score"}
                  isTruth={id === truth}
                />
              ))}
            </div>
            <p className="mt-3 text-xs text-subtle">
              {truth !== undefined ? (
                <>
                  Watched video: <span className="font-mono text-foam">{truth}</span>, ranked{" "}
                  <span className="font-mono text-gold">#{truthRank}</span> of 102 in this view
                  {acc && <> · {result.subject} top-5 {pct(acc.top5)}</>}.
                </>
              ) : (
                "No labels were given for this person: the watched videos are unknown, so correctness cannot be shown."
              )}
            </p>
          </section>

          <SpectrumPanel rec={rec} subject={result.subject} trial={trial} channels={channels} />

          <div className="grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
            <section className="card p-4" aria-labelledby="top10-title">
              <div className="mb-2 flex items-baseline justify-between">
                <h2 id="top10-title" className="card-title">Top 10 ranking</h2>
                <span className="text-[11px] text-subtle">of 102 candidates</span>
              </div>
              <table className="w-full text-left text-xs">
                <thead className="text-[11px] uppercase tracking-wider text-subtle">
                  <tr>
                    <th className="py-1.5 pr-2">Rank</th>
                    <th className="py-1.5 pr-2">Video</th>
                    <th className="py-1.5 pr-2 text-right">{view === "post" ? "p" : "score"}</th>
                    <th className="py-1.5 text-right">Preview</th>
                  </tr>
                </thead>
                <tbody>
                  {row.slice(0, 10).map((id, k) => (
                    <tr key={id} className={`border-t border-highlight-med/60 ${id === truth ? "bg-foam/10" : ""}`}>
                      <td className="py-1.5 pr-2 font-mono">{k + 1}</td>
                      <td className="py-1.5 pr-2 font-mono">
                        {id}
                        {id === truth && <span className="chip ml-2 bg-foam text-(--color-base)">watched</span>}
                      </td>
                      <td className="py-1.5 pr-2 text-right font-mono">{prob(id).toFixed(3)}</td>
                      <td className="py-1 text-right">
                        <video
                          src={`${videoUrl(id)}#t=0.5`}
                          className="ml-auto h-9 w-9 rounded object-cover"
                          muted
                          playsInline
                          preload="metadata"
                          aria-label={`Preview of video ${id}`}
                          onMouseEnter={(e) => void e.currentTarget.play().catch(() => {})}
                          onMouseLeave={(e) => e.currentTarget.pause()}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section className="card p-4" aria-labelledby="stats-title">
              <h2 id="stats-title" className="card-title">Inference stats</h2>
              <dl className="mt-2 grid gap-2 text-xs">
                <StatRow k="Model" v={info.name} />
                <StatRow k="Pipeline" v={info.formula} />
                <StatRow k="Scorers" v={info.scorers} />
                <StatRow
                  k="In-browser post-processing"
                  v={`${(result.timingsMs.postprocess ?? 0).toFixed(1)} ms for ${result.n} recordings`}
                />
                <StatRow k="Candidates" v="102 test videos" />
                <StatRow k="Input" v="17 channels × 3 s at 500 Hz" />
                <StatRow
                  k={`${result.subject} dev-test accuracy`}
                  v={result.postAcc ? `top-1 ${pct(result.postAcc.top1)} · top-5 ${pct(result.postAcc.top5)}` : "not available (no labels)"}
                />
              </dl>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}

function Verdict({ rank, unlabelled }: { rank: number | null; unlabelled: boolean }) {
  if (unlabelled || rank === null) return <span className="chip bg-overlay text-subtle">no labels</span>;
  if (rank === 1) return <span className="chip bg-foam text-(--color-base)">Correct · rank 1</span>;
  if (rank <= 5) return <span className="chip bg-iris/20 text-iris">True video at rank {rank}</span>;
  return <span className="chip border border-love/60 bg-love/15 text-text">True video at rank {rank}</span>;
}

function StatRow({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg bg-highlight-low px-3 py-2">
      <dt className="text-[11px] text-subtle">{k}</dt>
      <dd className="mt-0.5 text-text">{v}</dd>
    </div>
  );
}

function RankBadge({ rank }: { rank: number }) {
  const cls = rank === 1 ? "bg-foam/20 text-foam" : rank <= 5 ? "bg-iris/20 text-iris" : "bg-highlight-low text-subtle";
  return <span className={`chip w-10 justify-center font-mono ${cls}`} title={`true video ranked ${rank}`}>{rank}</span>;
}

function Stat({ label, value, sub, tone = "text" }: { label: string; value: string; sub: string; tone?: string }) {
  const color = { text: "text-text", foam: "text-foam", iris: "text-iris", gold: "text-gold", love: "text-gold" }[tone];
  return (
    <div className="card p-4">
      <p className="text-[11px] uppercase tracking-wider text-subtle">{label}</p>
      <p className={`mt-1 font-mono text-2xl font-semibold ${color}`}>{value}</p>
      <p className="mt-1 text-[11px] text-subtle">{sub}</p>
    </div>
  );
}
