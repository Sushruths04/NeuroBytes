import { useEffect, useMemo, useState } from "react";
import { loadView, saveView } from "../lib/session";
import { writeNpyInt64 } from "../lib/npy";
import { channelName } from "../lib/montage";
import { N_CHANNELS, type Recording } from "../lib/eeg";
import { type PipelineResult } from "../lib/pipeline";
import { modelInfo } from "../content/models";
import { EegCanvas } from "./EegCanvas";
import { SpectrumPanel } from "./AnalyticsView";
import { RankCharts } from "./RankCharts";
import { captionOf, fetchCaptions, videoUrl } from "../lib/api";
import { inferenceSummary, verdictOf } from "../lib/inference";

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
  const scoreLabel = view === "post" ? "p" : "score";
  // Candidate shown in the player (index into row); back to rank 1 when the recording, person or view changes.
  const selKey = `${result.subject}|${result.model}|${trial}|${view}`;
  const [sel, setSel] = useState({ key: selKey, i: 0 });
  const select = (i: number) => setSel({ key: selKey, i });
  const sum = inferenceSummary(row, prob, truth, sel.key === selKey ? sel.i : 0);
  // Qwen2.5-VL captions, display only; {} if unavailable.
  const [captions, setCaptions] = useState<Record<number, string>>({});
  useEffect(() => {
    let live = true;
    void fetchCaptions().then((c) => live && setCaptions(c));
    return () => {
      live = false;
    };
  }, []);
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

      {/* predicted video + top 5 (layout of the static site's Inference tab) */}
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <section className="card min-w-0 p-4" aria-labelledby="pred-title">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h2 id="pred-title" className="card-title">Predicted video</h2>
            <Verdict rank={truthRank} unlabelled={!result.isDevSubject} />
          </div>
          <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-highlight-low">
            <video
              key={`${view}-${sum.selectedId}`}
              src={videoUrl(sum.selectedId)}
              className="h-full w-full object-contain animate-fade-in"
              muted
              loop
              playsInline
              autoPlay
              controls
              preload="auto"
              aria-label={`Video ${sum.selectedId}, rank ${sum.selectedRank} of 102`}
            />
            <span className="chip pointer-events-none absolute left-3 top-3 bg-base/80 text-gold">Rank #{sum.selectedRank}</span>
            <span className="chip pointer-events-none absolute right-3 top-3 bg-base/80 font-mono text-text">
              {scoreLabel} {sum.selectedScore.toFixed(3)}
            </span>
          </div>
          <div className="mt-3 grid gap-3 sm:grid-cols-[auto_minmax(0,1fr)]">
            <div>
              <p className="text-[11px] text-subtle">
                {sum.selectedRank === 1 ? "Predicted top-1 video" : `Selected candidate · rank ${sum.selectedRank}`}
              </p>
              <p className="font-mono text-lg font-semibold text-text">{sum.selectedId}</p>
            </div>
            <div className="min-w-0">
              <p className="text-[11px] text-subtle">
                Description <span className="opacity-70">(caption generated by Qwen2.5-VL)</span>
              </p>
              <Caption text={captionOf(captions, sum.selectedId)} className="text-sm text-text" />
            </div>
          </div>
          {truth !== undefined && result.isDevSubject ? (
            <div className="mt-3 flex items-center gap-3 rounded-xl border border-foam/50 bg-foam/10 p-3">
              <video
                key={`truth-${truth}`}
                src={videoUrl(truth)}
                className="h-16 w-16 shrink-0 rounded-lg object-cover"
                muted
                loop
                playsInline
                autoPlay
                preload="metadata"
                aria-label={`Ground-truth video ${truth}`}
              />
              <div className="min-w-0 text-xs">
                <p className="text-[11px] text-subtle">Ground truth: video actually watched (organizer label)</p>
                <p className="mt-0.5 text-text">
                  <span className="font-mono font-semibold text-foam">{truth}</span> ·{" "}
                  <Caption text={captionOf(captions, truth)} />
                </p>
                <p className="mt-0.5 text-[11px] text-subtle">Caption generated by Qwen2.5-VL</p>
              </div>
            </div>
          ) : (
            <p className="mt-3 rounded-xl bg-highlight-low p-3 text-xs text-subtle">
              sub-05/06 are the scored people: their labels are not public, so no correctness is shown.
            </p>
          )}
          <p className="mt-2 text-[11px] text-subtle">
            Ground-truth video IDs come from the organizers' labels (sub-01..04). The one-line captions of all videos (predicted and ground truth) were generated by Qwen2.5-VL from the videos, only to make this page readable; the model does not use them.
          </p>
        </section>

        <section className="card min-w-0 p-4" aria-labelledby="top5-title">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h2 id="top5-title" className="card-title">Top 5</h2>
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
          <ul className="grid grid-cols-1 gap-2" aria-label="Top 5 candidates">
            {sum.top5.map((c, k) => (
              <li key={`${view}-${c.id}`}>
                <button
                  type="button"
                  aria-pressed={k === sum.selectedRank - 1}
                  aria-label={`Rank ${c.rank}, video ${c.id}${c.isTruth ? ", watched video" : ""}, ${scoreLabel} ${c.score.toFixed(3)}`}
                  onClick={() => select(k)}
                  className={`flex w-full items-center gap-3 rounded-xl border p-2 text-left transition ${
                    k === sum.selectedRank - 1
                      ? "border-iris bg-overlay"
                      : c.isTruth
                        ? "border-foam/60 bg-foam/10 hover:bg-overlay/60"
                        : "border-highlight-med hover:bg-overlay/60"
                  }`}
                >
                  <span className="w-5 shrink-0 text-center font-mono text-sm text-gold">{c.rank}</span>
                  <video
                    src={`${videoUrl(c.id)}#t=0.5`}
                    className="h-12 w-12 shrink-0 rounded-lg bg-highlight-low object-cover"
                    muted
                    playsInline
                    preload="metadata"
                    aria-hidden
                    tabIndex={-1}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block font-mono text-xs font-semibold text-text">
                      Video {c.id}
                      {c.isTruth && <span className="text-foam"> ✓</span>}
                    </span>
                    <Caption text={captionOf(captions, c.id)} className="block truncate text-[11px] text-subtle" />
                  </span>
                  <span className="shrink-0 font-mono text-xs text-text">{c.score.toFixed(3)}</span>
                </button>
              </li>
            ))}
          </ul>
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
      </div>

      <section className="card p-4" aria-labelledby="top10-title">
        <div className="mb-2 flex items-baseline justify-between">
          <h2 id="top10-title" className="card-title">Top 10 ranking</h2>
          <span className="text-[11px] text-subtle">of 102 candidates</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[34rem] text-left text-xs">
            <thead className="text-[11px] uppercase tracking-wider text-subtle">
              <tr>
                <th className="py-1.5 pr-2">Rank</th>
                <th className="py-1.5 pr-2">Video</th>
                <th className="py-1.5 pr-2">Caption (generated by Qwen2.5-VL)</th>
                <th className="py-1.5 pr-2 text-right">{scoreLabel}</th>
                <th className="py-1.5 text-right">Preview</th>
              </tr>
            </thead>
            <tbody>
              {sum.top10.map((c, k) => (
                <tr
                  key={c.id}
                  className={`border-t border-highlight-med/60 ${c.isTruth ? "bg-foam/10" : ""} ${k === sum.selectedRank - 1 ? "bg-overlay" : ""}`}
                >
                  <td className="py-1.5 pr-2 font-mono">{c.rank}</td>
                  <td className="whitespace-nowrap py-1.5 pr-2 font-mono">
                    {k < 5 ? (
                      <button type="button" className="font-mono hover:text-iris hover:underline" onClick={() => select(k)} aria-label={`Show video ${c.id} in the player`}>
                        {c.id}
                      </button>
                    ) : (
                      c.id
                    )}
                    {c.isTruth && <span className="chip ml-2 bg-foam text-(--color-base)">watched</span>}
                  </td>
                  <td className="py-1.5 pr-2 text-text">
                    {captionOf(captions, c.id) ?? <span className="text-subtle">—</span>}
                  </td>
                  <td className="py-1.5 pr-2 text-right font-mono">{c.score.toFixed(3)}</td>
                  <td className="py-1 text-right">
                    <video
                      src={`${videoUrl(c.id)}#t=0.5`}
                      className="ml-auto h-9 w-9 rounded object-cover"
                      muted
                      playsInline
                      preload="metadata"
                      aria-label={`Preview of video ${c.id}`}
                      onMouseEnter={(e) => void e.currentTarget.play().catch(() => {})}
                      onMouseLeave={(e) => e.currentTarget.pause()}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* pooled dev-test charts, as on the static site */}
      <RankCharts model={result.model} subject={result.subject} />

      <SpectrumPanel rec={rec} subject={result.subject} trial={trial} channels={channels} />

      <section className="card min-w-0 p-4" aria-labelledby="eeg-title">
        {/* EEG plot and the recordings list share one card, directly before Inference stats */}
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h2 id="eeg-title" className="card-title">
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
        <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_240px]">
          <div className="min-w-0">
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
          </div>
          <div className="flex max-h-80 flex-col rounded-xl border border-highlight-med p-2 lg:max-h-[400px]">
            <h3 id="trials-title" className="card-title px-1 pb-2">Recordings</h3>
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
          </div>
        </div>
      </section>

      {/* inference stats, last card on the page */}
      <section className="card p-4" aria-labelledby="stats-title">
        <h2 id="stats-title" className="card-title">Inference stats</h2>
        <dl className="mt-2 grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-4">
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
  );
}

function Verdict({ rank, unlabelled }: { rank: number | null; unlabelled: boolean }) {
  const v = verdictOf(rank, unlabelled);
  const cls = {
    correct: "bg-foam text-(--color-base)",
    near: "bg-iris/20 text-iris",
    miss: "border border-love/60 bg-love/15 text-text",
    private: "bg-overlay text-subtle",
  }[v.kind];
  return <span className={`chip ${cls}`}>{v.text}</span>;
}

function Caption({ text, className = "" }: { text: string | null; className?: string }) {
  return text ? <span className={className}>{text}</span> : <span className={`${className} text-subtle italic`}>caption unavailable</span>;
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
