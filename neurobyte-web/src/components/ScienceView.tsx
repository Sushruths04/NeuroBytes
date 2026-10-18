import { useTheme } from "../lib/themeMode";
import { useState } from "react";
import {
  BENCHMARKS,
  CITATIONS,
  KEY_CHOICES,
  PROGRESSION,
  SUPPORTING,
  type Citation,
  type Finding,
} from "../content/science";
import { href } from "../lib/router";
import { ERP_TIMELINE, LATENCY, TEAM_SCIENCE, type ScienceCard } from "../content/team";

const byId = new Map(CITATIONS.map((c, i) => [c.id, { ...c, n: i + 1 }]));

function Cite({ ids }: { ids: string[] }) {
  return (
    <span className="whitespace-nowrap">
      {ids.map((id) => {
        const c = byId.get(id);
        if (!c) return null;
        return (
          <a
            key={id}
            href={`#ref-${id}`}
            onClick={(e) => {
              e.preventDefault();
              document.getElementById(`ref-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
            }}
            className="ml-0.5 align-super text-[10px] font-semibold text-iris hover:underline"
            aria-label={`Reference ${c.n}: ${c.title}`}
          >
            [{c.n}]
          </a>
        );
      })}
    </span>
  );
}

export function ScienceView() {
  return (
    <main className="mx-auto grid max-w-5xl grid-cols-1 gap-10 pt-8">
      <section>
        <a href={href("home")} className="text-xs text-iris hover:underline">← Back to home</a>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">Why it works</h1>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-subtle">
          The final system, <b className="text-text">C13</b>, gets the watched video into its top 5 for{" "}
          <b className="text-text">67.16%</b> of recordings and ranks it first for <b className="text-text">36.03%</b> (chance:
          4.90% / 0.98%), on sub-01..04. It combines two pipelines: <b className="text-text">winfuse</b> (509 ridge scorers,
          30.64% / 60.54% on its own) and <b className="text-text">mindread</b> (77 feature scorers, a T2 encoder and two
          contrastive networks). Most of the accuracy came from asking what the brain signal actually is and how the task is
          built, not from bigger models.
        </p>
      </section>

      <section aria-labelledby="team-title" className="grid gap-4">
        <h2 id="team-title" className="text-xl font-semibold">What the brain signal can and cannot tell us</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {TEAM_SCIENCE.map((c) => <TeamCard key={c.title} c={c} />)}
        </div>
      </section>

      <Latency />

      <section className="grid gap-1">
        <h2 className="text-xl font-semibold">Inside winfuse: the design choices, with citations</h2>
        <p className="text-sm text-subtle">
          The numbers below are for winfuse on its own (submission2), before it was combined into C10–C13.
        </p>
      </section>

      <Progression />

      <section aria-labelledby="key-title" className="grid gap-4">
        <h2 id="key-title" className="text-xl font-semibold">The four choices that made the difference</h2>
        {KEY_CHOICES.map((f, i) => <FindingCard key={f.id} f={f} index={i + 1} />)}
      </section>

      <section aria-labelledby="sup-title" className="grid gap-4">
        <h2 id="sup-title" className="text-xl font-semibold">Choosing the video features</h2>
        {SUPPORTING.map((f) => <FindingCard key={f.id} f={f} />)}
      </section>

      <section aria-labelledby="bench-title">
        <h2 id="bench-title" className="text-xl font-semibold">Where this sits against published work</h2>
        <p className="mt-1 text-sm text-subtle">
          C13, the latest pipeline, against standalone Winfuse, the reference baseline for this data and the closest published system. This is not a like-for-like comparison: it is a different dataset, 102 candidates rather than 200, and video rather than still images. It is a sanity check on the order of magnitude.
          More recent contrastive systems report higher numbers on the image task (NeuroCLIP: 63.2% top-1, 200-way)<Cite ids={["neuroclip"]} />.
        </p>
        <div className="card mt-4 overflow-x-auto">
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead className="text-[11px] uppercase tracking-wider text-subtle">
              <tr><th className="px-4 py-3">System</th><th className="px-4 py-3">Task</th><th className="px-4 py-3 text-right">Top-1</th><th className="px-4 py-3 text-right">Top-5</th></tr>
            </thead>
            <tbody>
              {BENCHMARKS.map((b) => (
                <tr key={b.system} className={`border-t border-highlight-med/60 ${b.highlight ? "bg-iris/10" : ""}`}>
                  <td className={`px-4 py-2.5 ${b.highlight ? "font-semibold text-iris" : ""}`}>{b.system}{b.cite && <Cite ids={[b.cite]} />}</td>
                  <td className="px-4 py-2.5 text-subtle">{b.task}{"taskCite" in b && b.taskCite && <Cite ids={[b.taskCite]} />}</td>
                  <td className="px-4 py-2.5 text-right font-mono">{b.top1}</td>
                  <td className="px-4 py-2.5 text-right font-mono">{b.top5}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <References />

      <p className="pb-4 text-[11px] text-subtle">
        Source: <code className="font-mono">submission2/WHY_IT_WORKS.md</code> (full detail, including where these results disagree with the mindread pipeline).
        Every number on this page is taken from that document or from <code className="font-mono">submission2/README.md</code>.
      </p>
    </main>
  );
}

function TeamCard({ c }: { c: ScienceCard }) {
  return (
    <article className={`card p-5 ${c.big ? "md:col-span-2" : ""}`}>
      <p className="font-mono text-2xl font-semibold text-iris">{c.num}</p>
      <h3 className="mt-1 text-[1.05rem] font-semibold text-text">{c.title}</h3>
      {c.paragraphs.map((p) => (
        <p key={p} className="mt-2 text-sm leading-relaxed text-subtle">{p}</p>
      ))}
      {c.num === "800 ms" && <ErpTimeline />}
      {c.tiles && (
        <div className="mt-3 grid gap-2 sm:grid-cols-3">
          {c.tiles.map(([k, v]) => (
            <div key={k} className="rounded-xl bg-highlight-low px-3 py-2.5">
              <p className="text-sm font-semibold text-text">{k}</p>
              <p className="mt-0.5 text-xs text-subtle">{v}</p>
            </div>
          ))}
        </div>
      )}
      {c.after && <p className="mt-3 text-sm leading-relaxed text-subtle">{c.after}</p>}
      {c.bullets && (
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-subtle">
          {c.bullets.map((b) => <li key={b}>{b}</li>)}
        </ul>
      )}
    </article>
  );
}

/** #22: what each latency band of the visual response carries (team site). */
function ErpTimeline() {
  return (
    <ol className="mt-3 grid gap-2 sm:grid-cols-5" aria-label="Visual ERP timeline">
      {ERP_TIMELINE.map(([t, d], i) => (
        <li
          key={t}
          className={`rounded-xl px-3 py-2.5 ${i === ERP_TIMELINE.length - 1 ? "border border-dashed border-highlight-high" : "bg-highlight-low"}`}
        >
          <p className="font-mono text-sm font-semibold text-text">{t}</p>
          <p className="mt-0.5 text-xs text-subtle">{d}</p>
        </li>
      ))}
    </ol>
  );
}

/** #23: mean rank of the true video by latency window (team's sweep). Lower is better; chance is 51.5. */
function Latency() {
  const { t } = useTheme();
  const W = 720, H = 240, L = 44, B = 190, T = 16, R = W - 12;
  const max = 60;
  const bw = (R - L) / LATENCY.length;
  const Y = (v: number) => B - ((B - T) * v) / max;
  const best = Math.min(...LATENCY.map(([, v]) => v));
  return (
    <section className="card p-5" aria-labelledby="lat-title">
      <h2 id="lat-title" className="card-title">Where in time the signal is</h2>
      <p className="mt-1 text-xs text-subtle">
        Mean rank of the true video (of 102) when decoding from one latency window only. Lower is better; the dashed line is chance (51.5).
      </p>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 w-full min-w-[520px]" role="img"
          aria-label={`Best window ${LATENCY.find(([, v]) => v === best)![0]} ms with mean rank ${best}; windows after 800 ms are near chance.`}>
          {[0, 20, 40, 60].map((v) => (
            <g key={v}>
              <line x1={L} x2={R} y1={Y(v)} y2={Y(v)} stroke={t["highlight-med"]} strokeDasharray="2 4" />
              <text x={L - 6} y={Y(v) + 4} textAnchor="end" fontSize="10" fill={t.subtle}>{v}</text>
            </g>
          ))}
          <line x1={L} x2={R} y1={Y(51.5)} y2={Y(51.5)} stroke={t.love} strokeWidth="1.5" strokeDasharray="6 4" />
          {LATENCY.map(([w, v], i) => {
            const x = L + i * bw + bw * 0.18, wd = bw * 0.64, isBest = v === best;
            return (
              <g key={w}>
                <rect x={x} y={Y(v)} width={wd} height={B - Y(v)} rx="5" fill={isBest ? t.foam : v < 40 ? t.iris : t["highlight-high"]} />
                <text x={x + wd / 2} y={Y(v) - 5} textAnchor="middle" fontSize="10" fill={t.text} fontFamily="ui-monospace, monospace">{v}</text>
                <text x={x + wd / 2} y={B + 16} textAnchor="middle" fontSize="10" fill={t.subtle}>{w}</text>
              </g>
            );
          })}
          <text x={(L + R) / 2} y={H - 6} textAnchor="middle" fontSize="11" fill={t.subtle}>window after clip onset (ms)</text>
        </svg>
      </div>
    </section>
  );
}

function FindingCard({ f, index }: { f: Finding; index?: number }) {
  const [open, setOpen] = useState(false);
  const panel = `finding-${f.id}`;
  return (
    <article className="card p-5">
      <div className="flex flex-wrap items-start gap-4">
        {index !== undefined && (
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-iris font-bold text-on-iris">{index}</span>
        )}
        <div className="min-w-0 flex-1">
          <h3 className="text-[1.05rem] font-semibold text-text">{f.title}</h3>
          <p className="mt-1.5 text-sm leading-relaxed text-text/90">{f.plain}</p>
        </div>
      </div>

      <div className="mt-4 rounded-xl border-l-2 border-gold/70 bg-base/40 px-4 py-3">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-gold">Why the research predicts this</p>
        <p className="mt-1 text-sm leading-relaxed text-subtle">{f.why}<Cite ids={f.cites} /></p>
      </div>

      <p className="mt-3 text-sm text-text/90"><span className="font-semibold text-iris">Takeaway: </span>{f.takeaway}</p>

      <button
        type="button"
        className="mt-3 text-xs text-iris hover:underline"
        aria-expanded={open}
        aria-controls={panel}
        onClick={() => setOpen((o) => !o)}
      >
        {open ? "Hide" : "Show"} measurements
      </button>
      {open && (
        <ul id={panel} className="mt-2 list-disc space-y-1 pl-5 font-mono text-[11px] leading-relaxed text-subtle animate-fade-in">
          {f.measured.map((m) => <li key={m}>{m}</li>)}
        </ul>
      )}
    </article>
  );
}

function Progression() {
  const { t } = useTheme();
  const min = 30, max = 62;
  const W = 720, H = 210, padL = 36, padB = 26, padT = 12;
  const padR = 44;
  const x = (i: number) => padL + (i / (PROGRESSION.length - 1)) * (W - padL - padR);
  const y = (v: number) => padT + (1 - (v - min) / (max - min)) * (H - padT - padB);
  const d = PROGRESSION.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.top5)}`).join(" ");
  return (
    <section className="card p-5" aria-labelledby="prog-title">
      <h2 id="prog-title" className="card-title">How the top-5 accuracy was built up</h2>
      <p className="mt-1 text-xs text-subtle">Pooled top-5 on sub-01..04 after each change (from submission2/README.md).</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 w-full" role="img"
        aria-label={`Top-5 rose from ${PROGRESSION[0].top5}% to ${PROGRESSION.at(-1)!.top5}%; the biggest jumps were latency windows and the Sinkhorn bijection.`}>
        {[30, 40, 50, 60].map((v) => (
          <g key={v}>
            <line x1={padL} x2={W - padR} y1={y(v)} y2={y(v)} stroke={t["highlight-med"]} strokeDasharray="2 4" />
            <text x={padL - 6} y={y(v) + 3} textAnchor="end" fontSize="10" fill={t.subtle}>{v}%</text>
          </g>
        ))}
        <path d={d} fill="none" stroke={t.iris} strokeWidth="2.5" />
        {PROGRESSION.map((p, i) => (
          <g key={i}>
            <circle cx={x(i)} cy={y(p.top5)} r={p.step ? 5 : 3} fill={p.step ? t.foam : t["highlight-high"]} />
            {p.step && (
              <>
                <text x={x(i)} y={y(p.top5) - 10} textAnchor="middle" fontSize="10" fill={t.text} fontFamily="ui-monospace, monospace">{p.top5}%</text>
                <text x={x(i)} y={H - 8} textAnchor="middle" fontSize="10" fill={t.subtle}>{p.step}</text>
              </>
            )}
          </g>
        ))}
      </svg>
    </section>
  );
}

function References() {
  return (
    <section aria-labelledby="refs-title">
      <h2 id="refs-title" className="text-xl font-semibold">References</h2>
      <ol className="mt-3 grid gap-2">
        {CITATIONS.map((c: Citation, i) => (
          <li id={`ref-${c.id}`} key={c.id} className="card flex gap-3 p-3 text-xs target:ring-2 target:ring-iris">
            <span className="font-mono text-iris">[{i + 1}]</span>
            <div className="min-w-0">
              <p className="text-text">
                {c.authors && <span>{c.authors} </span>}
                <a href={c.url} target="_blank" rel="noopener noreferrer" className="font-medium text-foam hover:underline">{c.title}</a>
                {c.venue && <span className="text-subtle">. {c.venue}</span>}
                {c.checked === "verified" && <span className="chip ml-2 bg-pine/25 text-foam">verified against full text</span>}
                {c.checked === "abstract" && <span className="chip ml-2 bg-overlay text-subtle">read at abstract level</span>}
              </p>
              <p className="mt-0.5 text-subtle">{c.role}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
