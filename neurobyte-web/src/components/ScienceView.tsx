import { useTheme } from "../lib/themeMode";
import { BENCHMARKS, CITATIONS, CLAIMS, type Citation } from "../content/science";
import { href } from "../lib/router";
import { LATENCY } from "../content/team";

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
          <b className="text-text">C13</b> ranks the watched video first for <b className="text-text">36.03%</b> of recordings
          and in the top 5 for <b className="text-text">67.16%</b> (chance: 0.98% / 4.90%), on sub-01..04. Each design choice
          was kept only if it improved cross-validation on held-out training videos, and each rests on published findings.
        </p>
      </section>

      <section aria-labelledby="claims-title" className="grid gap-3">
        <h2 id="claims-title" className="text-xl font-semibold">Our claims and the research behind them</h2>
        <div className="card divide-y divide-highlight-med/60">
          {CLAIMS.map((c, i) => (
            <article key={c.claim} className="grid gap-2 p-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)] md:gap-6">
              <h3 className="flex gap-3 text-[0.98rem] font-semibold text-text">
                <span className="font-mono text-iris">{i + 1}</span>
                {c.claim}
              </h3>
              <p className="text-sm leading-relaxed text-subtle">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-gold">We measured</span>
                {c.measured}
              </p>
              <p className="text-sm leading-relaxed text-subtle">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-gold">Research</span>
                {c.research}<Cite ids={c.cites} />
              </p>
            </article>
          ))}
        </div>
      </section>

      <Latency />

      <section aria-labelledby="bench-title">
        <h2 id="bench-title" className="text-xl font-semibold">Where this sits against published work</h2>
        <p className="mt-1 text-sm text-subtle">
          The comparison with NICE is not like-for-like: a different dataset, 102 candidates rather than 200, and video rather
          than still images. It is a check on the order of magnitude. More recent contrastive systems report higher numbers on
          the image task (NeuroCLIP: 63.2% top-1, 200-way)<Cite ids={["neuroclip"]} />.
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
    </main>
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
              </p>
              <p className="mt-0.5 text-subtle">{c.role}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
