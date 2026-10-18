import { useEffect, useState, type ReactNode } from "react";
import type { ModelId } from "../content/models";
import { DEV_SUBJECTS, loadDevRanks, personStats, rankHistogram } from "../lib/rankStats";
import { useTheme } from "../lib/themeMode";

type State = { model: ModelId; per: Record<string, number[]> } | { model: ModelId; error: string } | null;

/**
 * "Where the true video lands" and "Per person" (as on the static site), computed
 * from the selected model's ranking pooled over the labelled people sub-01..04.
 */
export function RankCharts({ model, subject }: { model: ModelId; subject: string }) {
  const [state, setState] = useState<State>(null);
  useEffect(() => {
    let live = true;
    loadDevRanks(model).then(
      (per) => live && setState({ model, per }),
      (e: unknown) => live && setState({ model, error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      live = false;
    };
  }, [model]);

  const ready = state && state.model === model ? state : null;
  const sub = `${model} · dev test, sub-01..04 (408 recordings)`;
  const body = (chart: (per: Record<string, number[]>) => ReactNode) =>
    !ready ? (
      <p className="mt-3 text-xs text-subtle" role="status">loading…</p>
    ) : "error" in ready ? (
      <p className="mt-3 text-xs text-subtle" role="alert">Couldn't load the dev-test rankings: {ready.error}</p>
    ) : (
      chart(ready.per)
    );

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <section className="card min-w-0 p-4" aria-labelledby="rankhist-title">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="rankhist-title" className="card-title">Where the true video lands</h2>
          <span className="text-[11px] text-subtle">{sub}</span>
        </div>
        {body((per) => <RankHist ranks={DEV_SUBJECTS.flatMap((s) => per[s])} />)}
      </section>
      <section className="card min-w-0 p-4" aria-labelledby="persub-title">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="persub-title" className="card-title">Per person</h2>
          <span className="text-[11px] text-subtle">{sub}</span>
        </div>
        {body((per) => <PerPerson per={per} current={subject} />)}
        {(subject === "sub-05" || subject === "sub-06") && (
          <p className="mt-2 text-[11px] text-subtle">sub-05 and sub-06: labels private, not scored.</p>
        )}
      </section>
    </div>
  );
}

function RankHist({ ranks }: { ranks: number[] }) {
  const { t } = useTheme();
  const h = rankHistogram(ranks);
  const mx = Math.max(...h.map((b) => b.pct), 1);
  const W = 460, H = 266;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 w-full" role="img"
      aria-label={`Rank of the true video among 102: ${h.map((b) => `${b.label}: ${b.pct.toFixed(1)}%`).join(", ")}.`}>
      {h.map((b, i) => {
        const x = 30 + i * 72, hh = (190 * b.pct) / mx;
        const fill = i === 0 ? t.foam : i < 3 ? t.iris : t["highlight-high"];
        return (
          <g key={b.label} data-bar={b.label}>
            <rect x={x} y={220 - hh} width="56" height={hh} rx="6" fill={fill} />
            <text x={x + 28} y={214 - hh} textAnchor="middle" fontSize="12" fontWeight={600} fill={t.text} fontFamily="ui-monospace, monospace">{b.pct.toFixed(1)}%</text>
            <text x={x + 28} y="240" textAnchor="middle" fontSize="12" fill={t.subtle}>{b.label}</text>
          </g>
        );
      })}
      <text x="30" y="260" fontSize="11" fill={t.subtle}>rank of the true video among 102</text>
    </svg>
  );
}

function PerPerson({ per, current }: { per: Record<string, number[]>; current: string }) {
  const { t } = useTheme();
  const rows = DEV_SUBJECTS.map((s) => personStats(s, per[s]));
  const W = 480, H = 24 + rows.length * 56;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 w-full" role="img"
      aria-label={rows.map((r) => `${r.sub}: top-5 ${r.top5.toFixed(1)}%, top-1 ${r.top1.toFixed(1)}%`).join("; ")}>
      {rows.map((r, i) => {
        const y = 14 + i * 56, w5 = (330 * r.top5) / 100, w1 = (330 * r.top1) / 100, cur = r.sub === current;
        return (
          <g key={r.sub}>
            <text x="0" y={y + 25} fontSize="12" fontWeight={cur ? 700 : 500} fill={cur ? t.gold : t.text} fontFamily="ui-monospace, monospace">{r.sub}</text>
            <rect x="70" y={y} width={w5} height="18" rx="5" fill={t.foam} opacity="0.5" />
            <rect x="70" y={y + 22} width={w1} height="18" rx="5" fill={t.iris} />
            <text x={76 + w5} y={y + 14} fontSize="11" fill={t.text}>{r.top5.toFixed(1)}% top-5</text>
            <text x={76 + w1} y={y + 36} fontSize="11" fill={t.text}>{r.top1.toFixed(1)}% top-1</text>
          </g>
        );
      })}
    </svg>
  );
}
