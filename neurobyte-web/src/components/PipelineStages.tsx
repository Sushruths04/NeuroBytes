import { useTheme } from "../lib/themeMode";
import { type Stage } from "../lib/pipeline";

interface Props {
  stages: Stage[];
  index: number; // active stage; stages.length when finished
  progress: number;
  timingsMs?: Record<string, number>;
}

export function PipelineStages({ stages, index, progress }: Props) {
  return (
    <ol className="space-y-1.5" aria-label="Pipeline stages">
      {stages.map((s, i) => {
        const state = i < index ? "done" : i === index ? "active" : "pending";
        return (
          <li
            key={s.id}
            aria-current={state === "active" ? "step" : undefined}
            className={`relative overflow-hidden rounded-xl border px-3 py-2 transition ${
              state === "active"
                ? "border-iris/70 bg-overlay"
                : state === "done"
                  ? "border-highlight-med bg-surface"
                  : "border-highlight-low bg-surface/40 opacity-60"
            }`}
          >
            {state === "active" && (
              <span
                aria-hidden
                className="absolute inset-y-0 left-0 bg-iris/10"
                style={{ width: `${Math.round(progress * 100)}%` }}
              />
            )}
            <div className="relative flex items-start gap-3">
              <StatusIcon state={state} />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium">{s.title}</span>
                  <span
                    className={`chip ${s.kind === "live" ? "bg-foam/15 text-foam" : "bg-gold/15 text-gold"}`}
                    title={
                      s.kind === "live"
                        ? "Executed in your browser on the uploaded file"
                        : "Ran offline in Python; the browser loads its output (the fused score matrix)"
                    }
                  >
                    {s.kind === "live" ? "in browser" : "precomputed"}
                  </span>
                </div>
                <p className={`mt-0.5 text-xs ${state === "active" ? "text-text/90" : "text-subtle"}`}>{s.detail}</p>
              </div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function StatusIcon({ state }: { state: "done" | "active" | "pending" }) {
  const { t } = useTheme();
  if (state === "done")
    return (
      <svg viewBox="0 0 20 20" className="mt-0.5 h-5 w-5 shrink-0 text-pine" aria-label="done">
        <circle cx="10" cy="10" r="9" fill="currentColor" opacity="0.25" />
        <path d="M6 10.5l2.5 2.5L14 7.5" stroke={t.foam} strokeWidth="2" fill="none" strokeLinecap="round" />
      </svg>
    );
  if (state === "active")
    return (
      <svg viewBox="0 0 20 20" className="mt-0.5 h-5 w-5 shrink-0 animate-spin text-iris" aria-label="running">
        <circle cx="10" cy="10" r="8" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2.5" fill="none" />
        <path d="M10 2a8 8 0 0 1 8 8" stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="round" />
      </svg>
    );
  return (
    <svg viewBox="0 0 20 20" className="mt-0.5 h-5 w-5 shrink-0 text-muted" aria-label="pending">
      <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="2" fill="none" />
    </svg>
  );
}
