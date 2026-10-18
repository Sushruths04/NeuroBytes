const SUBJECTS = [
  { id: "sub-01", dev: true },
  { id: "sub-02", dev: true },
  { id: "sub-03", dev: true },
  { id: "sub-04", dev: true },
  { id: "sub-05", dev: false },
  { id: "sub-06", dev: false },
];

interface Props {
  busy: boolean;
  error: string | null;
  /** the clicked button is passed so the page transition can start from it */
  onSelect: (sub: string, from: HTMLElement) => void;
  /** subject currently open, if any */
  current?: string | null;
}

/** Pick a person's test recording to rank. */
export function SubjectPicker({ busy, error, onSelect, current }: Props) {
  return (
    <section aria-labelledby="pick-title">
      <p id="pick-title" className="eyebrow">Choose a person</p>
      <p className="mt-2 text-sm font-light text-subtle">
        Each person's test set: 102 recordings × 17 channels × 3 s. For sub-01..04 the watched videos
        are known, so accuracy is shown; for sub-05/06 they were not given, so only the ranking is shown.
      </p>
      <div className="mt-4 grid grid-cols-3 gap-2 sm:grid-cols-6">
        {SUBJECTS.map((s) => (
          <button
            key={s.id}
            type="button"
            disabled={busy}
            aria-pressed={current === s.id}
            onClick={(e) => onSelect(s.id, e.currentTarget)}
            className={`flex flex-col items-center rounded-3xl border px-3 py-3 transition disabled:opacity-50 ${
              current === s.id ? "border-iris bg-iris text-on-iris" : "border-highlight-high text-text hover:border-subtle hover:bg-overlay"
            }`}
          >
            <span className="font-mono text-sm">{s.id}</span>
            <span className={`mt-0.5 text-[10px] uppercase tracking-[0.08em] ${current === s.id ? "" : s.dev ? "text-subtle" : "text-gold"}`}>
              {s.dev ? "labelled" : "no labels"}
            </span>
          </button>
        ))}
      </div>
      {busy && <p className="mt-3 text-xs text-subtle" role="status">Loading recording…</p>}
      {error && (
        <p role="alert" className="mt-3 rounded-2xl border border-love/60 bg-love/15 px-3 py-2 text-xs text-text">
          {error}
        </p>
      )}
    </section>
  );
}
