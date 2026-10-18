import { MODELS, type ModelId } from "../content/models";

interface Props {
  model: ModelId;
  onChange: (m: ModelId) => void;
  /** models with a score matrix for the current subject (others are disabled) */
  available?: string[] | null;
  compact?: boolean;
}

/** Choose which pipeline ranks the videos. C13 is the default. */
export function ModelPicker({ model, onChange, available, compact }: Props) {
  return (
    <label className={`flex min-w-0 items-center gap-2 ${compact ? "text-xs" : "text-sm"}`}>
      <span className="hidden text-subtle sm:inline">Model</span>
      <select
        value={model}
        onChange={(e) => onChange(e.target.value as ModelId)}
        aria-label="Model used for ranking"
        className="min-w-0 max-w-[11rem] rounded-lg border border-highlight-high bg-surface px-2 py-1 font-medium text-text focus-visible:outline-iris sm:max-w-none"
      >
        {MODELS.map((m) => (
          <option key={m.id} value={m.id} disabled={!!available && !available.includes(m.id)}>
            {m.name} · top-1 {m.frozen.top1.toFixed(1)}%
          </option>
        ))}
      </select>
    </label>
  );
}
