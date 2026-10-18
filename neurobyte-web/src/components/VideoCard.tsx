import { videoUrl } from "../lib/api";

interface Props {
  id: number | null;
  rank?: number;
  score?: number;
  /** "p" for post-Sinkhorn match probabilities, "score" for relative fused scores */
  scoreLabel?: string;
  isTruth?: boolean;
  caption?: string;
  autoPlay?: boolean;
  compact?: boolean;
  placeholder?: string;
}

/** One 3-second stimulus clip. Muted so browsers allow autoplay. */
export function VideoCard({ id, rank, score, scoreLabel = "p", isTruth, caption, autoPlay = false, compact = false, placeholder }: Props) {
  return (
    <div
      className={`overflow-hidden rounded-xl border bg-base/60 ${
        isTruth ? "border-foam ring-2 ring-foam/40" : "border-highlight-med"
      }`}
    >
      <div className="relative aspect-square w-full bg-highlight-low">
        {id !== null ? (
          <video
            key={id}
            src={videoUrl(id)}
            className="h-full w-full object-cover animate-fade-in"
            muted
            loop
            playsInline
            autoPlay={autoPlay}
            controls={!autoPlay && !compact}
            preload={autoPlay ? "auto" : "metadata"}
            aria-label={`Stimulus video ${id}`}
            onMouseEnter={compact ? (e) => void e.currentTarget.play().catch(() => {}) : undefined}
            onMouseLeave={compact ? (e) => e.currentTarget.pause() : undefined}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center p-4 text-center text-xs text-subtle">
            {placeholder ?? "waiting for scores…"}
          </div>
        )}
        {rank !== undefined && (
          <span className="chip absolute left-2 top-2 bg-base/80 text-gold">#{rank}</span>
        )}
        {isTruth && <span className="chip absolute right-2 top-2 bg-foam text-(--color-base)">watched</span>}
      </div>
      <div className={`flex items-center justify-between gap-2 ${compact ? "px-2 py-1.5" : "px-3 py-2"}`}>
        <span className="font-mono text-xs text-text">{id !== null ? `video ${id}` : "—"}</span>
        {score !== undefined && <span className="font-mono text-[11px] text-subtle">{scoreLabel}={score.toFixed(3)}</span>}
      </div>
      {caption && <p className="px-3 pb-2 text-[11px] text-subtle">{caption}</p>}
    </div>
  );
}
