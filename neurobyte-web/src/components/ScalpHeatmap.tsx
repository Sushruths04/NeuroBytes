import { useEffect, useMemo, useRef } from "react";
import { CHANNELS, POSITIONS } from "../lib/montage";
import { useTheme } from "../lib/themeMode";

/**
 * Scalp heatmap (top view, nose up, cropped to the back of the head) of per-channel activity.
 *
 * Electrodes sit at the standard 10-10 positions of the organisers' channel names
 * (src/lib/montage.ts). Between electrodes the map is interpolated by inverse
 * distance weighting; further than MASK from every electrode it fades out, because
 * all 17 electrodes are parietal/occipital and anything painted over the front of
 * the head would be invented. Colour is relative: lowest channel at this moment =
 * coolest, highest = hottest.
 */
interface Props {
  /** activity per channel index, any scale (only the ordering/spread is shown) */
  activity: number[];
  label?: string;
}

const GRID = 72; // interpolation grid per side (upscaled smoothly by the browser)
const MASK = 0.3; // fade-out distance from the nearest electrode, head radius = 1
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const rgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

export function ScalpHeatmap({ activity, label }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { t } = useTheme();
  // cool → hot, from the theme: blue, violet, amber
  const stops = useMemo(() => [t.pine, t.iris, t.gold].map(rgb), [t]);

  // per-pixel interpolation weights depend only on the montage: compute once
  const weights = useMemo(() => {
    const W = new Float32Array(GRID * GRID * POSITIONS.length);
    const fade = new Float32Array(GRID * GRID);
    for (let gy = 0; gy < GRID; gy++)
      for (let gx = 0; gx < GRID; gx++) {
        const px = (gx / (GRID - 1)) * 2 - 1, py = 1 - (gy / (GRID - 1)) * 2;
        const cell = gy * GRID + gx;
        if (px * px + py * py > 1) continue;
        let sum = 0, dmin = Infinity;
        POSITIONS.forEach(([ex, ey], c) => {
          const d = Math.hypot(px - ex, py - ey);
          dmin = Math.min(dmin, d);
          const w = 1 / Math.max(1e-4, d) ** 2.5;
          W[cell * POSITIONS.length + c] = w;
          sum += w;
        });
        for (let c = 0; c < POSITIONS.length; c++) W[cell * POSITIONS.length + c] /= sum;
        fade[cell] = Math.max(0, 1 - Math.max(0, dmin - 0.06) / MASK);
      }
    return { W, fade };
  }, []);

  const lo = Math.min(...activity), hi = Math.max(...activity);
  const norm = activity.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5));
  const hottest = CHANNELS.filter((_, i) => norm[i] >= 0.999)[0];
  const coolest = CHANNELS.filter((_, i) => norm[i] <= 0.001)[0];

  useEffect(() => {
    const g = ref.current?.getContext("2d");
    if (!g) return;
    const img = g.createImageData(GRID, GRID);
    const n = POSITIONS.length;
    for (let cell = 0; cell < GRID * GRID; cell++) {
      const f = weights.fade[cell];
      if (!f) continue;
      let v = 0;
      for (let c = 0; c < n; c++) v += weights.W[cell * n + c] * norm[c];
      const s = Math.min(1.999, v * 2);
      const a = stops[Math.floor(s)], b = stops[Math.floor(s) + 1], k = s - Math.floor(s);
      img.data[cell * 4] = lerp(a[0], b[0], k);
      img.data[cell * 4 + 1] = lerp(a[1], b[1], k);
      img.data[cell * 4 + 2] = lerp(a[2], b[2], k);
      img.data[cell * 4 + 3] = 235 * f;
    }
    g.putImageData(img, 0, 0);
  });

  // SVG frame in the same -1..1 space as the grid (y flipped: nose up). All electrodes are
  // behind Cz, so the view is cropped to the back of the head (svg y from TOP to TOP + VH).
  const X = (x: number) => 50 + x * 40, Y = (y: number) => 50 - y * 40;
  const TOP = 40, VH = 54;
  return (
    <figure className="flex flex-col items-center">
      <div
        className="relative w-full max-w-[26rem] overflow-hidden"
        style={{ aspectRatio: `100 / ${VH}` }}
        role="img"
        aria-label={`Scalp heatmap: highest activity at ${hottest ?? "—"}, lowest at ${coolest ?? "—"}.`}
      >
        <canvas
          ref={ref}
          width={GRID}
          height={GRID}
          className="absolute rounded-full"
          style={{ left: "10%", width: "80%", top: `${((10 - TOP) / VH) * 100}%`, height: `${(80 / VH) * 100}%` }}
        />
        <svg viewBox={`0 ${TOP} 100 ${VH}`} className="absolute inset-0 h-full w-full" aria-hidden>
          {/* head, nose, ears */}
          <circle cx="50" cy="50" r="40" fill="none" stroke={t["highlight-high"]} strokeWidth="0.8" />
          <path d="M10 44 C6 46 6 54 10 56" fill="none" stroke={t["highlight-high"]} strokeWidth="0.8" />
          <path d="M90 44 C94 46 94 54 90 56" fill="none" stroke={t["highlight-high"]} strokeWidth="0.8" />
          <line x1="50" y1="49" x2="50" y2="51" stroke={t["highlight-high"]} strokeWidth="0.5" />
          <text x="50" y={TOP + 4} textAnchor="middle" fontSize="3" fill={t.muted}>↑ front of head (no electrodes) · Cz</text>
          {POSITIONS.map(([x, y], i) => (
            <g key={CHANNELS[i]}>
              <circle cx={X(x)} cy={Y(y)} r={norm[i] >= 0.999 ? 1.3 : 0.9} fill={t.text} stroke={t.base} strokeWidth="0.4" />
              <text x={X(x)} y={Y(y) - 1.8} textAnchor="middle" fontSize="2.4" fill={t.text} style={{ paintOrder: "stroke" }} stroke={t.base} strokeWidth="0.5">
                {CHANNELS[i]}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className="mt-2 flex w-full max-w-[16rem] items-center gap-2 text-[11px] text-muted" aria-hidden>
        <span>lower</span>
        <span className="h-1.5 flex-1 rounded-full" style={{ background: `linear-gradient(90deg, ${t.pine}, ${t.iris}, ${t.gold})` }} />
        <span>higher</span>
      </div>
      {label && <figcaption className="mt-2 text-center text-xs text-subtle">{label}</figcaption>}
    </figure>
  );
}
