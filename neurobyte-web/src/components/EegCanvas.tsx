import { alpha } from "../theme";
import { useTheme } from "../lib/themeMode";
import { useEffect, useRef } from "react";
import { EPOCH_S, N_CHANNELS, N_SAMPLES, trialChannel, type Recording } from "../lib/eeg";
import { type Window } from "../content/models";
import { channelName } from "../lib/montage";

export const channelColors = (t: Record<string, string>) => [t.foam, t.iris, t.rose, t.gold, t.pine, t.love];

interface Props {
  rec: Recording;
  trial: number;
  /** cursor position in seconds; null hides it */
  cursor: number | null;
  /** latency windows to shade (the selected model's); widest first */
  windows?: Window[];
  /** channels to draw (default: all 17) */
  channels?: number[];
  /** 0..1 per channel; brightens traces (optional) */
  activity?: number[];
  height?: number;
}

/**
 * 17-channel stacked trace plot of one 3-second trial, labelled with the
 * electrode names the organisers supplied (src/lib/montage.ts).
 */
export function EegCanvas({ rec, trial, cursor, windows = [], channels, activity, height = 340 }: Props) {
  const { t } = useTheme();
  const ref = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const widthRef = useRef(0);
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(([e]) => {
      widthRef.current = e.contentRect.width;
      drawRef.current(); // latest draw, so a resize after a theme switch uses the current colours
    });
    ro.observe(el);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(draw);

  function draw() {
    const c = ref.current;
    const W = widthRef.current;
    if (!c || !W) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(W * dpr);
    c.height = Math.round(height * dpr);
    c.style.width = `${W}px`;
    c.style.height = `${height}px`;
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, height);

    const left = 40, right = 8, bottom = 22;
    const top = 6 + windows.length * 14; // one label row per window, above the traces
    const pw = W - left - right, ph = height - top - bottom;
    const x = (s: number) => left + (s / EPOCH_S) * pw;
    const shown = channels ?? Array.from({ length: N_CHANNELS }, (_, i) => i);
    const lane = ph / Math.max(1, shown.length);
    const colors = channelColors(t);

    // grid
    g.strokeStyle = alpha(t.muted, 0.25);
    g.lineWidth = 1;
    g.font = "10px ui-monospace, Menlo, monospace";
    g.fillStyle = t.subtle;
    for (let s = 0; s <= EPOCH_S + 1e-9; s += 0.5) {
      g.beginPath();
      g.moveTo(x(s) + 0.5, top);
      g.lineTo(x(s) + 0.5, top + ph);
      g.stroke();
      g.textAlign = "center";
      g.fillText(`${s.toFixed(1)}s`, x(s), height - 6);
    }

    // The model's latency window: an amber band (amber is the one accent the traces
    // barely use), solid edges on both sides and a filled label so it reads at a glance.
    windows.forEach((w, k) => {
      const a = x(w.from / 1000), b = x(w.to / 1000);
      g.fillStyle = alpha(t.gold, 0.16);
      g.fillRect(a, top, b - a, ph);
      g.strokeStyle = alpha(t.gold, 0.95);
      g.lineWidth = 1.5;
      g.beginPath();
      for (const e of [a, b]) {
        g.moveTo(Math.round(e) + 0.5, top);
        g.lineTo(Math.round(e) + 0.5, top + ph);
      }
      g.stroke();
      g.lineWidth = 1;
      const tw = g.measureText(w.label).width + 10;
      g.fillStyle = t.gold;
      g.beginPath();
      g.roundRect(a, 3 + k * 13, tw, 12, 6);
      g.fill();
      g.fillStyle = t.base;
      g.textAlign = "left";
      g.fillText(w.label, a + 5, 12.5 + k * 13);
    });

    // traces: min/max per pixel column so nothing aliases away
    const cols = Math.max(1, Math.floor(pw));
    shown.forEach((ch, row) => {
      const sig = trialChannel(rec, trial, ch);
      const mid = top + lane * (row + 0.5);
      const gain = (Math.min(lane, 60) * 0.42) / (2.5 * rec.channelStd[ch]);
      const a = activity?.[ch] ?? 0.6;
      g.strokeStyle = colors[ch % colors.length];
      g.globalAlpha = 0.45 + 0.55 * a;
      g.lineWidth = 1;
      g.beginPath();
      for (let px = 0; px < cols; px++) {
        const i0 = Math.floor((px / cols) * N_SAMPLES);
        const i1 = Math.max(i0 + 1, Math.floor(((px + 1) / cols) * N_SAMPLES));
        let lo = Infinity, hi = -Infinity;
        for (let k = i0; k < i1; k++) {
          const v = sig[k];
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        const yLo = mid - Math.max(-2.5, Math.min(2.5, lo / rec.channelStd[ch])) * gain * rec.channelStd[ch];
        const yHi = mid - Math.max(-2.5, Math.min(2.5, hi / rec.channelStd[ch])) * gain * rec.channelStd[ch];
        if (px === 0) g.moveTo(left + px, yLo);
        g.lineTo(left + px, yLo);
        g.lineTo(left + px, yHi);
      }
      g.stroke();
      g.globalAlpha = 1;
      g.fillStyle = t.subtle;
      g.textAlign = "right";
      g.fillText(channelName(ch), left - 6, mid + 3);
    });
    if (!shown.length) {
      g.fillStyle = t.subtle;
      g.textAlign = "center";
      g.fillText("No channels selected", left + pw / 2, top + ph / 2);
    }

    if (cursor !== null) {
      const cx = x(Math.min(Math.max(cursor, 0), EPOCH_S));
      const grad = g.createLinearGradient(cx - 40, 0, cx, 0);
      grad.addColorStop(0, alpha(t.foam, 0));
      grad.addColorStop(1, alpha(t.foam, 0.18));
      g.fillStyle = grad;
      g.fillRect(cx - 40, top, 40, ph);
      g.strokeStyle = t.foam;
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(cx, top);
      g.lineTo(cx, top + ph);
      g.stroke();
    }
  }

  return (
    <div ref={wrap} className="w-full">
      <canvas
        ref={ref}
        role="img"
        aria-label={`EEG trial ${trial + 1}: 17 channels over 3 seconds${cursor !== null ? `, cursor at ${cursor.toFixed(2)} s` : ""}`}
      />
    </div>
  );
}
