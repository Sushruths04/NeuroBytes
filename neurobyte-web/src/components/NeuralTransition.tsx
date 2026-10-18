import { createContext, useCallback, useContext, useEffect, useRef, type ReactNode } from "react";
import { alpha, BRAIN_HUES_LIGHT } from "../theme";
import { useTheme } from "../lib/themeMode";

/**
 * Page transition in the landing page's language: particles stream out of the
 * thing you clicked into coloured strands that fill the screen (the page swaps
 * underneath while they cover it), then flow down and out to reveal the new page.
 *
 *   const transition = useNeuralTransition();
 *   transition(() => go("analysis"), event.currentTarget);
 *
 * The action may be async (e.g. loading a recording): the strands keep flowing,
 * covering the page, until it settles. Reduced motion: the action runs at once.
 */
type Style = "strands" | "slide";
type Run = (action: () => void | Promise<void>, from?: Element | null, opts?: { style?: Style }) => void;

const Ctx = createContext<Run>((action) => void action());
export const useNeuralTransition = () => useContext(Ctx);

const N = 520;
const STRANDS = 28;
const IN_S = 0.55; // particles stream in and cover the page
const OUT_S = 0.75; // strands flow down and out
const clamp = (v: number) => Math.min(1, Math.max(0, v));
const ease = (v: number) => v * v * (3 - 2 * v);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;

interface Particle { strand: number; u: number; delay: number; size: number }

export function NeuralTransitionHost({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { t, mode } = useTheme();
  const theme = useRef({ t, mode });
  theme.current = { t, mode };
  const busy = useRef(false);
  const particles = useRef<Particle[]>([]);
  if (!particles.current.length)
    particles.current = Array.from({ length: N }, (_, i) => ({
      strand: i % STRANDS,
      u: Math.random(),
      delay: Math.random() * 0.35,
      size: 0.8 + Math.random() * 1.4,
    }));

  /**
   * "slide": the page content slides off to the left (as if scrolling sideways) while
   * the brain spreads out into the background network; the next page slides in from
   * the right once it is ready.
   */
  const slide = useCallback(async (action: () => void | Promise<void>) => {
    const page = () => document.querySelector<HTMLElement>("[data-page-content]");
    const root = document.documentElement;
    busy.current = true;
    root.dataset.brainSpread = "1";
    try {
      const out = page()?.animate(
        [{ transform: "none", opacity: 1 }, { transform: "translateX(-45vw)", opacity: 0 }],
        { duration: 520, easing: "cubic-bezier(.55,0,.75,.45)", fill: "forwards" },
      );
      await out?.finished;
      try {
        await action();
      } finally {
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))); // let the new page render
        out?.cancel();
        const el = page();
        await el?.animate(
          [{ transform: "translateX(45vw)", opacity: 0 }, { transform: "none", opacity: 1 }],
          { duration: 620, easing: "cubic-bezier(.2,.7,.25,1)" },
        ).finished;
      }
    } finally {
      delete root.dataset.brainSpread;
      busy.current = false;
    }
  }, []);

  const run = useCallback<Run>((action, from, opts) => {
    const still = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (still || busy.current || !ref.current) return void action();
    if (opts?.style === "slide") return void slide(action);
    busy.current = true;
    const canvas = ref.current;
    const g = canvas.getContext("2d")!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = window.innerWidth, H = window.innerHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    canvas.style.display = "block";

    const r = from?.getBoundingClientRect();
    const ox = r ? r.left + r.width / 2 : W / 2, oy = r ? r.top + r.height / 2 : H * 0.3;
    const strandX = (k: number, y: number, time: number) =>
      ((k + 0.5) / STRANDS) * W + Math.sin(y * 0.01 + time * 1.6 + k * 0.7) * 10;

    let phase: "in" | "hold" | "out" = "in";
    let phaseStart = performance.now();
    const t0 = phaseStart;
    let done = false;
    let raf = 0;

    const finishAction = async () => {
      try {
        await action();
      } finally {
        // let the new page render a frame before uncovering it
        requestAnimationFrame(() => {
          phase = "out";
          phaseStart = performance.now();
        });
      }
    };

    const frame = (now: number) => {
      const { t: c, mode } = theme.current;
      const hues = mode === "dark" ? [c.iris, c.pine, c.foam] : BRAIN_HUES_LIGHT;
      const time = (now - t0) / 1000;
      const p = (now - phaseStart) / 1000;
      if (phase === "in" && p >= IN_S) {
        phase = "hold";
        phaseStart = now;
        void finishAction();
      }
      // cover: how much of the page is hidden; flow: how far strands have moved down and out
      const cover = phase === "in" ? ease(clamp(p / IN_S)) : phase === "hold" ? 1 : 1 - ease(clamp(p / OUT_S));
      const out = phase === "out" ? ease(clamp(p / OUT_S)) : 0;
      if (phase === "out" && p >= OUT_S) {
        done = true;
      }

      g.clearRect(0, 0, W, H);
      // veil in the page colour, so the swap underneath is never seen
      g.fillStyle = alpha(c.base, 0.94 * cover);
      g.fillRect(0, 0, W, H);

      g.globalCompositeOperation = mode === "dark" ? "lighter" : "source-over";
      // strands grow down from the top while covering; slide down and away when uncovering
      const grow = phase === "in" ? ease(clamp(p / IN_S)) : 1;
      g.lineWidth = 1.2;
      for (let k = 0; k < STRANDS; k++) {
        g.strokeStyle = alpha(hues[k % hues.length], 0.5 * cover);
        g.beginPath();
        const yStart = out * H * 1.1, yEnd = Math.min(H, H * grow * 1.15 + out * H);
        for (let y = yStart; y <= yEnd; y += 12) {
          const x = strandX(k, y, time);
          if (y === yStart) g.moveTo(x, y);
          else g.lineTo(x, y);
        }
        g.stroke();
      }
      // particles: leave the clicked point, curve onto their strand, then flow down it
      for (const q of particles.current) {
        const local = clamp((time - q.delay) / (IN_S - 0.05));
        if (local <= 0) continue;
        const flowY = ((q.u + time * 0.25) % 1) * H + out * H * 1.2;
        const sx = strandX(q.strand, flowY, time);
        const s = ease(local);
        const cx = mix(ox, sx, 0.5), cy = Math.min(oy, flowY) - 40; // control point above: an arc out of the origin
        const x = (1 - s) * (1 - s) * ox + 2 * (1 - s) * s * cx + s * s * sx;
        const y = (1 - s) * (1 - s) * oy + 2 * (1 - s) * s * cy + s * s * flowY;
        if (y > H + 10) continue;
        g.globalAlpha = Math.max(0.15, cover);
        g.fillStyle = hues[q.strand % hues.length];
        g.beginPath();
        g.arc(x, y, q.size, 0, 7);
        g.fill();
      }
      g.globalAlpha = 1;
      g.globalCompositeOperation = "source-over";

      if (!done) raf = requestAnimationFrame(frame);
      else {
        g.clearRect(0, 0, W, H);
        canvas.style.display = "none";
        busy.current = false;
      }
    };
    raf = requestAnimationFrame(frame);
    void raf;
  }, [slide]);

  useEffect(() => () => void (busy.current = false), []);

  return (
    <Ctx.Provider value={run}>
      {children}
      <canvas
        ref={ref}
        aria-hidden
        data-neural-transition
        className="pointer-events-none fixed inset-0 z-50 h-screen w-screen"
        style={{ display: "none" }}
      />
    </Ctx.Provider>
  );
}
