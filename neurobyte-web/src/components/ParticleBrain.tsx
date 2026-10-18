import { useEffect, useRef } from "react";
import { alpha, BRAIN_HUES_LIGHT } from "../theme";
import { useTheme } from "../lib/themeMode";

/**
 * Scroll-driven particle animation for the home page, drawn on one fixed,
 * full-viewport canvas behind the content (decorative, aria-hidden).
 *
 *   top of page   a rotating, glowing particle brain over [data-brain-anchor]
 *   scrolling     the brain drains through its stem: particles peel off from the
 *                 bottom up, drop to the stem and travel down coloured strands that
 *                 curve into a spinal cord in the page's centre gutter
 *   experiments   the cord spreads into a network of linked, glowing nodes around and
 *                 behind [data-brain-neurons] (web2's neural-field style)
 *   other pages   the same network, spread over the viewport as the page background
 *
 * Rendering follows the web2 site: soft glow sprites with additive blending, fine
 * filaments between nearby points, a travelling spike wave and rare amber sparks.
 * Every particle keeps one colour (violet, blue or cyan) through all three states.
 * Reduced motion: states still follow scroll; nothing rotates or flows.
 */
interface Pt {
  x: number; y: number; z: number;
  hue: number; strand: number; u: number; seed: number;
  release: number; // scroll point (0..1 of the unravel) at which it leaves the brain
  neuron: number; soma: boolean; ang: number; dist: number;
  fx: number; fy: number; // network position (0..1 across the results section), for nodes
  parent: number; // the node a non-node particle drifts around
}

const N_STRANDS = 12;
const N_NEURONS = 20;
const clamp = (v: number) => Math.min(1, Math.max(0, v));
const ease = (v: number) => v * v * (3 - 2 * v);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;

function makeParticles(n: number, seed: number): Pt[] {
  let s = seed;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const P: Pt[] = [];
  while (P.length < n) {
    const u = rnd() * 2 - 1, th = rnd() * Math.PI * 2, r = Math.sqrt(1 - u * u);
    const x = r * Math.cos(th), y = u, z = r * Math.sin(th);
    const hemi = x >= 0 ? 1 : -1;
    const fold = 1 + 0.07 * Math.sin(9 * y + 4 * z) * Math.sin(7 * z - 3 * y) + 0.035 * Math.sin(17 * (y + z)) + 0.02 * Math.sin(23 * x);
    const X = (0.58 * Math.abs(x) + 0.055) * hemi * fold, Z = z * fold * 1.04;
    let Y = 0.72 * y * fold;
    if (Y < -0.35 && Math.abs(Z) < 0.5) Y = -0.35 - (Y + 0.35) * 0.42; // flatten underside
    if (Y < -0.46 && Z < -0.08) continue; // temporal notch
    const i = P.length;
    const strand = i % N_STRANDS;
    const neuron = i % N_NEURONS;
    const branch = Math.floor(rnd() * 6);
    P.push({
      x: X, y: -Y, z: Z,
      strand,
      hue: strand, // strand colour == particle colour
      u: rnd(),
      seed: rnd() * 1000,
      // lowest points leave first (the brain unravels upward from its stem), with jitter
      release: 0.62 * Math.min(1, Math.max(0, (0.8 - Y) / 1.6)) + 0.08 * rnd(),
      neuron,
      soma: rnd() < 0.08,
      ang: ((branch + neuron * 0.37) / 6) * Math.PI * 2 + (rnd() - 0.5) * 0.35,
      dist: 0.25 + 0.75 * Math.sqrt(rnd()),
      fx: rnd(),
      fy: rnd(),
      parent: 0,
    });
  }
  const nodes = P.map((p, i) => (p.soma ? i : -1)).filter((i) => i >= 0);
  P.forEach((p, i) => (p.parent = p.soma ? i : nodes[i % nodes.length]));
  return P;
}

/**
 * Network layout around and behind the results section (viewport coordinates):
 * nodes spread over the section plus a margin, kept out of the heading's text block.
 */
function nodeAt(r: DOMRect, fx: number, fy: number, W: number): [number, number] {
  const pad = Math.max(30, Math.min(90, r.left * 0.8));
  let x = r.left - pad + fx * (r.width + 2 * pad);
  const y = r.top - 30 + fy * (r.height + 60);
  if (y < r.top + 210 && x > r.left - 10 && x < r.left + r.width * 0.5) x = r.left + r.width * (0.5 + fx * 0.5); // heading text
  return [Math.min(W - 6, Math.max(6, x)), y];
}

/** A soft round glow, pre-rendered once per colour (much cheaper than a gradient per dot). */
function glowSprite(color: string): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(16, 16, 0, 16, 16, 16);
  grad.addColorStop(0, alpha(color, 1));
  grad.addColorStop(0.25, alpha(color, 0.55));
  grad.addColorStop(1, alpha(color, 0));
  g.fillStyle = grad;
  g.fillRect(0, 0, 32, 32);
  return c;
}

export function ParticleBrain({ n = 1400 }: { n?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { t, mode } = useTheme();
  const theme = useRef({ t, mode });
  theme.current = { t, mode };

  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const count = window.innerWidth < 640 ? Math.round(n * 0.5) : n; // lighter on phones
    const P = makeParticles(count, 7);
    const still = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, raf = 0;
    const t0 = performance.now();
    const order = P.map((_, i) => i);
    const sprites = new Map<string, HTMLCanvasElement>();
    const sprite = (c: string) => sprites.get(c) ?? (sprites.set(c, glowSprite(c)), sprites.get(c)!);

    const size = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    size();
    window.addEventListener("resize", size);

    const bx = new Float32Array(count), by = new Float32Array(count), bz = new Float32Array(count), bf = new Float32Array(count);
    const px = new Float32Array(count), py = new Float32Array(count);
    const nodeIdx = P.map((p, i) => (p.soma ? i : -1)).filter((i) => i >= 0);

    let fieldReady = false;
    /** Background network for pages without the landing-page brain. */
    function drawField(time: number, hues: readonly string[], add: boolean, spark: string) {
      if (!fieldReady) {
        // first frame on this page with no previous positions (e.g. a reload): start in place
        for (let i = 0; i < count; i++) if (!px[i] && !py[i]) { px[i] = P[P[i].parent].fx * W; py[i] = P[P[i].parent].fy * H; }
        fieldReady = true;
      }
      const k = still ? 1 : 0.06; // per-frame approach to the field layout
      for (let i = 0; i < count; i++) {
        const p = P[i], par = P[p.parent];
        let tx = (0.03 + par.fx * 0.94) * W + Math.sin(time * 0.2 + par.seed) * 14;
        let ty = (0.04 + par.fy * 0.92) * H + Math.cos(time * 0.17 + par.seed) * 10;
        if (!p.soma) {
          tx += Math.cos(p.ang + time * 0.12) * p.dist * 34;
          ty += Math.sin(p.ang + time * 0.12) * p.dist * 34;
        }
        px[i] += (tx - px[i]) * k;
        py[i] += (ty - py[i]) * k;
      }
      ctx.globalCompositeOperation = add ? "lighter" : "source-over";
      const LINK = 180;
      ctx.lineWidth = 0.8;
      for (let a = 0; a < nodeIdx.length; a++) {
        const i = nodeIdx[a];
        for (let b = a + 1; b < nodeIdx.length; b++) {
          const j = nodeIdx[b];
          const dx = px[i] - px[j], dy = py[i] - py[j], d2 = dx * dx + dy * dy;
          if (d2 > LINK * LINK) continue;
          const w = 1 - Math.sqrt(d2) / LINK;
          ctx.strokeStyle = alpha(hues[P[i].hue % hues.length], (add ? 0.22 : 0.32) * w);
          ctx.beginPath();
          ctx.moveTo(px[i], py[i]);
          ctx.lineTo(px[j], py[j]);
          ctx.stroke();
          if (!still && (a * 7 + b) % 9 === 0) {
            const t = (time * 0.4 + a * 0.13) % 1;
            ctx.globalAlpha = w;
            ctx.drawImage(sprite(spark), px[i] + (px[j] - px[i]) * t - 3, py[i] + (py[j] - py[i]) * t - 3, 6, 6);
            ctx.globalAlpha = 1;
          }
        }
      }
      for (let i = 0; i < count; i++) {
        const p = P[i], x = px[i], y = py[i];
        if (y < -12 || y > H + 12 || x < -12 || x > W + 12) continue;
        const r = p.soma ? 2.6 : 0.9;
        const lum = p.soma ? 0.8 : 0.35;
        const colour = hues[p.hue % hues.length];
        if (add) {
          ctx.globalAlpha = lum;
          ctx.drawImage(sprite(colour), x - r * 3, y - r * 3, r * 6, r * 6);
        } else {
          ctx.globalAlpha = Math.min(1, 0.3 + lum * 0.7);
          ctx.fillStyle = colour;
          ctx.beginPath();
          ctx.arc(x, y, Math.max(0.9, r * 1.1), 0, 7);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
    }

    function frame(now: number) {
      const { t: c, mode } = theme.current;
      const hues = mode === "dark" ? [c.iris, c.pine, c.foam] : BRAIN_HUES_LIGHT;
      const add = mode === "dark"; // additive light only works on black
      const time = still ? 0 : (now - t0) / 1000;
      ctx.clearRect(0, 0, W, H);

      // a page transition can ask the brain to let go of its shape and spread into the background
      const spreadOut = document.documentElement.dataset.brainSpread === "1";
      const anchor = spreadOut ? undefined : document.querySelector("[data-brain-anchor]")?.getBoundingClientRect();
      const sec = document.querySelector("[data-brain-neurons]")?.getBoundingClientRect();
      if (!anchor) {
        // Any other page: the same network the landing page ends with, spread over the
        // viewport behind the glass cards. Particles ease in from wherever they were.
        drawField(time, hues, add, c.gold);
        raf = requestAnimationFrame(frame);
        return;
      }
      // Scroll phases from the real layout.
      const heroBottomAbs = anchor.bottom + window.scrollY;
      const e1 = clamp(window.scrollY / Math.max(1, heroBottomAbs * 0.9)); // unravel (per-particle easing below)
      const raw = sec ? clamp((H * 0.75 - sec.top) / (H * 0.6)) : 0;
      const bundle = ease(clamp(raw / 0.4));
      const e2 = ease(clamp((raw - 0.35) / 0.65));

      // 1. brain projection inside the hero anchor
      const a = (still ? 0.9 : time * 0.14) + 0.9, b = 0.2 + (still ? 0 : 0.05 * Math.sin(time * 0.3));
      const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
      const S = Math.min(anchor.width, anchor.height) * 0.56;
      const acx = anchor.left + anchor.width / 2, acy = anchor.top + anchor.height / 2 + (still ? 0 : Math.sin(time * 0.5) * 5);
      for (let i = 0; i < count; i++) {
        const p = P[i];
        const x = p.x * ca - p.z * sa;
        let z = p.x * sa + p.z * ca;
        const y = p.y * cb - z * sb;
        z = p.y * sb + z * cb;
        const f = 2.6 / (3.2 + z);
        bx[i] = acx + x * S * f; by[i] = acy + y * S * f; bz[i] = z; bf[i] = f;
      }

      // 2. the cord: strands leave the brain's base, curve into the centre gutter, then run straight down
      const stemX = acx, stemY = acy + S * 0.62;
      const spineX = W / 2;
      const bend = Math.min(280, H * 0.32);
      const spread = 1 - 0.85 * bundle;
      const baseX = (y: number) => mix(stemX, spineX, ease(clamp((y - stemY) / bend)));
      const strandX = (k: number, y: number) => {
        const flare = clamp((y - stemY) / 90); // strands start together at the stem and fan out
        const off = (k - (N_STRANDS - 1) / 2) * 5 + Math.sin(y * 0.011 + time * 1.4 + k * 0.55) * (5 + 3 * (k % 3));
        return baseX(y) + off * flare * spread + Math.sin(y * 0.006 + time * 0.9) * 6 * bundle;
      };
      const yTop = Math.max(stemY, -60), yBot = H + 60;

      ctx.globalCompositeOperation = add ? "lighter" : "source-over";

      // ambient light behind the brain (web2), fading as it drains
      if (add && e1 < 1) {
        const g = ctx.createRadialGradient(acx, acy, 0, acx, acy, S * 1.5);
        g.addColorStop(0, alpha(c.iris, 0.1 * (1 - e1)));
        g.addColorStop(0.55, alpha(c.pine, 0.04 * (1 - e1)));
        g.addColorStop(1, alpha(c.pine, 0));
        ctx.fillStyle = g;
        ctx.fillRect(acx - S * 1.5, acy - S * 1.5, S * 3, S * 3);
      }

      // strands grow down from the stem as particles flow into them
      const grow = clamp(e1 / 0.55);
      const strandA = grow * (1 - e2);
      if (strandA > 0.01) {
        const yEnd = mix(yTop, yBot, grow);
        ctx.lineWidth = 1 + bundle;
        for (let k = 0; k < N_STRANDS; k++) {
          ctx.strokeStyle = alpha(hues[k % hues.length], (0.3 + 0.25 * bundle) * strandA);
          ctx.beginPath();
          for (let y = yTop; y <= yEnd; y += 8) {
            if (y === yTop) ctx.moveTo(strandX(k, y), y);
            else ctx.lineTo(strandX(k, y), y);
          }
          ctx.stroke();
        }
      }

      // brain filaments (web2): short links between nearby points, fading as it drains
      if (e1 < 0.9) {
        ctx.lineWidth = 0.6;
        for (let i = 0; i < count; i += 2) {
          const j = (i * 7 + 3) % count;
          if (P[i].release < e1 || P[j].release < e1) continue; // already left the brain
          const dx = bx[i] - bx[j], dy = by[i] - by[j], d2 = dx * dx + dy * dy;
          if (d2 > 46 * 46) continue;
          ctx.strokeStyle = alpha(hues[P[i].hue % hues.length], (add ? 1 : 1.6) * (1 - e1) * (0.08 + 0.12 * (1 - d2 / 2116)));
          ctx.beginPath();
          ctx.moveTo(bx[i], by[i]);
          ctx.lineTo(bx[j], by[j]);
          ctx.stroke();
        }
      }

      // 3. particles: position first (brain -> strand -> network), then links, then dots
      if (e1 < 0.6) order.sort((i, j) => bz[j] - bz[i]);
      for (let i = 0; i < count; i++) {
        const p = P[i];
        // how far this particle is through leaving the brain: 0 in the brain, 1 on its strand
        const q = clamp((e1 - p.release) / 0.3);
        const flowY = mix(yTop, yBot, (p.u + time * 0.03) % 1);
        let x: number, y: number;
        if (q <= 0) {
          x = bx[i]; y = by[i];
        } else if (q < 0.4) {
          // drop to the stem along a curve that first falls, then pulls in
          const s = ease(q / 0.4);
          x = (1 - s) * (1 - s) * bx[i] + 2 * (1 - s) * s * bx[i] + s * s * stemX;
          y = (1 - s) * (1 - s) * by[i] + 2 * (1 - s) * s * stemY + s * s * stemY;
        } else {
          // travel down the strand itself to its place in the flow
          y = mix(stemY, flowY, ease((q - 0.4) / 0.6));
          x = strandX(p.strand, y);
        }
        if (sec && e2 > 0) {
          // network: nodes drift slowly in place; other particles hang around their node as fine dust
          const par = P[p.parent];
          let [nx, ny] = nodeAt(sec, par.fx, par.fy, W);
          nx += Math.sin(time * 0.25 + par.seed) * 10;
          ny += Math.cos(time * 0.21 + par.seed) * 8;
          if (!p.soma) {
            nx += Math.cos(p.ang + time * 0.15) * p.dist * 30;
            ny += Math.sin(p.ang + time * 0.15) * p.dist * 30;
          }
          x = mix(x, nx, e2);
          y = mix(y, ny, e2);
        }
        px[i] = x; py[i] = y;
      }
      fieldReady = true;

      // network links between nearby nodes, with spikes travelling along some of them
      if (e2 > 0.05) {
        const LINK = 170;
        ctx.lineWidth = 0.8;
        for (let a = 0; a < nodeIdx.length; a++) {
          const i = nodeIdx[a];
          for (let b = a + 1; b < nodeIdx.length; b++) {
            const j = nodeIdx[b];
            const dx = px[i] - px[j], dy = py[i] - py[j], d2 = dx * dx + dy * dy;
            if (d2 > LINK * LINK) continue;
            const k = 1 - Math.sqrt(d2) / LINK;
            ctx.strokeStyle = alpha(hues[P[i].hue % hues.length], (add ? 0.28 : 0.4) * k * e2);
            ctx.beginPath();
            ctx.moveTo(px[i], py[i]);
            ctx.lineTo(px[j], py[j]);
            ctx.stroke();
            if (!still && (a * 7 + b) % 9 === 0) {
              const t = (time * 0.4 + a * 0.13) % 1;
              ctx.globalAlpha = e2 * k;
              ctx.drawImage(sprite(c.gold), mix(px[i], px[j], t) - 3, mix(py[i], py[j], t) - 3, 6, 6);
              ctx.globalAlpha = 1;
            }
          }
        }
      }

      for (const i of order) {
        const p = P[i];
        const x = px[i], y = py[i];
        if (y < -12 || y > H + 12 || x < -12 || x > W + 12) continue;
        const q = clamp((e1 - p.release) / 0.3);
        const depth = clamp((bz[i] + 1.3) / 2.6) * (1 - q) + q; // on the strand: full brightness
        const spike = still ? 0 : Math.max(0, Math.sin(time * 1.3 - p.seed * 0.03) - 0.78) * 4.5;
        const lum = clamp((0.25 + depth * 0.55 + spike * (1 - e2)) * (q > 0 && q < 1 ? 1.15 : 1) * (p.soma ? 1 : 1 - 0.45 * e2));
        const colour = p.soma && spike > 0.1 ? c.gold : hues[p.hue % hues.length];
        const r = mix((p.soma ? 1.7 : 1) * (0.6 + depth * 1.2) * bf[i], 1.1, q) * (1 + spike * 0.3) * (p.soma ? 1 + 0.9 * e2 : 1 - 0.3 * e2);
        if (add) {
          // dark: soft additive glow
          ctx.globalAlpha = lum;
          ctx.drawImage(sprite(colour), x - r * 3, y - r * 3, r * 6, r * 6);
        } else {
          // light: glows wash out on cream, so draw crisp, more opaque dots that keep their colour
          ctx.globalAlpha = Math.min(1, 0.35 + lum * 0.75);
          ctx.fillStyle = colour;
          ctx.beginPath();
          ctx.arc(x, y, Math.max(0.9, r * 1.15), 0, 7);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", size);
    };
  }, [n]);

  return <canvas ref={ref} data-brain-canvas aria-hidden className="pointer-events-none fixed inset-0 z-0 h-screen w-screen" />;
}
