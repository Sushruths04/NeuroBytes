/**
 * Visual check of the home-page particle animation (brain → spine → neurons).
 *
 *   npm run e2e:scroll              # headless, screenshots in e2e/out/
 *   npm run e2e:scroll -- --headed  # opens a visible browser you can watch
 *
 * Needs the dev server on http://127.0.0.1:5173 (npm run dev).
 * For each scroll stop it saves a screenshot and measures where the coloured
 * canvas pixels are, so "did it move" is a number, not an impression.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const URL = process.env.E2E_URL ?? "http://127.0.0.1:5173/";
const OUT = path.join(import.meta.dirname, "out");
const headed = process.argv.includes("--headed");
fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ headless: !headed });
const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "dark" })).newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
await page.goto(URL);
await page.evaluate(() => localStorage.clear());
await page.reload();
await page.waitForTimeout(1200);

/** Stats of the animation canvas: lit-pixel share, horizontal spread, centre of mass. */
async function canvasStats() {
  return page.evaluate(() => {
    const c = document.querySelector("canvas[data-brain-canvas]");
    if (!c) return null;
    const g = c.getContext("2d");
    const { width: w, height: h } = c;
    const d = g.getImageData(0, 0, w, h).data;
    let n = 0, sx = 0, sy = 0, sxx = 0, centre = 0;
    for (let y = 0; y < h; y += 3)
      for (let x = 0; x < w; x += 3) {
        if (d[(y * w + x) * 4 + 3] < 40) continue;
        n++; sx += x; sy += y; sxx += x * x;
        if (Math.abs(x - w / 2) < w * 0.06) centre++;
      }
    const mx = n ? sx / n : 0;
    return {
      lit: +(n / ((w / 3) * (h / 3))).toFixed(4),
      cx: +(mx / w).toFixed(3),
      cy: +(n ? sy / n / h : 0).toFixed(3),
      spreadX: +(n ? Math.sqrt(sxx / n - mx * mx) / w : 0).toFixed(3),
      centreShare: +(n ? centre / n : 0).toFixed(3),
    };
  });
}

const maxScroll = await page.evaluate(() => document.documentElement.scrollHeight - innerHeight);
const sec = await page.evaluate(() => {
  const s = document.querySelector("[data-brain-neurons]");
  return s ? s.getBoundingClientRect().top + scrollY : null;
});
const stops = [
  ["0_top", 0],
  ["1a_flow", 120],
  ["1b_flow", 250],
  ["1c_flow", 420],
  ["2_spine", Math.round((sec ?? maxScroll) * 0.55)],
  ["3_bundle", Math.round((sec ?? maxScroll) - 700)],
  ["4_neurons", Math.round((sec ?? maxScroll) - 150)],
  ["5_bottom", maxScroll],
];
const results = {};
for (const [name, y] of stops) {
  // Real wheel input, like a user, rather than scrollTo().
  const cur = await page.evaluate(() => scrollY);
  await page.mouse.move(720, 450);
  await page.mouse.wheel(0, y - cur);
  await page.waitForTimeout(900);
  const s = await canvasStats();
  // frames actually rendered in 1 s (60 = smooth on a 60 Hz display)
  const fps = await page.evaluate(() => new Promise((res) => {
    let n = 0; const t0 = performance.now();
    const f = (t) => (t - t0 < 1000 ? (n++, requestAnimationFrame(f)) : res(n));
    requestAnimationFrame(f);
  }));
  results[name] = { scrollY: await page.evaluate(() => Math.round(scrollY)), fps, ...s };
  await page.screenshot({ path: path.join(OUT, `${name}.png`) });
}
console.table(results);

// Text the spine would run through: every text line outside a (frosted, blurred) card whose
// box enters the spine lane (page centre ± 45 px), from the top of the page down
// to the first results card (below that the spine has become neurons).
const crossings = await page.evaluate(() => {
  const lane = [innerWidth / 2 - 45, innerWidth / 2 + 45];
  const stop = document.querySelector("[data-brain-neurons] .card");
  const stopY = stop ? stop.getBoundingClientRect().top + scrollY : Infinity;
  const hits = [];
  const walk = document.createTreeWalker(document.querySelector("#root"), NodeFilter.SHOW_TEXT);
  for (let n = walk.nextNode(); n; n = walk.nextNode()) {
    if (!n.textContent.trim() || n.parentElement.closest(".card, header, canvas")) continue;
    const r = document.createRange();
    r.selectNodeContents(n);
    for (const b of r.getClientRects()) {
      const top = b.top + scrollY;
      if (top >= stopY || b.width === 0) continue;
      if (b.right > lane[0] && b.left < lane[1]) hits.push(n.textContent.trim().slice(0, 50));
    }
  }
  return [...new Set(hits)];
});
results._spineCrossings = crossings;
console.log(crossings.length ? `FAIL spine crosses text: ${JSON.stringify(crossings)}` : "PASS spine lane is clear of text");
if (errors.length) console.log("console errors:", errors);
if (headed) await page.waitForTimeout(4000);
await browser.close();
fs.writeFileSync(path.join(OUT, "stats.json"), JSON.stringify(results, null, 2));
if (errors.length || crossings.length) process.exitCode = 1;
