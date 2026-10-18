/**
 * Visual check of the #/analysis results layout.
 *
 *   npm run e2e:analysis              # headless, screenshots in e2e/out/
 *   npm run e2e:analysis -- --headed  # visible browser
 *
 * Needs the dev server on http://127.0.0.1:5173 (npm run dev) and the local data.
 * Opens sub-01 from the home page, then switches to sub-05 with the person tabs,
 * saves full-page screenshots and checks the section order and the label rules
 * (sub-05/06: no ground truth, no ✓ marks).
 *
 * Expected order: person tabs, 5 stat cards, Predicted video | Top 5, Top 10 ranking,
 * Where the true video lands | Per person (pooled sub-01..04), Frequency analysis,
 * one card with Recording N · EEG and the Recordings list, and last Inference stats.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const URL = process.env.E2E_URL ?? "http://127.0.0.1:5173/";
const OUT = path.join(import.meta.dirname, "out");
const headed = process.argv.includes("--headed");
fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({
  headless: !headed,
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
  args: ["--no-sandbox", "--disable-setuid-sandbox"],
});
const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "dark" })).newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("response", (res) => { if (res.status() >= 400) console.log(res.status(), "URL:", res.url()); });

await page.goto(URL);
await page.evaluate(() => localStorage.clear());
await page.reload();
await page.locator('section[aria-labelledby="pick-title"] button', { hasText: "sub-01" }).click();

/** Wait until the results screen is up, skipping the stage animation if shown. */
async function waitResults() {
  const skip = page.getByRole("button", { name: /Skip to results/ });
  const pred = page.locator("#pred-title");
  for (let i = 0; i < 120 && !(await pred.isVisible()); i++) {
    if (await skip.isVisible().catch(() => false)) await skip.click().catch(() => {});
    await page.waitForTimeout(250);
  }
  await pred.waitFor({ state: "visible", timeout: 5000 });
  await page.waitForTimeout(1500); // captions, thumbnails
}

async function check(sub) {
  return page.evaluate(() => {
    const ids = ["pred-title", "top5-title", "top10-title", "rankhist-title", "persub-title", "eeg-title", "trials-title", "stats-title"];
    const tops = Object.fromEntries(ids.map((id) => [id, Math.round(document.getElementById(id)?.getBoundingClientRect().top ?? -1)]));
    const spectrum = [...document.querySelectorAll("h2")].find((h) => /frequency|spectrum/i.test(h.textContent ?? ""));
    const pred = document.getElementById("pred-title")?.closest("section");
    const top5 = document.getElementById("top5-title")?.closest("section");
    const eegSection = document.getElementById("eeg-title")?.closest("section") ?? null;
    const statsSection = document.getElementById("stats-title")?.closest("section") ?? null;
    const isLast = (s) => !!s && s.parentElement?.lastElementChild === s &&
      [...(s.parentElement?.children ?? [])].every((c) => c === s || c.getBoundingClientRect().bottom <= s.getBoundingClientRect().top + 1);
    return {
      tops,
      spectrumTop: spectrum ? Math.round(spectrum.getBoundingClientRect().top) : null,
      verdict: pred?.querySelector(".chip")?.textContent ?? null,
      groundTruthBox: !!pred?.textContent?.includes("Ground truth: video actually watched"),
      checkMarks: (top5?.textContent?.match(/✓/g) ?? []).length,
      captionsUnavailable: (document.body.textContent?.match(/caption unavailable/g) ?? []).length,
      predCaption: pred?.querySelector(".text-sm.text-text")?.textContent ?? null,
      trialsInEeg: !!eegSection && document.getElementById("trials-title")?.closest("section") === eegSection,
      // Inference stats is the last card of the results grid, the EEG card directly before it
      statsLast: isLast(statsSection),
      eegBeforeStats: !!eegSection && !!statsSection && eegSection.nextElementSibling === statsSection,
      histBars: [...document.querySelectorAll("section[aria-labelledby=rankhist-title] [data-bar]")].map((g) => g.textContent),
      histCaption: !!document.querySelector("section[aria-labelledby=rankhist-title]")?.textContent?.includes("rank of the true video among 102"),
      persubRows: (document.querySelector("section[aria-labelledby=persub-title] svg")?.textContent?.match(/sub-0\d/g) ?? []),
    };
  });
}

await waitResults();
const r1 = await check("sub-01");
await page.screenshot({ path: path.join(OUT, "analysis_sub-01.png"), fullPage: true });

await page.getByRole("tab", { name: /sub-05/ }).click();
await page.waitForFunction(() => document.querySelector('[role="tab"][aria-selected="true"]')?.textContent?.includes("sub-05"), null, { timeout: 60000 });
await waitResults();
const r5 = await check("sub-05");
await page.screenshot({ path: path.join(OUT, "analysis_sub-05.png"), fullPage: true });

console.log(JSON.stringify({ "sub-01": r1, "sub-05": r5 }, null, 2));
const fail = [];
for (const [sub, r] of [["sub-01", r1], ["sub-05", r5]]) {
  const t = r.tops;
  // Predicted video | Top 5, Top 10, rank charts (one row), spectrum, EEG + Recordings, Inference stats last.
  if (!(t["pred-title"] < t["top10-title"] && t["top10-title"] < t["rankhist-title"] && Math.abs(t["rankhist-title"] - t["persub-title"]) <= 4 &&
    t["rankhist-title"] < (r.spectrumTop ?? -Infinity) && (r.spectrumTop ?? Infinity) < t["eeg-title"] && t["eeg-title"] < t["stats-title"]))
    fail.push(`${sub}: section order ${JSON.stringify(t)} spectrum ${r.spectrumTop}`);
  if (!r.trialsInEeg) fail.push(`${sub}: Recordings list is not inside the EEG section`);
  if (!r.statsLast) fail.push(`${sub}: Inference stats is not the last card`);
  if (!r.eegBeforeStats) fail.push(`${sub}: EEG section is not directly before Inference stats`);
  if (r.histBars.length !== 6 || !r.histCaption) fail.push(`${sub}: rank histogram ${JSON.stringify(r.histBars)} caption ${r.histCaption}`);
  if (r.persubRows.join() !== "sub-01,sub-02,sub-03,sub-04") fail.push(`${sub}: per person rows ${r.persubRows}`);
}
if (r5.verdict !== "scored person · labels private" || r5.groundTruthBox || r5.checkMarks) fail.push("sub-05 shows correctness");
if (!r1.groundTruthBox) fail.push("sub-01 has no ground-truth box");
console.log(fail.length ? `FAIL ${fail.join("; ")}` : "PASS layout order and label rules");
if (errors.length) console.log("console errors:", errors);
if (headed) await page.waitForTimeout(4000);
await browser.close();
if (errors.length || fail.length) process.exitCode = 1;
