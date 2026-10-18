import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { BENCHMARKS, CITATIONS, KEY_CHOICES, PROGRESSION, SUPPORTING } from "./science";
import { MODELS } from "./models";
import { PATHS } from "../../test/fixtures";

const findings = [...KEY_CHOICES, ...SUPPORTING];
const ALLOWED = /^https:\/\/(doi\.org|pubmed\.ncbi\.nlm\.nih\.gov|www\.ncbi\.nlm\.nih\.gov\/pmc|arxiv\.org\/abs|scholar\.google\.com)\//;

// The source document lives next to the winfuse submission (not committed to every repo).
const SOURCE = [path.join(PATHS.submissionDir, "WHY_IT_WORKS.md"), path.join(PATHS.dataRoot, "WHY_IT_WORKS.md")].find((p) =>
  fs.existsSync(p),
);
const README = path.join(PATHS.submissionDir, "README.md");

describe("science content", () => {
  it("every citation id used exists, is unique, and has an allowed link", () => {
    const ids = new Set(CITATIONS.map((c) => c.id));
    expect(ids.size).toBe(CITATIONS.length);
    const view = fs.readFileSync(path.join(import.meta.dirname, "../components/ScienceView.tsx"), "utf8");
    const inline = [...view.matchAll(/ids=\{\[([^\]]*)\]\}/g)].flatMap((m) => [...m[1].matchAll(/"([\w-]+)"/g)].map((x) => x[1]));
    const used = [
      ...findings.flatMap((f) => f.cites),
      ...BENCHMARKS.flatMap((b) => [..."cite" in b && b.cite ? [b.cite] : [], ..."taskCite" in b && b.taskCite ? [b.taskCite] : []]),
      ...inline,
    ];
    for (const id of used) expect(ids.has(id), id).toBe(true);
    for (const c of CITATIONS) expect(c.url, c.id).toMatch(ALLOWED);
    // and the reference list has no entry that nothing on the page cites
    for (const c of CITATIONS) expect(used.includes(c.id), `${c.id} is never cited`).toBe(true);
  });

  it("the comparison table's C13 row equals the frozen C13 numbers", () => {
    const c13 = MODELS.find((m) => m.id === "C13")!;
    const row = BENCHMARKS.find((b) => b.system.startsWith("C13"))!;
    expect(row.top1).toBe(`${c13.frozen.top1.toFixed(2)}%`);
    expect(row.top5).toBe(`${c13.frozen.top5.toFixed(2)}%`);
  });

  it("progression is monotonic and ends at the headline 60.54%", () => {
    for (let i = 1; i < PROGRESSION.length; i++) expect(PROGRESSION[i].top5).toBeGreaterThan(PROGRESSION[i - 1].top5);
    expect(PROGRESSION.at(-1)!.top5).toBe(60.54);
  });

  it.skipIf(!SOURCE)("every number on the page appears in WHY_IT_WORKS.md or submission2/README.md", () => {
    const corpus = fs.readFileSync(SOURCE!, "utf8") + (fs.existsSync(README) ? fs.readFileSync(README, "utf8") : "");
    const text = [
      ...findings.flatMap((f) => [f.plain, f.why, f.takeaway, ...f.measured]),
      ...BENCHMARKS.filter((b) => !b.system.startsWith("C13")).flatMap((b) => [b.top1, b.top5]), // C13: checked above
      ...PROGRESSION.map((p) => String(p.top5)),
      ...CITATIONS.map((c) => c.role),
    ].join(" ");
    // decimals, percentages and "N ms"/"N pp"-style integers that carry a claim
    const nums = new Set(text.match(/\d+\.\d+|\d+(?=\s?(%|pp|ms|×))/g) ?? []);
    const missing = [...nums].filter((n) => !corpus.includes(n));
    expect(missing).toEqual([]);
  });
});
