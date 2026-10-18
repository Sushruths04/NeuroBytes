import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { EXPERIMENTS, LATENCY, SYSTEMS, TEAM_SCIENCE } from "./team";
import { MODELS } from "./models";
import { APP_DIR } from "../../test/fixtures";

// The team site's content file, when this checkout sits next to it.
const SRC = [path.join(APP_DIR, "..", "website", "content.json")].find((p) => fs.existsSync(p));

// Sentences deliberately left out of the client-facing site (they list rejected ideas).
const OMITTED = ["Ideas that looked good once but failed these checks"];

describe("team content", () => {
  it("model registry and progression chart agree on every model's dev numbers", () => {
    const byName: Record<string, string> = { WF: "WF1" };
    for (const m of MODELS) {
      const s = SYSTEMS.find((x) => x.name === (byName[m.id] ?? m.id));
      if (m.id === "WF") continue; // WF1 in the chart is the earlier winfuse; WF here is submission2
      expect(s, m.id).toBeDefined();
      expect(s!.top1, m.id).toBeCloseTo(m.frozen.top1, 1);
      expect(s!.top5, m.id).toBeCloseTo(m.frozen.top5, 1);
    }
    expect(SYSTEMS.at(-1)!.name).toBe("C13");
  });

  it.skipIf(!SRC)("matches website/content.json (except C13, which it leaves for the build script)", () => {
    const c = JSON.parse(fs.readFileSync(SRC!, "utf8"));
    // Robustness is replaced by the final repo's checks (neurobytes RESULTS.md, h6/h10 networks)
    const UPDATED = ["Robustness"];
    expect(EXPERIMENTS.filter((e) => !UPDATED.includes(e.id))).toEqual(c.experiments.filter((e: { id: string }) => !UPDATED.includes(e.id)));
    expect(LATENCY).toEqual(c.latency);
    for (const [i, s] of c.systems.entries()) {
      if (s.auto) continue;
      expect({ ...SYSTEMS[i] }).toEqual(s);
    }
    expect(TEAM_SCIENCE.map((x) => x.title)).toEqual(c.science.map((x: { title: string }) => x.title));
    // every sentence of the original cards survives the HTML -> text conversion
    for (const [i, card] of c.science.entries()) {
      const text = [...TEAM_SCIENCE[i].paragraphs, TEAM_SCIENCE[i].after ?? "", ...(TEAM_SCIENCE[i].bullets ?? []),
        ...(TEAM_SCIENCE[i].tiles ?? []).flat()].join(" ");
      const plain = card.html.replace(/<div class='timeline'[\s\S]*?<\/div><\/div>/, "").replace(/<[^>]+>/g, " ").replace(/&gt;/g, ">");
      for (const sentence of plain.split(/(?<=\.)\s+|\s{2,}/).map((x: string) => x.trim()).filter((x: string) => x.length > 12))
        if (!OMITTED.some((o) => sentence.startsWith(o))) expect(text, `card ${i}`).toContain(sentence);
    }
  });
});
