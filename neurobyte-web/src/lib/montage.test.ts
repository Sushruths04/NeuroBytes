import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { CHANNELS, POSITIONS } from "./montage";
import { N_CHANNELS } from "./eeg";
import { APP_DIR } from "../../test/fixtures";

describe("montage", () => {
  it("names one electrode per channel, in the organisers' file order", () => {
    expect(CHANNELS).toHaveLength(N_CHANNELS);
    expect(new Set(CHANNELS).size).toBe(N_CHANNELS);
    expect(CHANNELS.join(" ")).toBe("Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2");
  });

  // Only when the project README records the organisers' list (it does from the 18 Oct update on).
  const readme = path.join(APP_DIR, "..", "README.md");
  const text = fs.existsSync(readme) ? fs.readFileSync(readme, "utf8") : "";
  it.skipIf(!/electrode list/i.test(text))("matches the list recorded in the project README", () => {
    expect(text).toContain(CHANNELS.join(" "));
  });

  it("puts each electrode where its name says: odd = left, even = right, z = midline, all posterior", () => {
    CHANNELS.forEach((name, i) => {
      const [x, y] = POSITIONS[i];
      expect(Math.hypot(x, y), name).toBeLessThan(1);
      expect(y, `${name} is behind Cz`).toBeLessThan(0);
      const side = name.endsWith("z") ? 0 : Number(name.at(-1)) % 2 ? -1 : 1;
      if (side === 0) expect(Math.abs(x), name).toBeLessThan(1e-9);
      else expect(Math.sign(x), name).toBe(side);
    });
    const y = (n: string) => POSITIONS[CHANNELS.indexOf(n as never)][1];
    expect(y("Oz")).toBeLessThan(y("POz")); // occipital behind parieto-occipital behind parietal
    expect(y("POz")).toBeLessThan(y("Pz"));
  });
});
