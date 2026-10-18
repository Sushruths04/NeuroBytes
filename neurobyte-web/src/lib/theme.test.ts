import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { APP_DIR } from "../../test/fixtures";
import { BRAIN_HUES_LIGHT, BRAIN_HUES_LIGHT_REF, DARK, DARK_SCHEME, LIGHT, PALETTE_REF, PALETTE_REF_LIGHT, SCHEME, alpha, type Token } from "../theme";

const json = JSON.parse(fs.readFileSync(path.join(APP_DIR, "color-scheme.json"), "utf8"));
const css = fs.readFileSync(path.join(APP_DIR, "src/index.css"), "utf8");
// "nord8" (flat palette) or "tea_green.200" (palette with shades)
const lookup = (scheme: string, ref: string) => {
  const [name, shade] = ref.split(".");
  const v = shade === undefined ? json[scheme]?.[name] : json[scheme]?.[name]?.[shade];
  return String(v ?? "").toLowerCase();
};
// dark tokens live in @theme { … }, light ones in :root[data-theme="light"] { … }
const block = (start: string) => {
  const i = css.indexOf(start);
  return css.slice(i, css.indexOf("}", i));
};
const MODES = [
  { name: "dark", scheme: DARK_SCHEME, css: block("@theme {"), refs: PALETTE_REF as Record<Token, string>, tokens: DARK },
  { name: "light", scheme: SCHEME, css: block(':root[data-theme="light"] {'), refs: PALETTE_REF_LIGHT, tokens: LIGHT },
];

const lum = (h: string) => {
  const c = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255).map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
};
const ratio = (a: string, b: string) => {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};

describe(`theme (dark ${DARK_SCHEME}, light ${SCHEME})`, () => {
  for (const m of MODES) {
    it(`${m.name}: every token in index.css comes from color-scheme.json and matches src/theme.ts`, () => {
      for (const [token, ref] of Object.entries(m.refs)) {
        const r = new RegExp(`--color-${token}:\\s*(#[0-9a-fA-F]{6});\\s*/\\*\\s*([\\w.]+)\\s*\\*/`).exec(m.css);
        expect(r, `${m.name} ${token}`).not.toBeNull();
        expect(r![2], `${m.name} ${token} palette ref`).toBe(ref);
        expect(r![1].toLowerCase(), `${m.name} ${token} css vs json`).toBe(lookup(m.scheme, ref));
        expect(m.tokens[token as Token].toLowerCase(), `${m.name} ${token} theme.ts vs json`).toBe(lookup(m.scheme, ref));
      }
    });

    it(`${m.name}: contrast — text AA everywhere, accents AA on base/surface, error colour >= 3:1`, () => {
      const t = m.tokens;
      // body text is read on every background, including raised/active panels
      for (const fg of ["text", "subtle"] as const)
        for (const bg of ["base", "surface", "overlay"] as const)
          expect(ratio(t[fg], t[bg]), `${m.name}: ${fg} on ${bg}`).toBeGreaterThanOrEqual(4.5);
      // accent-coloured text (links, chips, ranks) sits on page and card backgrounds
      for (const fg of ["muted", "iris", "foam", "gold"] as const)
        for (const bg of ["base", "surface"] as const)
          expect(ratio(t[fg], t[bg]), `${m.name}: ${fg} on ${bg}`).toBeGreaterThanOrEqual(4.5);
      // `love` is only used for borders, tints and brain nodes (never text): WCAG 1.4.11 non-text 3:1
      for (const bg of ["base", "surface"] as const)
        expect(ratio(t.love, t[bg]), `${m.name}: love on ${bg}`).toBeGreaterThanOrEqual(3);
      expect(ratio(t["on-iris"], t.iris), `${m.name}: text on filled iris (buttons, pills)`).toBeGreaterThanOrEqual(4.5);
      expect(ratio(t.base, t.foam), `${m.name}: base text on foam badges`).toBeGreaterThanOrEqual(4.5);
    });
  }

  it("light-mode brain colours come from earth_tones and are visible on the page (>= 3:1)", () => {
    expect(BRAIN_HUES_LIGHT.map((h) => h.toLowerCase())).toEqual(BRAIN_HUES_LIGHT_REF.map((r) => lookup(SCHEME, r)));
    expect(new Set(BRAIN_HUES_LIGHT).size).toBe(BRAIN_HUES_LIGHT.length);
    for (const h of BRAIN_HUES_LIGHT) expect(ratio(h, LIGHT.base), h).toBeGreaterThanOrEqual(3);
  });

  it("light and dark define the same tokens", () => {
    expect(Object.keys(LIGHT).sort()).toEqual(Object.keys(DARK).sort());
  });

  it("index.html applies the saved theme before first paint, with the same storage key", () => {
    const html = fs.readFileSync(path.join(APP_DIR, "index.html"), "utf8");
    const mode = fs.readFileSync(path.join(APP_DIR, "src/lib/themeMode.tsx"), "utf8");
    const key = /const KEY = "([^"]+)"/.exec(mode)![1];
    expect(html).toContain(`localStorage.getItem("${key}")`);
    expect(html).toContain("dataset.theme");
    // light is the default regardless of the OS setting, in both places
    expect(html).not.toContain("prefers-color-scheme");
    expect(mode).not.toContain("prefers-color-scheme");
    expect(html).toContain('data-theme="light"');
  });

  it("love is never used as a text colour (it can't reach 4.5:1 on Nord)", () => {
    const dir = path.join(APP_DIR, "src");
    for (const f of fs.readdirSync(dir, { recursive: true }).map(String).filter((f) => f.endsWith(".tsx")))
      expect(fs.readFileSync(path.join(dir, f), "utf8"), f).not.toMatch(/text-love\b/);
  });

  it("no component hard-codes a colour outside the theme", () => {
    const dir = path.join(APP_DIR, "src");
    const files = fs.readdirSync(dir, { recursive: true }).map(String).filter((f) => /\.tsx?$/.test(f) && !/test|theme\.ts$/.test(f));
    for (const f of files) {
      const src = fs.readFileSync(path.join(dir, f), "utf8");
      expect(src.match(/#[0-9a-fA-F]{6}\b|rgba?\(\s*\d/g), f).toBeNull();
    }
  });

  it("alpha() converts hex", () => {
    expect(alpha("#4ee1ff", 0.5)).toBe("rgba(78,225,255,0.5)");
  });
});
