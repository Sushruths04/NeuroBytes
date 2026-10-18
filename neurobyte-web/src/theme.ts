/**
 * The site's colour tokens, drawn from color-scheme.json: dark mode uses "web2"
 * (pure black, violet → blue → cyan light sources, rare amber sparks; from the
 * web2/ site), light mode uses "earth_tones" (tea green). index.css declares the
 * same values as CSS variables (dark in @theme, light under [data-theme="light"]);
 * this module is for places CSS classes can't reach (canvas, SVG attributes) —
 * read it through useTheme(). theme.test.ts checks index.css, this file and the JSON agree.
 */
export const SCHEME = "earth_tones_hex"; // light
export const DARK_SCHEME = "web2_hex";

export type Mode = "dark" | "light";

export const PALETTE_REF = {
  base: "void",
  surface: "void",
  overlay: "ink_raised",
  muted: "muted",
  subtle: "mist",
  text: "text",
  love: "amber", // warnings, "hot" brain nodes (never text)
  gold: "amber", // emphasis: "precomputed" chips, ranks
  rose: "cyan",
  pine: "blue",
  foam: "cyan", // success, "in browser", cursor
  iris: "violet_aa", // primary accent, filled pills
  "on-iris": "white", // text on filled iris
  "highlight-low": "ink",
  "highlight-med": "line",
  "highlight-high": "line_strong",
} as const;

export type Token = keyof typeof PALETTE_REF;

export const PALETTE_REF_LIGHT: Record<Token, string> = {
  base: "cornsilk.500",
  surface: "cornsilk.800",
  overlay: "beige.500",
  muted: "tea_green.200",
  subtle: "tea_green.200",
  text: "tea_green.100",
  love: "light_bronze.200",
  gold: "papaya_whip.100",
  rose: "light_bronze.300",
  pine: "tea_green.300",
  foam: "cornsilk.100",
  iris: "tea_green.200",
  "on-iris": "cornsilk.500",
  "highlight-low": "beige.600",
  "highlight-med": "tea_green.500",
  "highlight-high": "tea_green.400",
};

export const DARK: Record<Token, string> = {
  base: "#000000",
  surface: "#000000",
  overlay: "#12121c",
  muted: "#92929f",
  subtle: "#b4b4c2",
  text: "#f5f4fa",
  love: "#ffbe62",
  gold: "#ffbe62",
  rose: "#20d9ff",
  pine: "#356cff",
  foam: "#20d9ff",
  iris: "#8052ff",
  "on-iris": "#ffffff",
  "highlight-low": "#0a0a12",
  "highlight-med": "#1b1b26",
  "highlight-high": "#2c2c3a",
};

export const LIGHT: Record<Token, string> = {
  base: "#fefae0",
  surface: "#fffdf3",
  overlay: "#e9edc9",
  muted: "#5b6635",
  subtle: "#5b6635",
  text: "#2d331a",
  love: "#64411f",
  gold: "#533e08",
  rose: "#96622e",
  pine: "#88994f",
  foam: "#5d5103",
  iris: "#5b6635",
  "on-iris": "#fefae0",
  "highlight-low": "#edf1d4",
  "highlight-med": "#ccd5ae",
  "highlight-high": "#acbb7b",
};

export const TOKENS: Record<Mode, Record<Token, string>> = { dark: DARK, light: LIGHT };

/** Dark tokens; kept for code that has no mode (tests). Components use useTheme().t */
export const T = DARK;

/** `#rrggbb` + alpha → `rgba(r,g,b,a)` for canvas gradients and fills. */
export function alpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/**
 * Light-mode brain palette: six earth_tones shades (olive, bronze, gold, lime,
 * brown, mustard) so the particles vary in colour on the cream background. Dark mode
 * uses the web2 light sources (iris, pine, foam) instead.
 */
export const BRAIN_HUES_LIGHT_REF = ["tea_green.200", "light_bronze.300", "papaya_whip.200", "beige.200", "light_bronze.200", "cornsilk.100"] as const;
export const BRAIN_HUES_LIGHT = ["#5b6635", "#96622e", "#a57b10", "#79842c", "#64411f", "#5d5103"];
