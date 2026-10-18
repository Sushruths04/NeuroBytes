import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { TOKENS, type Mode, type Token } from "../theme";

const KEY = "neurobytes:theme";

/** Stored choice, else light, whatever the OS prefers. index.html applies the same rule before first paint. */
export function initialMode(): Mode {
  try {
    const s = localStorage.getItem(KEY);
    if (s === "light" || s === "dark") return s;
  } catch {
    /* storage blocked */
  }
  return "light";
}

interface ThemeCtx {
  mode: Mode;
  t: Record<Token, string>;
  toggle: () => void;
}

const Ctx = createContext<ThemeCtx>({ mode: "dark", t: TOKENS.dark, toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>(initialMode);

  useEffect(() => {
    document.documentElement.dataset.theme = mode;
    try {
      localStorage.setItem(KEY, mode);
    } catch {
      /* ignore */
    }
  }, [mode]);

  const toggle = useCallback(() => setMode((m) => (m === "dark" ? "light" : "dark")), []);
  return <Ctx.Provider value={{ mode, t: TOKENS[mode], toggle }}>{children}</Ctx.Provider>;
}

export const useTheme = () => useContext(Ctx);
