import { useCallback, useEffect, useState } from "react";

/** Tiny hash router: #/ (home, incl. results), #/analysis, #/science. Works with any static host. */
export type Route = "home" | "analysis" | "science";

const parse = (h: string): Route => {
  const p = h.replace(/^#\/?/, "").split(/[?#]/)[0];
  return p === "analysis" || p === "science" ? p : "home";
};

export function useRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parse(location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = useCallback((r: Route) => {
    const h = href(r);
    if (location.hash !== h) location.hash = h;
    window.scrollTo({ top: 0 });
  }, []);
  return [route, go];
}

export const href = (r: Route) => (r === "home" ? "#/" : `#/${r}`);
