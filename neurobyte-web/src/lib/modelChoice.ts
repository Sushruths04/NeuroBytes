import { useCallback, useState } from "react";
import { DEFAULT_MODEL, isModelId, type ModelId } from "../content/models";

const KEY = "neurobytes:model";

/** The selected model, persisted per browser. Defaults to C13 (the team's final pipeline). */
export function useModelChoice(): [ModelId, (m: ModelId) => void] {
  const [model, setModel] = useState<ModelId>(() => {
    try {
      const s = localStorage.getItem(KEY);
      if (isModelId(s)) return s;
    } catch {
      /* storage blocked */
    }
    return DEFAULT_MODEL;
  });
  const set = useCallback((m: ModelId) => {
    setModel(m);
    try {
      localStorage.setItem(KEY, m);
    } catch {
      /* ignore */
    }
  }, []);
  return [model, set];
}
