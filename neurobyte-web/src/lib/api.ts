import { parseNpy } from "./npy";

export interface SubjectInfo {
  id: string;
  sha256: string | null;
  hasScores: boolean;
  hasRanked: boolean;
  hasLabels: boolean;
}

export interface Manifest {
  subjects: SubjectInfo[];
  config: {
    name: string;
    scorers: number;
    scoreTransform: string;
    postprocess: string;
    sinkhorn: { iters: number; tau: number };
  };
  candidates: { first: number; last: number };
  /** per model: subjects whose score matrix is available */
  models: Record<string, string[]>;
}

async function get(url: string): Promise<Response> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r;
}

export const fetchManifest = async (): Promise<Manifest> => (await get("/api/manifest")).json();

export async function fetchNpy(kind: "scores" | "ranked" | "labels", sub: string) {
  return parseNpy(await (await get(`/api/${kind}/${sub}`)).arrayBuffer());
}

export async function fetchModelNpy(model: string, sub: string, kind: "pre" | "ranked") {
  return parseNpy(await (await get(`/api/model/${model}/${sub}/${kind}`)).arrayBuffer());
}

export async function fetchSample(sub: string): Promise<ArrayBuffer> {
  return (await get(`/api/sample/${sub}`)).arrayBuffer();
}

export const videoUrl = (id: number) => `/api/video/${id}`;

/**
 * Qwen2.5-VL one-line captions of the test videos, keyed by id. Display only.
 * Returns {} on any failure so the results page still works without captions.
 */
export async function fetchCaptions(): Promise<Record<number, string>> {
  try {
    const raw = (await (await get("/api/captions")).json()) as Record<string, unknown>;
    const out: Record<number, string> = {};
    for (const [k, v] of Object.entries(raw)) if (typeof v === "string") out[Number(k)] = v;
    return out;
  } catch {
    return {};
  }
}

export const captionOf = (caps: Record<number, string>, id: number): string | null => caps[id] ?? null;
