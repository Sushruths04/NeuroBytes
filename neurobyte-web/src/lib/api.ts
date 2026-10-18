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
