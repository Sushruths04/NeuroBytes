/**
 * Read-only, local-only data API for the frontend.
 *
 *   GET /api/manifest              subjects, recording fingerprints, ranking config
 *   GET /api/scores/<sub>          <submission>/scores/<sub>_fused_scores.npy
 *   GET /api/ranked/<sub>          <submission>/ranked/<sub>_ranked_candidate_list.npy
 *   GET /api/labels/<sub>          tst_stim/<sub>_tst_stim.npy   (sub-01..04 only; every read is logged)
 *   GET /api/model/<M>/<sub>/pre    <models>/<M>/<sub>_pre.npy     fused scores before post-processing
 *   GET /api/model/<M>/<sub>/ranked <models>/<M>/<sub>_ranked.npy  team code's ranking, for checking
 *                                  M in C13, C12, C11, C10, WF (WF falls back to the submission dir)
 *   GET /api/sample/<sub>          tst/<sub>_tst.npy             (demo "select sample")
 *   GET /api/video/<id>            stimuli_test/<id>.mp4, with HTTP Range support
 *
 * Every path is built from a whitelisted pattern, never from raw user input, so
 * there is no path traversal. No authentication: the dev/preview servers bind to
 * 127.0.0.1 only (see vite.config.ts). Do not expose this to a network as is.
 * Locations come from server/paths.ts.
 */
import type { Plugin, Connect } from "vite";
import type { ServerResponse } from "node:http";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import type { Paths } from "./paths.ts";

const SUBJECT_RE = /^sub-0[1-6]$/;
// Ground truth for the held-out subjects does not exist and must never be served.
const LABELLED = new Set(["sub-01", "sub-02", "sub-03", "sub-04"]);
const MODEL_RE = /^(C1[0-3]|WF)$/;
const MODEL_IDS = ["C13", "C12", "C11", "C10", "WF"];
const FIRST_ID = 1001;
const LAST_ID = 1102;

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

function sha256File(p: string): string {
  return crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
}

const DEFAULT_CONFIG = { name: "unknown", scorers: 0, score_transform: "?", postprocess: "sinkhorn" };

export function buildManifest(P: Paths): Manifest {
  const cfgPath = path.join(P.submissionDir, "weights", "config.json");
  const cfg = fs.existsSync(cfgPath) ? JSON.parse(fs.readFileSync(cfgPath, "utf8")) : DEFAULT_CONFIG;
  const subjects: SubjectInfo[] = [];
  for (let i = 1; i <= 6; i++) {
    const id = `sub-0${i}`;
    const tst = samplePath(P, id);
    subjects.push({
      id,
      sha256: fs.existsSync(tst) ? sha256File(tst) : null,
      hasScores: fs.existsSync(scoresPath(P, id)),
      hasRanked: fs.existsSync(rankedPath(P, id)),
      hasLabels: LABELLED.has(id) && fs.existsSync(labelsPath(P, id)),
    });
  }
  const p = cfg.postprocess_params ?? { iters: 3, tau: 0.5 };
  return {
    subjects,
    config: {
      name: cfg.name,
      scorers: cfg.scorers,
      scoreTransform: cfg.score_transform,
      postprocess: cfg.postprocess,
      sinkhorn: { iters: p.iters, tau: p.tau },
    },
    candidates: { first: FIRST_ID, last: LAST_ID },
    models: Object.fromEntries(
      MODEL_IDS.map((mid) => [mid, subjects.map((s) => s.id).filter((s) => fs.existsSync(modelPath(P, mid, s, "pre")))]),
    ),
  };
}

const scoresPath = (P: Paths, s: string) => path.join(P.submissionDir, "scores", `${s}_fused_scores.npy`);
const rankedPath = (P: Paths, s: string) =>
  path.join(P.submissionDir, "ranked", `${s}_ranked_candidate_list.npy`);
const labelsPath = (P: Paths, s: string) => path.join(P.dataRoot, "tst_stim", `${s}_tst_stim.npy`);
const samplePath = (P: Paths, s: string) => path.join(P.dataRoot, "tst", `${s}_tst.npy`);
/** WF is winfuse submission2: use the exported copy if present, else the submission dir itself. */
function modelPath(P: Paths, model: string, sub: string, kind: "pre" | "ranked"): string {
  const p = path.join(P.modelsDir, model, `${sub}_${kind}.npy`);
  if (model === "WF" && !fs.existsSync(p)) return kind === "pre" ? scoresPath(P, sub) : rankedPath(P, sub);
  return p;
}

/** Append-only audit trail of ground-truth reads, like the pipelines' dev-test logs. */
function logLabelRead(P: Paths, sub: string, req: Connect.IncomingMessage) {
  fs.mkdirSync(path.dirname(P.labelLog), { recursive: true });
  const line = [new Date().toISOString(), sub, "app:/api/labels", req.headers["user-agent"] ?? "-"].join("\t");
  fs.appendFileSync(P.labelLog, line + "\n");
}

function send(res: ServerResponse, code: number, body: string, type = "text/plain") {
  res.statusCode = code;
  res.setHeader("Content-Type", type);
  res.end(body);
}

function sendFile(req: Connect.IncomingMessage, res: ServerResponse, p: string, type: string) {
  if (!fs.existsSync(p)) return send(res, 404, "not found");
  const st = fs.statSync(p);
  res.statusCode = 200;
  res.setHeader("Content-Type", type);
  res.setHeader("Content-Length", String(st.size));
  res.setHeader("Cache-Control", "no-cache");
  if (req.method === "HEAD") return res.end();
  fs.createReadStream(p).pipe(res);
}

function sendVideo(req: Connect.IncomingMessage, res: ServerResponse, p: string) {
  if (!fs.existsSync(p)) return send(res, 404, "not found");
  const size = fs.statSync(p).size;
  res.setHeader("Content-Type", "video/mp4");
  res.setHeader("Accept-Ranges", "bytes");
  res.setHeader("Cache-Control", "public, max-age=3600");
  const range = req.headers.range;
  if (!range) {
    res.statusCode = 200;
    res.setHeader("Content-Length", String(size));
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(p).pipe(res);
    return;
  }
  // Single ranges only; multi-range or malformed headers are refused explicitly
  // rather than silently answered with the whole file.
  const m = /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!m || (m[1] === "" && m[2] === "")) {
    res.statusCode = 416;
    res.setHeader("Content-Range", `bytes */${size}`);
    return res.end();
  }
  let start = m[1] === "" ? size - Number(m[2]) : Number(m[1]);
  let end = m[1] !== "" && m[2] !== "" ? Number(m[2]) : size - 1;
  start = Math.max(0, start);
  end = Math.min(end, size - 1);
  if (!(start <= end)) {
    res.statusCode = 416;
    res.setHeader("Content-Range", `bytes */${size}`);
    return res.end();
  }
  res.statusCode = 206;
  res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
  res.setHeader("Content-Length", String(end - start + 1));
  if (req.method === "HEAD") return res.end();
  fs.createReadStream(p, { start, end }).pipe(res);
}

export function createHandler(P: Paths): Connect.NextHandleFunction {
  let manifest: Manifest | null = null;
  return (req, res, next) => {
    const url = (req.url ?? "").split("?")[0];
    if (!url.startsWith("/api/")) return next();
    if (req.method !== "GET" && req.method !== "HEAD") return send(res, 405, "method not allowed");

    if (url === "/api/manifest") {
      manifest ??= buildManifest(P); // hashes are computed once
      return send(res, 200, JSON.stringify(manifest), "application/json");
    }

    const mm = /^\/api\/model\/([^/]+)\/([^/]+)\/(pre|ranked)$/.exec(url);
    if (mm) {
      const [, model, sub, kind] = mm;
      if (!MODEL_RE.test(model)) return send(res, 400, "model must be one of " + MODEL_IDS.join(", "));
      if (!SUBJECT_RE.test(sub)) return send(res, 400, "subject must be sub-01..sub-06");
      return sendFile(req, res, modelPath(P, model, sub, kind as "pre" | "ranked"), "application/octet-stream");
    }

    const m = /^\/api\/(scores|ranked|labels|sample|video)\/([^/]+)$/.exec(url);
    if (!m) return send(res, 404, "unknown endpoint");
    const [, kind, arg] = m;

    if (kind === "video") {
      const id = Number(arg);
      if (!/^\d{4}$/.test(arg) || id < FIRST_ID || id > LAST_ID)
        return send(res, 400, "video id must be 1001..1102");
      return sendVideo(req, res, path.join(P.dataRoot, "stimuli_test", `${arg}.mp4`));
    }

    if (!SUBJECT_RE.test(arg)) return send(res, 400, "subject must be sub-01..sub-06");
    const npy = "application/octet-stream";
    switch (kind) {
      case "scores":
        return sendFile(req, res, scoresPath(P, arg), npy);
      case "ranked":
        return sendFile(req, res, rankedPath(P, arg), npy);
      case "sample":
        return sendFile(req, res, samplePath(P, arg), npy);
      case "labels":
        if (!LABELLED.has(arg))
          return send(res, 403, "ground truth is not available for held-out subjects");
        if (req.method === "GET" && fs.existsSync(labelsPath(P, arg))) logLabelRead(P, arg, req);
        return sendFile(req, res, labelsPath(P, arg), npy);
    }
    return next();
  };
}

export function dataPlugin(paths: Paths): Plugin {
  const handler = createHandler(paths);
  return {
    name: "neurobytes-data",
    configureServer(server) {
      server.config.logger.info(
        `  data: ${paths.dataRoot}\n  submission: ${paths.submissionDir}\n  models: ${paths.modelsDir}\n  ground truth: sub-01..04 (reads logged to ${paths.labelLog})`,
      );
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}
