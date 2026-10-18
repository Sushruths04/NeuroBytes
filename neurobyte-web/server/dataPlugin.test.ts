import { afterAll, beforeAll, describe, expect, it } from "vitest";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import crypto from "node:crypto";
import type { AddressInfo } from "node:net";
import { createHandler } from "./dataPlugin";
import { resolvePaths, type Paths } from "./paths";

/**
 * Runs against a synthetic data tree in a temp dir, so it needs no challenge
 * data and works on a fresh clone / CI.
 */
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), "neurobytes-"));
const DATA = path.join(TMP, "data");
const SUB = path.join(TMP, "submission");
const MOD = path.join(TMP, "models");
const VIDEO = crypto.randomBytes(5000);
const EEG = crypto.randomBytes(4096);

function write(p: string, b: Buffer | string) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, b);
}

async function serve(P: Paths) {
  const h = createHandler(P);
  const server = http.createServer((req, res) => h(req, res, () => { res.statusCode = 404; res.end("next"); }));
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  return { server, base: `http://127.0.0.1:${(server.address() as AddressInfo).port}` };
}

let off: { server: http.Server; base: string };
let on: { server: http.Server; base: string };
let pathsOn: Paths;

beforeAll(async () => {
  for (let i = 1; i <= 6; i++) {
    write(path.join(DATA, "tst", `sub-0${i}_tst.npy`), Buffer.concat([EEG, Buffer.from([i])]));
    write(path.join(DATA, "tst_stim", `sub-0${i}_tst_stim.npy`), `labels-${i}`);
    write(path.join(SUB, "scores", `sub-0${i}_fused_scores.npy`), `scores-${i}`);
    write(path.join(SUB, "ranked", `sub-0${i}_ranked_candidate_list.npy`), `ranked-${i}`);
    write(path.join(MOD, "C13", `sub-0${i}_pre.npy`), `c13-pre-${i}`);
    write(path.join(MOD, "C13", `sub-0${i}_ranked.npy`), `c13-ranked-${i}`);
  }
  write(path.join(MOD, "C12", "sub-01_pre.npy"), "c12-pre-1");
  write(path.join(DATA, "stimuli_test", "1001.mp4"), VIDEO);
  write(path.join(SUB, "weights", "config.json"), JSON.stringify({
    name: "test", scorers: 509, score_transform: "rrf+hub", postprocess: "sinkhorn",
    postprocess_params: { iters: 3, tau: 0.5 },
  }));
  write(path.join(TMP, "README.md"), "secret");

  const appDir = path.join(TMP, "app");
  const env = { NEUROBYTES_DATA_ROOT: DATA, NEUROBYTES_SUBMISSION_DIR: SUB, NEUROBYTES_MODELS_DIR: MOD };
  pathsOn = resolvePaths(appDir, env);
  off = on = await serve(pathsOn);
});

afterAll(async () => {
  await new Promise<void>((r) => off.server.close(() => r()));
  fs.rmSync(TMP, { recursive: true, force: true });
});

describe("data API", () => {
  it("manifest lists six subjects with sha256 fingerprints and the config", async () => {
    const m = await (await fetch(`${off.base}/api/manifest`)).json();
    expect(m.subjects.map((s: { id: string }) => s.id)).toEqual(["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]);
    const want = crypto.createHash("sha256").update(fs.readFileSync(path.join(DATA, "tst/sub-03_tst.npy"))).digest("hex");
    expect(m.subjects[2].sha256).toBe(want);
    expect(m.subjects.every((s: { hasScores: boolean }) => s.hasScores)).toBe(true);
    expect(m.config.sinkhorn).toEqual({ iters: 3, tau: 0.5 });
    expect(m.subjects.map((s: { hasLabels: boolean }) => s.hasLabels)).toEqual([true, true, true, true, false, false]);
    expect(m.models.C13).toHaveLength(6);
    expect(m.models.C12).toEqual(["sub-01"]);
    expect(m.models.WF).toHaveLength(6); // falls back to the submission dir
    expect(m.models.C10).toEqual([]);
  });

  it("serves scores/ranked/sample byte-for-byte; HEAD sends no body", async () => {
    expect(await (await fetch(`${off.base}/api/scores/sub-02`)).text()).toBe("scores-2");
    expect(await (await fetch(`${off.base}/api/ranked/sub-06`)).text()).toBe("ranked-6");
    const s = Buffer.from(await (await fetch(`${off.base}/api/sample/sub-01`)).arrayBuffer());
    expect(s.equals(fs.readFileSync(path.join(DATA, "tst/sub-01_tst.npy")))).toBe(true);
    const h = await fetch(`${off.base}/api/scores/sub-02`, { method: "HEAD" });
    expect(h.headers.get("content-length")).toBe("8");
    expect((await h.arrayBuffer()).byteLength).toBe(0);
  });

  it("ground truth: always served for sub-01..04, every read logged, never sub-05/06", async () => {
    expect(await (await fetch(`${on.base}/api/labels/sub-01`)).text()).toBe("labels-1");
    expect((await fetch(`${on.base}/api/labels/sub-05`)).status).toBe(403);
    expect((await fetch(`${on.base}/api/labels/sub-06`)).status).toBe(403);
    const log = fs.readFileSync(pathsOn.labelLog, "utf8").trim().split("\n");
    expect(log).toHaveLength(1);
    expect(log[0]).toMatch(/\tsub-01\tapp:\/api\/labels\t/);
    expect(fs.readFileSync(pathsOn.labelLog, "utf8")).not.toMatch(/sub-0[56]/);
  });

  it("serves model score matrices; WF falls back to the submission; rejects unknown models", async () => {
    expect(await (await fetch(`${on.base}/api/model/C13/sub-03/pre`)).text()).toBe("c13-pre-3");
    expect(await (await fetch(`${on.base}/api/model/C13/sub-03/ranked`)).text()).toBe("c13-ranked-3");
    expect(await (await fetch(`${on.base}/api/model/WF/sub-02/pre`)).text()).toBe("scores-2");
    expect(await (await fetch(`${on.base}/api/model/WF/sub-02/ranked`)).text()).toBe("ranked-2");
    expect((await fetch(`${on.base}/api/model/C10/sub-01/pre`)).status).toBe(404);
    for (const u of ["/api/model/C99/sub-01/pre", "/api/model/..%2F/sub-01/pre", "/api/model/C13/sub-09/pre"])
      expect((await fetch(on.base + u)).status, u).toBe(400);
    expect((await fetch(`${on.base}/api/model/C13/sub-01/other`)).status).toBe(404);
  });

  it("rejects bad input and path traversal", async () => {
    for (const u of ["/api/scores/sub-07", "/api/scores/..%2F..%2FREADME.md", "/api/video/0001", "/api/video/9999", "/api/video/1001.mp4"])
      expect((await fetch(off.base + u)).status, u).toBe(400);
    const t = await fetch(`${off.base}/api/scores/..%2F..%2F..%2FREADME.md`);
    expect(await t.text()).not.toBe("secret");
    expect((await fetch(`${off.base}/api/nope`)).status).toBe(404);
    expect((await fetch(`${off.base}/api/manifest`, { method: "POST" })).status).toBe(405);
  });

  it("streams video with single HTTP ranges and refuses multi-range", async () => {
    const r = await fetch(`${off.base}/api/video/1001`, { headers: { Range: "bytes=0-99" } });
    expect(r.status).toBe(206);
    expect(r.headers.get("content-range")).toBe(`bytes 0-99/${VIDEO.length}`);
    expect(Buffer.from(await r.arrayBuffer()).equals(VIDEO.subarray(0, 100))).toBe(true);
    const tail = await fetch(`${off.base}/api/video/1001`, { headers: { Range: "bytes=-10" } });
    expect(Buffer.from(await tail.arrayBuffer()).equals(VIDEO.subarray(-10))).toBe(true);
    expect((await fetch(`${off.base}/api/video/1001`, { headers: { Range: `bytes=${VIDEO.length + 10}-` } })).status).toBe(416);
    expect((await fetch(`${off.base}/api/video/1001`, { headers: { Range: "bytes=0-10,20-30" } })).status).toBe(416);
    expect((await fetch(`${off.base}/api/video/1002`)).status).toBe(404);
  });
});
