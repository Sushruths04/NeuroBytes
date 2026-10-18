/**
 * Where the app finds its data, following the repo's data-handling convention
 * (see the top-level README "Data" section and docker/entrypoint.sh):
 *
 *   The official CHALLENGE DATA (EEG recordings tst/, dev labels tst_stim/, stimulus
 *   videos stimuli_test/) is NEVER redistributed in this repository. It is supplied by
 *   the organizers and passed in from OUTSIDE the repo — natively as DATA=/path/to/
 *   eeg-mind-reading-challenge, and in Docker as a read-only mount at /data
 *   (-v "$D:/data:ro"), exactly like the C13 inference image (WINFUSE_DATA_ROOT=/data).
 *
 *   The repo's OWN shipped artefacts stay in the repo: the winfuse fused scores
 *   (results/winfuse) and the per-model matrices (results/neurobytes_models, written by
 *   scripts/export_models.py from combo/). These are what the browser ranks with.
 *
 * Environment:
 *   DATA / NEUROBYTES_DATA_ROOT   challenge folder with tst/, tst_stim/, stimuli_test/
 *                                 (default /data in-container; a sibling checkout natively)
 *   NEUROBYTES_SUBMISSION_DIR     winfuse dir: scores/, ranked/, weights/config.json
 *   NEUROBYTES_MODELS_DIR         <model>/<sub>_pre.npy, <model>/<sub>_ranked.npy
 *   NEUROBYTES_LABEL_LOG          file that every ground-truth read is appended to
 *   NEUROBYTES_CAPTIONS           captions.json of Qwen2.5-VL video captions (display only;
 *                                 default <data>/captions.json, not shipped in the repo)
 *
 * Ground truth (tst_stim/) is served for sub-01..04 only, and every read is logged.
 */
import fs from "node:fs";
import path from "node:path";

export interface Paths {
  dataRoot: string;
  submissionDir: string;
  modelsDir: string;
  /** Qwen2.5-VL one-line captions per video id; display only, never a model input */
  captions: string;
  /** every ground-truth read is appended here */
  labelLog: string;
}

const isDir = (p: string) => fs.existsSync(p) && fs.statSync(p).isDirectory();
const hasChallengeData = (p: string) => isDir(path.join(p, "tst")) || isDir(path.join(p, "tst_stim"));

export function resolvePaths(appDir: string, env: NodeJS.ProcessEnv = process.env): Paths {
  // repo = the neurobytes repo root (neurobyte-web lives directly inside it).
  const repo = path.resolve(appDir, "..");

  // Challenge data is external, never bundled. Follow the repo's DATA= convention, and
  // default to the /data mount point used by the inference image. A few repo-relative
  // candidates are tried natively so a sibling checkout "just works".
  const dataCandidates = [
    env.NEUROBYTES_DATA_ROOT,
    env.DATA,
    "/data",
    path.resolve(repo, "..", "eeg-mind-reading-challenge"),
    path.resolve(repo, "..", "data", "challenge_official", "eeg-mind-reading-challenge"),
    repo,
  ].filter((p): p is string => !!p);
  const explicit = env.NEUROBYTES_DATA_ROOT ?? env.DATA;
  const dataRoot = path.resolve(
    explicit ?? dataCandidates.find((p) => hasChallengeData(p)) ?? "/data",
  );

  // The repo's own winfuse artefacts (shipped in the repo).
  const subCandidates = [
    env.NEUROBYTES_SUBMISSION_DIR,
    path.join(repo, "results", "winfuse"),
    path.join(dataRoot, "results", "winfuse"),
  ].filter((p): p is string => !!p);
  const submissionDir = path.resolve(
    env.NEUROBYTES_SUBMISSION_DIR ?? subCandidates.find((p) => isDir(path.join(p, "scores"))) ?? subCandidates[0],
  );

  // The exported per-model matrices (regenerable, under results/, written in-repo).
  const modelCandidates = [
    env.NEUROBYTES_MODELS_DIR,
    path.join(repo, "results", "neurobytes_models"),
    path.join(dataRoot, "results", "neurobytes_models"),
  ].filter((p): p is string => !!p);
  const modelsDir = path.resolve(
    env.NEUROBYTES_MODELS_DIR ?? modelCandidates.find((p) => isDir(p)) ?? modelCandidates[0],
  );

  // Captions are derived from the stimulus videos, so like the challenge data they are not shipped.
  const captionCandidates = [
    path.join(dataRoot, "captions.json"),
    path.resolve(repo, "..", "outputs", "mindread_features", "captions.json"),
    path.resolve(dataRoot, "..", "..", "..", "outputs", "mindread_features", "captions.json"),
  ];
  const captions = path.resolve(
    env.NEUROBYTES_CAPTIONS ?? captionCandidates.find((p) => fs.existsSync(p)) ?? captionCandidates[0],
  );

  return {
    dataRoot,
    submissionDir,
    modelsDir,
    captions,
    labelLog: path.resolve(env.NEUROBYTES_LABEL_LOG ?? path.join(appDir, "logs", "label_access.log")),
  };
}
