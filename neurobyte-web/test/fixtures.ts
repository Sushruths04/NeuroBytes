/**
 * Shared test fixtures. The challenge data and submission artefacts are never
 * committed, so tests that need them skip (with a reason) when they are absent —
 * e.g. on a fresh clone or in CI. Point NEUROBYTES_DATA_ROOT /
 * NEUROBYTES_SUBMISSION_DIR at them to run everything.
 */
import fs from "node:fs";
import path from "node:path";
import { resolvePaths } from "../server/paths.ts";

export const APP_DIR = path.resolve(import.meta.dirname, "..");
export const PATHS = resolvePaths(APP_DIR);

export const dataFile = (rel: string) => path.join(PATHS.dataRoot, rel);
export const subFile = (rel: string) => path.join(PATHS.submissionDir, rel);
export const modelFile = (rel: string) => path.join(PATHS.modelsDir, rel);

export const HAS_EEG = fs.existsSync(dataFile("tst/sub-01_tst.npy"));
export const HAS_LABELS = fs.existsSync(dataFile("tst_stim/sub-01_tst_stim.npy"));
export const HAS_VIDEOS = fs.existsSync(dataFile("stimuli_test/1001.mp4"));
export const HAS_SUBMISSION =
  fs.existsSync(subFile("scores/sub-01_fused_scores.npy")) && fs.existsSync(subFile("weights/config.json"));

export const HAS_MODELS = fs.existsSync(modelFile("C13/sub-01_pre.npy"));

export const asArrayBuffer = (b: Buffer) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
