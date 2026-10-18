import { describe, expect, it } from "vitest";
import fs from "node:fs";
import { NpyError, parseNpy, writeNpyInt64 } from "./npy";
import { asArrayBuffer as ab, dataFile, HAS_EEG, HAS_LABELS, HAS_SUBMISSION, subFile } from "../../test/fixtures";

describe("parseNpy", () => {
  it.skipIf(!HAS_EEG)("reads a real float64 EEG file", () => {
    const a = parseNpy(ab(fs.readFileSync(dataFile("tst/sub-01_tst.npy"))));
    expect(a.dtype).toBe("<f8");
    expect(a.shape).toEqual([102, 17, 1500]);
    expect(a.data.length).toBe(102 * 17 * 1500);
    expect(Number.isFinite(a.data[12345])).toBe(true);
  });

  it.skipIf(!HAS_SUBMISSION || !HAS_LABELS)("reads float32 scores and int64 labels", () => {
    const s = parseNpy(ab(fs.readFileSync(subFile("scores/sub-01_fused_scores.npy"))));
    expect(s.dtype).toBe("<f4");
    expect(s.shape).toEqual([102, 102]);
    const l = parseNpy(ab(fs.readFileSync(dataFile("tst_stim/sub-01_tst_stim.npy"))));
    expect(l.dtype).toBe("<i8");
    expect(l.shape).toEqual([102]);
  });

  it.skipIf(!HAS_SUBMISSION)("writeNpyInt64 output is byte-identical to numpy's for the shipped ranking", () => {
    const orig = fs.readFileSync(subFile("ranked/sub-05_ranked_candidate_list.npy"));
    const a = parseNpy(ab(orig));
    const out = Buffer.from(writeNpyInt64(a.data, a.shape));
    expect(out.equals(orig)).toBe(true);
  });

  it("round-trips int64 (synthetic, always runs)", () => {
    const vals = Array.from({ length: 102 * 102 }, (_, i) => 1001 + (i % 102));
    const a = parseNpy(writeNpyInt64(vals, [102, 102]));
    expect(a.dtype).toBe("<i8");
    expect(a.shape).toEqual([102, 102]);
    expect(Array.from(a.data)).toEqual(vals);
    const one = parseNpy(writeNpyInt64([7, 8, 9], [3]));
    expect(one.shape).toEqual([3]);
  });

  it("rejects non-npy and unsupported input", () => {
    expect(() => parseNpy(new ArrayBuffer(4))).toThrow(NpyError);
    expect(() => parseNpy(new TextEncoder().encode("hello world, not numpy").buffer)).toThrow(/magic/);
    const bad = Buffer.from(writeNpyInt64([1, 2], [2])).toString("latin1").replace("<i8", "<c8");
    expect(() => parseNpy(ab(Buffer.from(bad, "latin1")))).toThrow(/unsupported dtype/);
    const trunc = Buffer.from(writeNpyInt64([1, 2, 3], [3])).subarray(0, 80);
    expect(() => parseNpy(ab(trunc))).toThrow(/truncated/);
  });
});
