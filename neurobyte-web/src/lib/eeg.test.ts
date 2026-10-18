import { describe, expect, it } from "vitest";
import fs from "node:fs";
import crypto from "node:crypto";
import { channelActivity, loadRecording, N_CHANNELS, N_SAMPLES } from "./eeg";
import { writeNpyInt64 } from "./npy";
import { stageAt, STAGES, TOTAL_SECONDS } from "./pipeline";
import { asArrayBuffer as ab, dataFile, HAS_EEG } from "../../test/fixtures";

/** Minimal float64 .npy writer for synthetic EEG. */
function npyF8(values: Float64Array, shape: number[]): ArrayBuffer {
  let header = `{'descr': '<f8', 'fortran_order': False, 'shape': (${shape.join(", ")}), }`;
  const total = Math.ceil((10 + header.length + 1) / 64) * 64;
  header = header.padEnd(total - 11, " ") + "\n";
  const out = new ArrayBuffer(10 + header.length + values.length * 8);
  const u8 = new Uint8Array(out);
  u8.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out).setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) u8[10 + i] = header.charCodeAt(i);
  new Float64Array(out, 10 + header.length).set(values);
  return out;
}

describe("loadRecording", () => {
  it.skipIf(!HAS_EEG)("loads a real test set and fingerprints it like sha256sum", async () => {
    const buf = fs.readFileSync(dataFile("tst/sub-04_tst.npy"));
    const rec = await loadRecording(ab(buf), "sub-04_tst.npy");
    expect(rec.nTrials).toBe(102);
    expect(rec.sha256).toBe(crypto.createHash("sha256").update(buf).digest("hex"));
  });

  it("loads synthetic EEG; activity is per-channel and bounded (always runs)", async () => {
    const n = 2;
    const x = new Float64Array(n * N_CHANNELS * N_SAMPLES);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i / 7) * ((i / N_SAMPLES) % N_CHANNELS + 1);
    const buf = npyF8(x, [n, N_CHANNELS, N_SAMPLES]);
    const rec = await loadRecording(buf, "synthetic.npy");
    expect(rec.nTrials).toBe(n);
    expect(rec.sha256).toBe(crypto.createHash("sha256").update(Buffer.from(buf)).digest("hex"));
    const a = channelActivity(rec, 1, 0.2);
    expect(a).toHaveLength(N_CHANNELS);
    expect(a.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it("reports the expected shape for a wrong-shaped file, whatever its dtype", async () => {
    await expect(loadRecording(writeNpyInt64([1, 2, 3], [3]), "x.npy")).rejects.toThrow(/expected shape/);
  });

  it("rejects NaN values", async () => {
    const x = new Float64Array(N_CHANNELS * N_SAMPLES);
    x[0] = NaN;
    await expect(loadRecording(npyF8(x, [1, N_CHANNELS, N_SAMPLES]), "nan.npy")).rejects.toThrow(/NaN/);
  });
});

describe("stage timeline", () => {
  it("walks every stage in order and then finishes", () => {
    let acc = 0;
    STAGES.forEach((s, i) => {
      expect(stageAt(acc + s.seconds / 2).index).toBe(i);
      acc += s.seconds;
    });
    expect(stageAt(TOTAL_SECONDS + 0.01).index).toBe(STAGES.length);
  });
});
