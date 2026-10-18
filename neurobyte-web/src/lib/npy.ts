/**
 * Minimal reader/writer for NumPy .npy files (format versions 1.0, 2.0, 3.0).
 * Supports little-endian float64/float32/int64/int32, C order only — exactly what
 * this project's files use. Anything else is rejected with a clear error.
 */
export type NpyDtype = "<f8" | "<f4" | "<i8" | "<i4";

export interface NpyArray {
  dtype: NpyDtype;
  shape: number[];
  /** Values as float64, row-major. int64 is converted (safe for |v| < 2^53). */
  data: Float64Array;
}

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

export class NpyError extends Error {}

export function parseNpy(buf: ArrayBuffer): NpyArray {
  const u8 = new Uint8Array(buf);
  if (u8.length < 10 || MAGIC.some((b, i) => u8[i] !== b))
    throw new NpyError("not a .npy file (bad magic bytes)");
  const major = u8[6];
  const dv = new DataView(buf);
  let headerLen: number, offset: number;
  if (major === 1) {
    headerLen = dv.getUint16(8, true);
    offset = 10;
  } else if (major === 2 || major === 3) {
    headerLen = dv.getUint32(8, true);
    offset = 12;
  } else throw new NpyError(`unsupported .npy version ${major}`);

  const header = new TextDecoder(major === 3 ? "utf-8" : "latin1").decode(
    u8.subarray(offset, offset + headerLen),
  );
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shapeStr = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!descr || !fortran || shapeStr === undefined) throw new NpyError("malformed .npy header");
  if (fortran === "True") throw new NpyError("Fortran-ordered arrays are not supported");
  const dtype = descr.replace(/^\|/, "<") as NpyDtype;
  if (!["<f8", "<f4", "<i8", "<i4"].includes(dtype))
    throw new NpyError(`unsupported dtype ${descr} (need float64/float32/int64/int32)`);

  const shape = shapeStr
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const n = shape.reduce((a, b) => a * b, 1);
  const start = offset + headerLen;
  const itemsize = dtype === "<f8" || dtype === "<i8" ? 8 : 4;
  if (buf.byteLength - start < n * itemsize)
    throw new NpyError(`file truncated: expected ${n * itemsize} data bytes`);

  // Copy so the typed-array view is aligned regardless of header length.
  const raw = buf.slice(start, start + n * itemsize);
  let data: Float64Array;
  switch (dtype) {
    case "<f8":
      data = new Float64Array(raw);
      break;
    case "<f4":
      data = Float64Array.from(new Float32Array(raw));
      break;
    case "<i4":
      data = Float64Array.from(new Int32Array(raw));
      break;
    case "<i8": {
      const b = new BigInt64Array(raw);
      data = new Float64Array(n);
      for (let i = 0; i < n; i++) data[i] = Number(b[i]);
      break;
    }
  }
  return { dtype, shape, data };
}

/** Serialise an int64 C-order array as .npy v1.0 (the submission format). */
export function writeNpyInt64(values: ArrayLike<number>, shape: number[]): ArrayBuffer {
  const shapeTxt = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
  let header = `{'descr': '<i8', 'fortran_order': False, 'shape': ${shapeTxt}, }`;
  // Pad so that magic(6)+ver(2)+len(2)+header is a multiple of 64, ending in \n.
  const total = Math.ceil((10 + header.length + 1) / 64) * 64;
  header = header.padEnd(total - 10 - 1, " ") + "\n";
  const out = new ArrayBuffer(10 + header.length + values.length * 8);
  const u8 = new Uint8Array(out);
  u8.set(MAGIC, 0);
  u8[6] = 1;
  u8[7] = 0;
  new DataView(out).setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) u8[10 + i] = header.charCodeAt(i);
  const body = new BigInt64Array(out, 10 + header.length, values.length);
  for (let i = 0; i < values.length; i++) body[i] = BigInt(Math.round(values[i]));
  return out;
}
