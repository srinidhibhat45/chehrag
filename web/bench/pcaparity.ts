/**
 * Cross-language PCA projection check.
 *
 * Same silent-failure class as bit packing: if JS and numpy disagree on the
 * component matrix layout (row- vs column-major), projection still produces
 * numbers, they are just the wrong numbers, and every query lands in the wrong
 * region of the space.
 */
import { readFileSync } from "node:fs";

function readF32(path: string): Float32Array {
  const b = readFileSync(path);
  return new Float32Array(b.buffer as ArrayBuffer, b.byteOffset, b.byteLength / 4);
}
const IN = 384, OUT = 192, N = 32;
const mean = readF32("/tmp/pca_mean.bin");
const comp = readF32("/tmp/pca_comp.bin");
const input = readF32("/tmp/pca_in.bin");
const expect = readF32("/tmp/pca_out.bin");

// Exactly the projectQuery body from loader.ts.
function projectQuery(raw: Float32Array, out: Float32Array): void {
  let norm = 0;
  for (let i = 0; i < OUT; i++) {
    let s = 0;
    const off = i * IN;
    for (let j = 0; j < IN; j++) s += comp[off + j] * (raw[j] - mean[j]);
    out[i] = s;
    norm += s * s;
  }
  norm = Math.sqrt(norm) || 1e-9;
  for (let i = 0; i < OUT; i++) out[i] /= norm;
}

let worstCos = 1, worstAbs = 0;
const out = new Float32Array(OUT);
for (let n = 0; n < N; n++) {
  projectQuery(input.subarray(n * IN, (n + 1) * IN), out);
  const exp = expect.subarray(n * OUT, (n + 1) * OUT);
  let dot = 0;
  for (let i = 0; i < OUT; i++) {
    dot += out[i] * exp[i];
    worstAbs = Math.max(worstAbs, Math.abs(out[i] - exp[i]));
  }
  worstCos = Math.min(worstCos, dot);
}
console.log(`worst cosine vs numpy : ${worstCos.toFixed(7)}`);
console.log(`max abs elementwise   : ${worstAbs.toExponential(2)}`);
console.log(worstCos > 0.99999
  ? "PASS — JS projection matches numpy"
  : "FAIL — projection layout differs; every query would land in the wrong space");
if (worstCos <= 0.99999) process.exit(1);
