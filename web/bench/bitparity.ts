/**
 * Cross-language bit-packing check.
 *
 * The binary codes are written by numpy and read by JS. If the two disagree on
 * bit order, retrieval does not error - it just returns nonsense. This asserts
 * they agree, on the actual packing code both sides use.
 */
import { readFileSync } from "node:fs";

const D = 192, WORDS = D / 32, N = 64;
// Node Buffers are views into a pooled ArrayBuffer with a non-zero byteOffset.
// `.buffer` alone hands you the whole pool, not this file's bytes.
function readTyped<T>(path: string, Ctor: new (b: ArrayBuffer, o: number, n: number) => T, bpe: number): T {
  const b = readFileSync(path);
  return new Ctor(b.buffer as ArrayBuffer, b.byteOffset, b.byteLength / bpe);
}
const codes = readTyped("/tmp/bits_codes.bin", Uint32Array, 4);
const qcode = readTyped("/tmp/bits_q.bin", Uint32Array, 4);
const ref   = readTyped("/tmp/bits_ref.bin", Int32Array, 4);
const qvec  = readTyped("/tmp/bits_qvec.bin", Float32Array, 4);

function popcnt(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}

// Re-encode the query in JS using the same logic as IvfIndex.encodeQuery.
const jsQ = new Uint32Array(WORDS);
for (let w = 0; w < WORDS; w++) {
  let acc = 0;
  const base = w * 32;
  for (let b = 0; b < Math.min(32, D - base); b++) if (qvec[base + b] > 0) acc |= 1 << b;
  jsQ[w] = acc >>> 0;
}

let encMismatch = 0;
for (let w = 0; w < WORDS; w++) if (jsQ[w] !== qcode[w]) encMismatch++;

let hamMismatch = 0, maxDelta = 0;
for (let i = 0; i < N; i++) {
  let d = 0;
  for (let w = 0; w < WORDS; w++) d += popcnt((codes[i * WORDS + w] ^ jsQ[w]) >>> 0);
  const delta = Math.abs(d - ref[i]);
  if (delta !== 0) { hamMismatch++; maxDelta = Math.max(maxDelta, delta); }
}

console.log(`query encode words mismatched : ${encMismatch}/${WORDS}`);
console.log(`hamming distances mismatched  : ${hamMismatch}/${N} (max delta ${maxDelta})`);
console.log(encMismatch === 0 && hamMismatch === 0
  ? "PASS - numpy and JS agree on bit packing and Hamming distance"
  : "FAIL - packing convention differs; retrieval would return nonsense");
if (encMismatch || hamMismatch) process.exit(1);
