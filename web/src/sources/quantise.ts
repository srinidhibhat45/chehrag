/**
 * Quantisation for user-source vectors.
 *
 * Must produce byte-identical output to `pipeline/src/build_index.py`
 * (`project`, `binarise`, and the int8 passage quantiser). User and corpus
 * chunks fuse in one RRF pass under one calibrated confidence value, so a
 * difference here would leave the guardrail reading two incompatible score
 * scales.
 *
 * Its own module because the store needs the exact bytes twice — once to search
 * and once to persist — and doing this inside the index would quantise twice.
 */

import type { Strategy } from "./chunk";

/** Stable ids for persistence. Append only: reordering re-labels the chunks of
 *  every stored source, and the fusion weights then land on the wrong ones. */
export const STRATEGY_NAMES = [
  "whole", "sentence", "sliding", "contextual", "semantic", "document",
] as const satisfies readonly Strategy[];

export const STRATEGY_IDS: Record<Strategy, number> =
  Object.fromEntries(STRATEGY_NAMES.map((s, i) => [s, i])) as Record<Strategy, number>;

/**
 * Centre, project onto the PCA basis, re-L2-normalise.
 *
 * Mirrors `build_index.py::project`. The re-normalisation is not optional: PCA
 * does not preserve unit norm, and every downstream comparison assumes it.
 */
export function project(
  raw: Float32Array, pcaMean: Float32Array, pcaComp: Float32Array,
  dim: number, out: Float32Array,
): Float32Array {
  const rawDim = pcaMean.length;
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    let s = 0;
    const off = i * rawDim;
    for (let j = 0; j < rawDim; j++) s += pcaComp[off + j] * (raw[j] - pcaMean[j]);
    out[i] = s;
    norm += s * s;
  }
  norm = Math.sqrt(norm) || 1e-9;
  for (let i = 0; i < dim; i++) out[i] /= norm;
  return out;
}

/**
 * Sign bits packed little-endian into uint32 words.
 *
 * Vectors are L2-normalised, so the threshold is 0 and no per-dimension
 * statistics are needed. Matches `binarise()` in build_index.py and the query
 * encoder in ivf.ts.
 */
export function binarise(vec: Float32Array, dim: number, codeWords: number,
                         out: Uint32Array, offset: number): void {
  for (let w = 0; w < codeWords; w++) {
    let acc = 0;
    const base = w * 32;
    const lim = Math.min(32, dim - base);
    for (let b = 0; b < lim; b++) if (vec[base + b] > 0) acc |= 1 << b;
    out[offset + w] = acc >>> 0;
  }
}

/**
 * Per-vector int8 quantisation.
 *
 * The scale is the vector's own maximum absolute value rather than a global one,
 * which would flatten low-magnitude vectors into near-zero codes. The returned
 * scale is pre-multiplied by 127 so the scorer's `dot / 16129 * scale` lands on
 * cosine, matching `PassageRescorer`.
 */
export function quantiseInt8(vec: Float32Array, dim: number,
                             out: Int8Array, offset: number): number {
  let max = 0;
  for (let i = 0; i < dim; i++) { const a = Math.abs(vec[i]); if (a > max) max = a; }
  const scale = max > 0 ? max / 127 : 1;
  for (let i = 0; i < dim; i++) {
    const v = Math.round(vec[i] / scale);
    out[offset + i] = v > 127 ? 127 : v < -128 ? -128 : v;
  }
  return scale * 127;
}
