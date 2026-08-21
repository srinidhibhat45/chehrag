/**
 * BM25 over the shipped corpus — the one index that does not embed anything.
 *
 * The six dense indices differ in *what* they embed. This one differs in the
 * matching rule: exact terms, weighted by how rare they are and how long the
 * passage is. It exists because a 384-dimension vector averaged over a passage
 * cannot hold an exact token — a year, an acronym, a surname — and a quarter of
 * this corpus's queries are numeric. See `pipeline/src/lexical.py` for the build
 * side and the argument in full.
 *
 * Shape of the shipped index, and why:
 *
 *   hashHi/hashLo   the term dictionary, as sorted 64-bit FNV-1a hashes rather
 *                   than strings. Binary-searched in place: no JSON parse at
 *                   boot, no string comparison per lookup.
 *   offsets         prefix offsets into the postings, so df is a subtraction.
 *   docs/tf         postings, ascending by passage within each term.
 *   norm            k1 * (1 - b + b * len/avgLen), precomputed per passage.
 *                   It is the whole of the inner loop's arithmetic that does
 *                   not depend on the query, so it is not paid per query.
 *
 * P100 discipline, same as `ivf.ts`: every buffer is allocated at construction,
 * the accumulator is cleared by walking the touched list rather than the whole
 * corpus, and the candidate set is capped so a query of very common terms cannot
 * cost more than a query of rare ones.
 */

import type { StrategySearchResult } from "./ivf";

export interface LexicalIndexData {
  hashHi: Uint32Array;
  hashLo: Uint32Array;
  offsets: Uint32Array;   // terms + 1
  docs: Uint32Array;      // postings
  tf: Uint8Array;         // postings
  lengths: Uint16Array;   // numPassages — content tokens per passage
  numPassages: number;
  avgLen: number;
  k1: number;
  b: number;
}

/**
 * 64-bit FNV-1a, in 32-bit halves.
 *
 * A port of `lexical.py::fnv1a64`, and the one place where the browser and the
 * builder must agree bit for bit: a mismatch is not a crash but a silent miss —
 * every lookup fails and the index returns nothing at all, which is
 * indistinguishable from a query with no rare terms. `bench/lexparity.ts`
 * asserts the two agree over real corpus vocabulary.
 *
 * BigInt would express this in three lines. It is avoided because this runs per
 * query term inside the measured budget, and because the arithmetic below is
 * exact: the 16-bit partial products are all under 2^32, so every intermediate
 * is an integer a double represents exactly.
 */
const ENC = new TextEncoder();

export function fnv1a64(s: string): { hi: number; lo: number } {
  // Offset basis 0xCBF29CE484222325, in halves.
  let hi = 0xcbf29ce4, lo = 0x84222325;
  const bytes = ENC.encode(s);
  for (let i = 0; i < bytes.length; i++) {
    lo = (lo ^ bytes[i]) >>> 0;
    // Multiply by the FNV prime 0x100000001B3. Split across 32-bit halves that
    // is primeHi = 0x100 and primeLo = 0x1B3 — the prime is 2^40 + 0x1B3, so
    // its high word is 256 and not 1, which is the easiest thing in this
    // function to get wrong and costs only the high half of every hash.
    // Modulo 2^64:
    //
    //   lo'  = (lo * primeLo) mod 2^32
    //   hi'  = (carry out of the above) + hi * primeLo + lo * primeHi
    //
    // `lo * 0x1B3` is under 2^41 and `lo * 0x100` under 2^40, so both are
    // exact doubles — no 16-bit splitting, which is the other easy thing to
    // get wrong here.
    const full = lo * 0x1b3;
    const nlo = full >>> 0;
    hi = (Math.floor(full / 4294967296) + Math.imul(hi, 0x1b3) + lo * 0x100) >>> 0;
    lo = nlo;
  }
  return { hi, lo };
}

export class LexicalIndex {
  private readonly d: LexicalIndexData;
  private readonly terms: number;

  /** Per-passage BM25 length normalisation, precomputed at construction. */
  private readonly norm: Float32Array;

  // Scratch, allocated once.
  private readonly scores: Float32Array;
  private readonly touched: Int32Array;
  private readonly outParents: Int32Array;
  private readonly outScores: Float32Array;
  private readonly result: StrategySearchResult;

  readonly topK: number;
  /**
   * Ceiling on postings visited per query.
   *
   * Without it, cost is set by the commonest term the user happens to say. The
   * df ceiling in the builder already removes the worst offenders; this bounds
   * what is left, including the case of a query with many moderately common
   * terms. Terms are visited rarest-first, so what a truncation drops is always
   * the least informative part of the query.
   */
  readonly maxPostings: number;

  constructor(data: LexicalIndexData, topK = 48, maxPostings = 60_000) {
    this.d = data;
    this.terms = data.offsets.length - 1;
    this.topK = topK;
    this.maxPostings = maxPostings;

    const { k1, b, avgLen, lengths, numPassages } = data;
    this.norm = new Float32Array(numPassages);
    for (let i = 0; i < numPassages; i++) {
      this.norm[i] = k1 * (1 - b + (b * lengths[i]) / (avgLen || 1));
    }

    this.scores = new Float32Array(numPassages);
    this.touched = new Int32Array(numPassages);
    this.outParents = new Int32Array(topK);
    this.outScores = new Float32Array(topK);
    this.result = { parentIds: this.outParents, scores: this.outScores, n: 0 };
  }

  /** Term slot for a hash, or -1. Binary search over the sorted dictionary. */
  private lookup(hi: number, lo: number): number {
    const { hashHi, hashLo } = this.d;
    let lowIdx = 0, highIdx = this.terms - 1;
    while (lowIdx <= highIdx) {
      const mid = (lowIdx + highIdx) >>> 1;
      const mhi = hashHi[mid];
      if (mhi < hi) { lowIdx = mid + 1; continue; }
      if (mhi > hi) { highIdx = mid - 1; continue; }
      const mlo = hashLo[mid];
      if (mlo < lo) { lowIdx = mid + 1; continue; }
      if (mlo > lo) { highIdx = mid - 1; continue; }
      return mid;
    }
    return -1;
  }

  /**
   * Score the corpus against a query's content tokens.
   *
   * @param tokens de-duplicated content tokens, from `contentTokens`. The same
   *               function the builder used, which is what makes a lookup able
   *               to hit at all.
   * @param k      passages to return
   */
  search(tokens: Iterable<string>, k: number): StrategySearchResult {
    const K = Math.min(k, this.topK);
    const { offsets, docs, tf, numPassages, k1 } = this.d;
    const scores = this.scores;
    const touched = this.touched;
    const norm = this.norm;
    let nTouched = 0;

    // Resolve every term first, so they can be visited rarest-first. IDF is
    // computed here too: it is per-term, not per-posting.
    const slots: number[] = [];
    const idfs: number[] = [];
    for (const t of tokens) {
      const h = fnv1a64(t);
      const slot = this.lookup(h.hi, h.lo);
      if (slot < 0) continue;
      const df = offsets[slot + 1] - offsets[slot];
      if (df === 0) continue;
      // Robertson-Sparck-Jones IDF with the +1 that keeps it positive: without
      // it a term in over half the corpus scores negative and *penalises* the
      // passages containing it.
      idfs.push(Math.log(1 + (numPassages - df + 0.5) / (df + 0.5)));
      slots.push(slot);
    }
    if (!slots.length) { this.result.n = 0; return this.result; }

    // Rarest first: highest IDF carries the most information, so if the
    // postings cap truncates, it truncates the least useful terms.
    const order = slots.map((_, i) => i).sort((a, b) => idfs[b] - idfs[a]);

    let visited = 0;
    for (const oi of order) {
      const slot = slots[oi];
      const idf = idfs[oi];
      const start = offsets[slot], end = offsets[slot + 1];
      if (visited >= this.maxPostings) break;
      for (let p = start; p < end; p++) {
        const doc = docs[p];
        const f = tf[p];
        // BM25 term contribution. `norm[doc]` already carries k1, b and the
        // length ratio.
        const add = (idf * f * (k1 + 1)) / (f + norm[doc]);
        if (scores[doc] === 0) {
          // First time this passage is seen in this query.
          if (nTouched < numPassages) touched[nTouched++] = doc;
        }
        scores[doc] += add;
      }
      visited += end - start;
    }

    // ---- top-K by insertion ------------------------------------------------
    // K is ~24 and the candidate set is tens of thousands, so a full sort would
    // cost more than the scan that produced it.
    const parents = this.outParents;
    const outScores = this.outScores;
    let n = 0;
    let worst = -Infinity;
    for (let i = 0; i < nTouched; i++) {
      const doc = touched[i];
      const s = scores[doc];
      if (n === K && s <= worst) continue;
      let pos = n < K ? n : K - 1;
      while (pos > 0 && outScores[pos - 1] < s) {
        outScores[pos] = outScores[pos - 1];
        parents[pos] = parents[pos - 1];
        pos--;
      }
      outScores[pos] = s;
      parents[pos] = doc;
      if (n < K) n++;
      worst = outScores[n - 1];
    }

    // Clear only what was written. Walking 99k entries per query to zero a few
    // thousand would cost more than the search.
    for (let i = 0; i < nTouched; i++) scores[touched[i]] = 0;

    // Rescaled to [0,1] against this query's own best hit.
    //
    // BM25 is unbounded and lives on a completely different scale from cosine.
    // Fusion reads rank, not score, so ranking is unaffected either way — but
    // `FusedHit.bestRawScore` takes a max across strategies, and an unbounded
    // number there would dominate a field of cosines. In normal operation the
    // rescore stage overwrites that field with a real int8 cosine before gate 2
    // reads it; this keeps the value sane in the degraded path where rescoring
    // was skipped, rather than relying on that one guarantee alone.
    const top = n ? outScores[0] : 0;
    if (top > 0) for (let i = 0; i < n; i++) outScores[i] /= top;

    this.result.n = n;
    return this.result;
  }
}
