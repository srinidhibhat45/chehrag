/**
 * Search structure for user-added sources.
 *
 * Flat rather than IVF. `retrieval/ivf.ts` exists because 810k corpus chunks
 * cannot be scanned inside the budget; user sources are three or four orders of
 * magnitude smaller, and clustering them would mean running k-means in the
 * browser on every document added — seconds of work to save microseconds of
 * query, plus a recall loss on the corpus the user cares most about.
 *
 * What is load-bearing for P100: a flat scan is O(n) in a number the user
 * controls, so it is hard-capped. Beyond `maxScan` chunks the index stops
 * looking and reports that it did.
 *
 * Results come back bucketed by chunking strategy, matching the shape the corpus
 * indices return, because fusion weights are per-strategy — a `document` hit and
 * a `whole` hit are not worth the same.
 *
 * Layout and scoring match the corpus index exactly — same PCA space, same
 * sign-bit binarisation, same int8 rescoring — so hits from both fuse without
 * per-source normalisation.
 */

import type { Strategy } from "./chunk";

/** Ceiling on chunks scanned per query. At 8 words per code this is ~1.2M
 *  popcounts, measured at ~2ms — inside the retrieve stage's 80ms budget with
 *  room for the six corpus indices alongside. */
const MAX_SCAN = 150_000;
/** Candidate pool held during the scan. Wide enough that the rarer strategies
 *  still reach their bucket, small enough that the replace-worst step stays cheap. */
const POOL = 512;
const GROW = 4096;

export interface StrategyBucket {
  strategy: Strategy;
  parentIds: Int32Array;
  scores: Float32Array;
  n: number;
}

/** Same routine as `ivf.ts`. `Math.imul` and `>>>` are correctness, not style:
 *  plain `*` overflows into a double and the following shift returns a wrong
 *  count, ranking results by garbage without erroring. */
function popcnt(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}

export interface ChunkRef {
  /** Global user-passage ordinal. */
  passage: number;
  strategy: Strategy;
  /** Which source this came from, so a disabled source can be skipped in-scan. */
  source: number;
}

export class UserIndex {
  private codes: Uint32Array;
  private refs: ChunkRef[] = [];
  private count = 0;

  /** Per-passage int8 vectors for rescoring, mirroring PassageRescorer. */
  private p8: Int8Array;
  private pScale: Float32Array;
  private pCount = 0;

  private qCode: Uint32Array;
  private qInt8: Int8Array;
  private candIdx = new Int32Array(POOL);
  private candDist = new Int32Array(POOL);

  /** One preallocated bucket per strategy; reused every query. */
  private buckets = new Map<Strategy, StrategyBucket>();
  private seen = new Map<Strategy, Set<number>>();
  private out: StrategyBucket[] = [];

  /** True when the last search hit the scan ceiling. Surfaced in telemetry, so a
   *  truncated search is visible rather than silently partial. */
  truncated = false;

  /** Source ids currently excluded from search. */
  readonly disabled = new Set<number>();

  constructor(
    private readonly dim: number,
    private readonly codeWords: number,
    private readonly topK = 48,
    private readonly maxScan = MAX_SCAN,
  ) {
    this.codes = new Uint32Array(GROW * codeWords);
    this.p8 = new Int8Array(GROW * dim);
    this.pScale = new Float32Array(GROW);
    this.qCode = new Uint32Array(codeWords);
    this.qInt8 = new Int8Array(dim);
    for (const s of ["whole", "sentence", "sliding", "contextual", "semantic", "document"] as Strategy[]) {
      this.buckets.set(s, {
        strategy: s, parentIds: new Int32Array(topK), scores: new Float32Array(topK), n: 0,
      });
      this.seen.set(s, new Set());
    }
  }

  get size(): number { return this.count; }
  get passageCount(): number { return this.pCount; }

  // -- building ------------------------------------------------------------

  /**
   * Register a block of already-quantised passage vectors, returning the
   * user-passage ordinal of the first.
   *
   * Quantisation happens in `quantise.ts` so the store holds the exact bytes it
   * persists; doing it here would quantise once for search and again for
   * IndexedDB.
   */
  addPassages(int8: Int8Array, scale: Float32Array): number {
    const n = scale.length;
    const base = this.pCount;
    if (base + n > this.pScale.length) {
      const size = base + n + GROW;
      this.p8 = grow8(this.p8, size * this.dim);
      this.pScale = growF(this.pScale, size);
    }
    this.p8.set(int8.subarray(0, n * this.dim), base * this.dim);
    this.pScale.set(scale.subarray(0, n), base);
    this.pCount += n;
    return base;
  }

  /**
   * Register a block of chunk codes.
   *
   * @param codes      packed binary codes, `n * codeWords` words
   * @param parents    global user-passage ordinal per chunk
   * @param strategies strategy id per chunk, indexing into `names`
   */
  addChunks(codes: Uint32Array, parents: Int32Array, strategies: Uint8Array,
            source: number, names: readonly Strategy[]): void {
    const n = parents.length;
    const cw = this.codeWords;
    if (this.count + n > this.codes.length / cw) {
      this.codes = grow32(this.codes, (this.count + n + GROW) * cw);
    }
    this.codes.set(codes.subarray(0, n * cw), this.count * cw);
    for (let i = 0; i < n; i++) {
      this.refs.push({ passage: parents[i], strategy: names[strategies[i]], source });
    }
    this.count += n;
  }

  // -- readback, for persistence -------------------------------------------
  //
  // The store decides what is written to IndexedDB, but the bytes live here.
  // These copy out one source's slice without keeping a duplicate of every
  // array in memory for the whole session.

  refsOf(source: number): Array<ChunkRef & { slot: number }> {
    const out: Array<ChunkRef & { slot: number }> = [];
    for (let i = 0; i < this.count; i++) {
      if (this.refs[i].source === source) out.push({ ...this.refs[i], slot: i });
    }
    return out;
  }

  copyCode(slot: number, dest: Uint32Array, at: number): void {
    const cw = this.codeWords;
    dest.set(this.codes.subarray(slot * cw, slot * cw + cw), at);
  }

  copyPassage(passageId: number, dest: Int8Array, at: number): void {
    const d = this.dim;
    dest.set(this.p8.subarray(passageId * d, passageId * d + d), at);
  }

  scaleOf(passageId: number): number { return this.pScale[passageId]; }

  // -- mutation ------------------------------------------------------------

  /** Drop every chunk belonging to a source. */
  removeSource(source: number): void {
    if (!this.refs.some((r) => r.source === source)) return;
    // Compact codes in place. Passage vectors stay put: their ordinals are
    // permanent keys for every surviving chunk, so renumbering would mean
    // rewriting every ref. The orphans are unreachable, not wrong.
    let w = 0;
    const cw = this.codeWords;
    const kept: ChunkRef[] = [];
    for (let r = 0; r < this.count; r++) {
      if (this.refs[r].source === source) continue;
      if (w !== r) this.codes.copyWithin(w * cw, r * cw, r * cw + cw);
      kept.push(this.refs[r]);
      w++;
    }
    this.refs = kept;
    this.count = w;
    this.disabled.delete(source);
  }

  clear(): void {
    this.refs = [];
    this.count = 0;
    this.pCount = 0;
    this.disabled.clear();
  }

  // -- search --------------------------------------------------------------

  /**
   * Rank user passages against a projected query vector, bucketed by strategy.
   *
   * @param q reduced, L2-normalised query vector (same space as the corpus)
   * @param k passages per strategy
   */
  search(q: Float32Array, k: number): StrategyBucket[] {
    this.out.length = 0;
    this.truncated = false;
    if (this.count === 0) return this.out;

    const K = Math.min(k, this.topK);
    const cw = this.codeWords;
    const qc = this.qCode;
    for (let w = 0; w < cw; w++) {
      let acc = 0;
      const base = w * 32;
      const lim = Math.min(32, this.dim - base);
      for (let b = 0; b < lim; b++) if (q[base + b] > 0) acc |= 1 << b;
      qc[w] = acc >>> 0;
    }

    // Bounded top-POOL selection during the scan itself. An inline pool avoids a
    // second pass over the full array and keeps the memory touched per query
    // constant however much the user has added.
    const codes = this.codes;
    const skipAny = this.disabled.size > 0;
    const limit = Math.min(this.count, this.maxScan);
    if (limit < this.count) this.truncated = true;

    let m = 0;
    let worstAt = 0;

    for (let i = 0; i < limit; i++) {
      if (skipAny && this.disabled.has(this.refs[i].source)) continue;
      let dist = 0;
      const off = i * cw;
      for (let w = 0; w < cw; w++) dist += popcnt((codes[off + w] ^ qc[w]) >>> 0);

      if (m < POOL) {
        this.candIdx[m] = i; this.candDist[m] = dist; m++;
        if (m === POOL) worstAt = argmax(this.candDist, POOL);
      } else if (dist < this.candDist[worstAt]) {
        // Replace the current worst, then rescan for the new one. O(POOL) only
        // when a candidate actually beats the cut, which becomes rare fast.
        this.candIdx[worstAt] = i; this.candDist[worstAt] = dist;
        worstAt = argmax(this.candDist, POOL);
      }
    }
    if (m === 0) return this.out;

    for (const set of this.seen.values()) set.clear();
    for (const b of this.buckets.values()) b.n = 0;

    // Selection sort over the pool, stopping once every bucket is full or
    // starved. Sorting all POOL candidates is wasted work: the tail never
    // reaches a bucket.
    const want = K * 6;
    const take = Math.min(m, want);
    let filled = 0;
    for (let i = 0; i < take && filled < want; i++) {
      let best = i;
      for (let j = i + 1; j < m; j++) if (this.candDist[j] < this.candDist[best]) best = j;
      let t = this.candIdx[i]; this.candIdx[i] = this.candIdx[best]; this.candIdx[best] = t;
      t = this.candDist[i]; this.candDist[i] = this.candDist[best]; this.candDist[best] = t;

      const ref = this.refs[this.candIdx[i]];
      const bucket = this.buckets.get(ref.strategy)!;
      if (bucket.n >= K) continue;
      const seen = this.seen.get(ref.strategy)!;
      if (seen.has(ref.passage)) continue;   // best chunk of this passage wins
      seen.add(ref.passage);
      bucket.parentIds[bucket.n] = ref.passage;
      bucket.scores[bucket.n] = 1 - this.candDist[i] / this.dim;
      bucket.n++;
      filled++;
    }

    for (const b of this.buckets.values()) if (b.n > 0) this.out.push(b);
    return this.out;
  }

  /** int8 rescore of one user passage — the corpus rescorer's twin. */
  prepare(q: Float32Array): void {
    for (let i = 0; i < this.dim; i++) {
      const v = Math.round(q[i] * 127);
      this.qInt8[i] = v > 127 ? 127 : v < -128 ? -128 : v;
    }
  }

  score(passageId: number): number {
    if (passageId < 0 || passageId >= this.pCount) return 0;
    const off = passageId * this.dim;
    const a = this.p8;
    const qi = this.qInt8;
    let dot = 0;
    for (let j = 0; j < this.dim; j++) dot += a[off + j] * qi[j];
    return (dot / 16129) * this.pScale[passageId];
  }
}

function argmax(a: Int32Array, n: number): number {
  let at = 0;
  for (let i = 1; i < n; i++) if (a[i] > a[at]) at = i;
  return at;
}

function grow32(a: Uint32Array, size: number): Uint32Array {
  const b = new Uint32Array(size); b.set(a); return b;
}
function grow8(a: Int8Array, size: number): Int8Array {
  const b = new Int8Array(size); b.set(a); return b;
}
function growF(a: Float32Array, size: number): Float32Array {
  const b = new Float32Array(size); b.set(a); return b;
}
