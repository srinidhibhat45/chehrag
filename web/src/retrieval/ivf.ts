/**
 * IVF over binary-quantised chunk vectors.
 *
 * A full Hamming scan of ~810k chunks costs tens of ms in JS, which is fine on
 * average and fatal at P100 once a GC pause lands on top. HNSW ships a large
 * graph and has a data-dependent tail. IVF bounds the work instead: probe a
 * fixed number of clusters, scan a capped number of candidates.
 *
 * Precision is two-stage. Binary codes and Hamming here; int8 dot product on
 * the fused passage candidates in PassageRescorer below. Stage 2 is per-passage
 * rather than per-chunk because per-chunk int8 would be 155 MB against 19 MB,
 * and fusion collapses to passages anyway.
 *
 * Every buffer is allocated in the constructor. The search path allocates
 * nothing, which is what keeps the tail flat.
 */

export interface IvfIndexData {
  dim: number;
  codeWords: number;        // Uint32 words per binary code (dim/32)
  nlist: number;
  count: number;
  centroids: Float32Array;  // nlist * dim
  listOffsets: Uint32Array; // nlist + 1 prefix offsets into the slot arrays
  slotParent: Int32Array;   // count - passage ordinal per slot, list-ordered
  codes: Uint32Array;       // count * codeWords, list-ordered
}

/**
 * popcount, branch-free.
 *
 * `Math.imul` and `>>>` are correctness rather than style: plain `*` yields a
 * double past 2^31 and the following shift then does a lossy ToInt32, returning
 * a wrong and often negative count. Retrieval keeps running and ranks garbage.
 */
function popcnt(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}

export interface StrategySearchResult {
  /** Passage ordinals, best first, de-duplicated. */
  parentIds: Int32Array;
  /** Similarity proxy in [0,1]: 1 - hamming/dim. Rank is what matters downstream. */
  scores: Float32Array;
  n: number;
}

/**
 * Dimensions used for the first pass over the centroids.
 *
 * PCA orders components by variance, so a prefix of the vector carries most of
 * what separates one cluster from another. The prefix pass shortlists, then the
 * shortlist is scored on the full vector, so the cost of the wide pass drops
 * without the final choice being made on partial information.
 *
 * 96 and 3x were picked by measurement: against full-dim ranking they select
 * the same cluster set 99.7% of the time and the same nearest cluster 100% of
 * the time, for 57% of the cost. Widening to 128 dims buys 0.2 points of set
 * agreement for a third more time.
 */
const PREFIX_DIM = 96;
const SHORTLIST_MULT = 3;

export class IvfIndex {
  private d: IvfIndexData;

  private centroidScores: Float32Array;
  private probeOrder: Int32Array;
  private shortlist: Int32Array;
  private qCode: Uint32Array;
  private candSlot: Int32Array;
  private candDist: Int32Array;
  private sortSlot: Int32Array;
  private sortDist: Int32Array;
  private distCount: Int32Array;
  private distStart: Int32Array;
  private outParents: Int32Array;
  private outScores: Float32Array;
  private seenParent: Int32Array;   // parent -> epoch stamp, for O(1) dedupe
  private epoch = 0;
  private result: StrategySearchResult;

  /** Prefix length actually used, clamped to dim and to a multiple of 4. */
  private readonly prefixDim: number;
  /** Set when the unrolled inner loops apply. */
  private readonly fastDot: boolean;
  private readonly fastHamming: boolean;

  readonly maxCandidates: number;
  readonly topK: number;

  constructor(
    data: IvfIndexData,
    numPassages: number,
    maxCandidates = 6144,
    topK = 48,
  ) {
    this.d = data;
    this.maxCandidates = maxCandidates;
    this.topK = topK;
    this.prefixDim = Math.min(PREFIX_DIM, data.dim) & ~3;
    this.fastDot = data.dim % 4 === 0 && this.prefixDim >= 4;
    this.fastHamming = data.codeWords === 8;

    this.centroidScores = new Float32Array(data.nlist);
    this.probeOrder = new Int32Array(data.nlist);
    this.shortlist = new Int32Array(data.nlist);
    this.qCode = new Uint32Array(data.codeWords);
    this.candSlot = new Int32Array(maxCandidates);
    this.candDist = new Int32Array(maxCandidates);
    this.sortSlot = new Int32Array(maxCandidates);
    this.sortDist = new Int32Array(maxCandidates);
    // Hamming distance is an integer in [0, codeWords*32].
    this.distCount = new Int32Array(data.codeWords * 32 + 2);
    this.distStart = new Int32Array(data.codeWords * 32 + 2);
    this.outParents = new Int32Array(topK);
    this.outScores = new Float32Array(topK);
    // Stamped epochs avoid clearing a 99k array on every query.
    this.seenParent = new Int32Array(numPassages).fill(-1);
    this.result = { parentIds: this.outParents, scores: this.outScores, n: 0 };
  }

  /** Sign-bit binarisation. Vectors are L2-normalised so the threshold is 0. */
  private encodeQuery(q: Float32Array): void {
    const { codeWords, dim } = this.d;
    const code = this.qCode;
    for (let w = 0; w < codeWords; w++) {
      let acc = 0;
      const base = w * 32;
      const lim = Math.min(32, dim - base);
      for (let b = 0; b < lim; b++) if (q[base + b] > 0) acc |= 1 << b;
      code[w] = acc >>> 0;
    }
  }

  /** Dot product of centroid `c` against `q` over the first `upto` dimensions. */
  private dot(q: Float32Array, off: number, upto: number): number {
    const cent = this.d.centroids;
    if (this.fastDot) {
      // Four accumulators so the adds do not form one dependency chain.
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let i = 0; i < upto; i += 4) {
        s0 += cent[off + i] * q[i];
        s1 += cent[off + i + 1] * q[i + 1];
        s2 += cent[off + i + 2] * q[i + 2];
        s3 += cent[off + i + 3] * q[i + 3];
      }
      return s0 + s1 + s2 + s3;
    }
    let s = 0;
    for (let i = 0; i < upto; i++) s += cent[off + i] * q[i];
    return s;
  }

  /**
   * Fill `probeOrder[0..np)` with the nearest clusters, best first.
   *
   * One pass with bounded insertion rather than a partial selection sort: the
   * sort was O(np * nlist) and this is O(nlist) with a rare shift.
   */
  private rankClusters(q: Float32Array, np: number): void {
    const { nlist, dim } = this.d;
    const cs = this.centroidScores;
    const wide = this.prefixDim < dim && nlist > np * SHORTLIST_MULT;
    const pass1 = wide ? this.prefixDim : dim;

    for (let c = 0; c < nlist; c++) cs[c] = this.dot(q, c * dim, pass1);

    if (!wide) {
      this.selectTop(cs, nlist, np, this.probeOrder, null);
      return;
    }

    const shortN = Math.min(nlist, np * SHORTLIST_MULT);
    this.selectTop(cs, nlist, shortN, this.shortlist, null);
    for (let j = 0; j < shortN; j++) {
      const c = this.shortlist[j];
      cs[c] = this.dot(q, c * dim, dim);
    }
    this.selectTop(cs, shortN, np, this.probeOrder, this.shortlist);
  }

  /**
   * Top-`k` of `scores`, descending, into `out`.
   *
   * `via` maps position to candidate id when selecting from a shortlist rather
   * than from the whole range.
   */
  private selectTop(
    scores: Float32Array, count: number, k: number, out: Int32Array, via: Int32Array | null,
  ): void {
    let n = 0, worst = Infinity;
    for (let i = 0; i < count; i++) {
      const c = via ? via[i] : i;
      const s = scores[c];
      if (n === k && s <= worst) continue;
      let pos = n < k ? n : k - 1;
      while (pos > 0 && scores[out[pos - 1]] < s) { out[pos] = out[pos - 1]; pos--; }
      out[pos] = c;
      if (n < k) n++;
      worst = scores[out[n - 1]];
    }
  }

  /**
   * @param q      L2-normalised, PCA-reduced query vector (length = dim)
   * @param nprobe clusters to visit; higher = better recall, linearly more work
   * @param k      passages to return
   * @param cap    candidates to scan before stopping, defaulting to the
   *               allocation size. Lowering it per query is how the deadline
   *               planner buys time back on a slow machine.
   */
  search(q: Float32Array, nprobe: number, k: number, cap = this.maxCandidates): StrategySearchResult {
    const d = this.d;
    const K = Math.min(k, this.topK);
    const dim = d.dim;
    this.epoch++;
    this.encodeQuery(q);

    const np = Math.min(nprobe, d.nlist);
    this.rankClusters(q, np);

    // ---- Hamming over the probed lists -------------------------------------
    const codes = d.codes;
    const cw = d.codeWords;
    const qc = this.qCode;
    const cslot = this.candSlot;
    const cdist = this.candDist;
    const limit = cap > this.maxCandidates ? this.maxCandidates : cap;
    let nc = 0;

    if (this.fastHamming) {
      const q0 = qc[0], q1 = qc[1], q2 = qc[2], q3 = qc[3];
      const q4 = qc[4], q5 = qc[5], q6 = qc[6], q7 = qc[7];
      fast:
      for (let p = 0; p < np; p++) {
        const list = this.probeOrder[p];
        const end = d.listOffsets[list + 1];
        for (let s = d.listOffsets[list]; s < end; s++) {
          if (nc >= limit) break fast;
          const off = s << 3;
          let x = 0, dist = 0;
          x = (codes[off] ^ q0) >>> 0;
          x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
          dist += Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
          x = (codes[off + 1] ^ q1) >>> 0;
          x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
          dist += Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
          x = (codes[off + 2] ^ q2) >>> 0;
          x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
          dist += Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
          x = (codes[off + 3] ^ q3) >>> 0;
          x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
          dist += Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
          x = (codes[off + 4] ^ q4) >>> 0;
          x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
          dist += Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
          x = (codes[off + 5] ^ q5) >>> 0;
          x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
          dist += Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
          x = (codes[off + 6] ^ q6) >>> 0;
          x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
          dist += Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
          x = (codes[off + 7] ^ q7) >>> 0;
          x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
          dist += Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
          cslot[nc] = s; cdist[nc] = dist; nc++;
        }
      }
    } else {
      generic:
      for (let p = 0; p < np; p++) {
        const list = this.probeOrder[p];
        const end = d.listOffsets[list + 1];
        for (let s = d.listOffsets[list]; s < end; s++) {
          if (nc >= limit) break generic;
          let dist = 0;
          const off = s * cw;
          for (let w = 0; w < cw; w++) dist += popcnt((codes[off + w] ^ qc[w]) >>> 0);
          cslot[nc] = s; cdist[nc] = dist; nc++;
        }
      }
    }
    if (nc === 0) { this.result.n = 0; return this.result; }

    // ---- order the head by distance ----------------------------------------
    // Counting sort. Distances are small integers, so this is O(nc) where a
    // partial selection sort was O(need * nc) and cost more than the scan that
    // produced the candidates.
    const need = Math.min(nc, K * 4);
    const counts = this.distCount;
    const starts = this.distStart;
    const maxDist = cw * 32;
    counts.fill(0);
    for (let i = 0; i < nc; i++) counts[cdist[i]]++;

    let acc = 0, cut = maxDist;
    for (let dd = 0; dd <= maxDist; dd++) {
      acc += counts[dd];
      if (acc >= need) { cut = dd; break; }
    }
    let off = 0;
    for (let dd = 0; dd <= cut; dd++) { starts[dd] = off; off += counts[dd]; }

    const sslot = this.sortSlot;
    const sdist = this.sortDist;
    for (let i = 0; i < nc; i++) {
      const dd = cdist[i];
      if (dd > cut) continue;
      const p = starts[dd]++;
      sslot[p] = cslot[i];
      sdist[p] = dd;
    }

    // ---- collapse to distinct passages -------------------------------------
    const seen = this.seenParent;
    const ep = this.epoch;
    const lim = Math.min(off, need);
    let n = 0;
    for (let i = 0; i < lim && n < K; i++) {
      const parent = d.slotParent[sslot[i]];
      if (seen[parent] === ep) continue;     // best chunk of this passage already taken
      seen[parent] = ep;
      this.outParents[n] = parent;
      this.outScores[n] = 1 - sdist[i] / dim;
      n++;
    }
    this.result.n = n;
    return this.result;
  }
}

/**
 * Rescoring of fused passage candidates using int8 passage vectors.
 *
 * Binary codes discard magnitude, which is fine for casting a wide net and not
 * fine for final ordering or for a confidence threshold. This restores cosine
 * over the few dozen passages that survived fusion, and that value is what
 * gate 2 thresholds on.
 */
export class PassageRescorer {
  private qInt8: Int8Array;

  constructor(
    private readonly int8: Int8Array,      // numPassages * dim
    private readonly scale: Float32Array,  // numPassages
    private readonly dim: number,
  ) {
    this.qInt8 = new Int8Array(dim);
  }

  prepare(q: Float32Array): void {
    for (let i = 0; i < this.dim; i++) {
      const v = Math.round(q[i] * 127);
      this.qInt8[i] = v > 127 ? 127 : v < -128 ? -128 : v;
    }
  }

  score(passageId: number): number {
    const off = passageId * this.dim;
    const a = this.int8;
    const q = this.qInt8;
    let dot = 0;
    for (let j = 0; j < this.dim; j++) dot += a[off + j] * q[j];
    return (dot / 16129) * this.scale[passageId];   // 127*127
  }
}
