/**
 * IVF + binary-quantised chunk index, built for a hard P100 budget.
 *
 * Not brute force: a full Hamming scan over ~810k chunk vectors costs tens of ms
 * in JS — affordable on average, lethal at P100 with a GC pause on top.
 *
 * Not HNSW: the graph is large to ship and its traversal has a long,
 * data-dependent tail, which is the wrong shape when the worst query is what
 * matters.
 *
 * IVF bounds the work per query: probe a fixed number of clusters, scan a capped
 * number of candidates. Latency goes near-constant and the tail flattens.
 *
 * Two-stage precision, split across this file and the caller:
 *   stage 1 (here)   binary codes + Hamming — cheap, wide net, per strategy
 *   stage 2 (caller) int8 dot product on fused passage candidates only
 *
 * Stage 2 is at passage level rather than chunk level: per-chunk int8 vectors
 * would be 810k x 192 = 155 MB to ship against 99k x 192 = 19 MB per passage,
 * and fusion collapses to passages anyway.
 *
 * P100 discipline enforced here:
 *   - every buffer allocated at construction; steady-state search allocates none
 *   - candidate count hard-capped, so work per query has a ceiling
 *   - no closures, iterators, or dynamic arrays in the hot loop
 */

export interface IvfIndexData {
  dim: number;              // reduced embedding dim (192 after PCA)
  codeWords: number;        // Uint32 words per binary code (dim/32)
  nlist: number;            // number of clusters
  count: number;            // number of chunk vectors
  centroids: Float32Array;  // nlist * dim
  listOffsets: Uint32Array; // nlist + 1 prefix offsets into the slot arrays
  slotParent: Int32Array;   // count — passage ordinal per slot (list-ordered)
  codes: Uint32Array;       // count * codeWords — binary codes (list-ordered)
}

/**
 * popcount for one 32-bit word, branch-free.
 *
 * `Math.imul` and `>>>` are correctness, not style: plain `*` produces a double
 * that exceeds 2^31 for large inputs, and the following shift then does a lossy
 * ToInt32 conversion and returns a wrong, often negative count. Retrieval keeps
 * running and ranks by garbage.
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

export class IvfIndex {
  private d: IvfIndexData;

  // preallocated scratch — never resized
  private centroidScores: Float32Array;
  private probeOrder: Int32Array;
  private qCode: Uint32Array;
  private candSlot: Int32Array;
  private candDist: Int32Array;
  private outParents: Int32Array;
  private outScores: Float32Array;
  private seenParent: Int32Array;   // parent -> epoch stamp, for O(1) dedupe
  private epoch = 0;
  private result: StrategySearchResult;

  readonly maxCandidates: number;
  readonly topK: number;

  constructor(
    data: IvfIndexData,
    numPassages: number,
    maxCandidates = 3072,
    topK = 48,
  ) {
    this.d = data;
    this.maxCandidates = maxCandidates;
    this.topK = topK;
    this.centroidScores = new Float32Array(data.nlist);
    this.probeOrder = new Int32Array(data.nlist);
    this.qCode = new Uint32Array(data.codeWords);
    this.candSlot = new Int32Array(maxCandidates);
    this.candDist = new Int32Array(maxCandidates);
    this.outParents = new Int32Array(topK);
    this.outScores = new Float32Array(topK);
    // Stamped-epoch dedupe avoids clearing a 99k array on every query.
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

  /**
   * @param q      L2-normalised, PCA-reduced query vector (length = dim)
   * @param nprobe clusters to visit; higher = better recall, linearly more work
   * @param k      passages to return
   */
  search(q: Float32Array, nprobe: number, k: number): StrategySearchResult {
    const d = this.d;
    const K = Math.min(k, this.topK);
    this.epoch++;
    this.encodeQuery(q);

    // ---- stage 0: rank centroids -------------------------------------------
    const cs = this.centroidScores;
    const cent = d.centroids;
    const dim = d.dim;
    for (let c = 0; c < d.nlist; c++) {
      let s = 0;
      const off = c * dim;
      for (let i = 0; i < dim; i++) s += cent[off + i] * q[i];
      cs[c] = s;
      this.probeOrder[c] = c;
    }
    // Partial selection: nprobe is small and its cost is fixed rather than
    // data-dependent, which matters more than raw speed at P100.
    const np = Math.min(nprobe, d.nlist);
    for (let i = 0; i < np; i++) {
      let best = i;
      for (let j = i + 1; j < d.nlist; j++) {
        if (cs[this.probeOrder[j]] > cs[this.probeOrder[best]]) best = j;
      }
      const t = this.probeOrder[i];
      this.probeOrder[i] = this.probeOrder[best];
      this.probeOrder[best] = t;
    }

    // ---- stage 1: Hamming scan over probed lists ---------------------------
    const codes = d.codes;
    const cw = d.codeWords;
    const qc = this.qCode;
    let nc = 0;
    outer:
    for (let p = 0; p < np; p++) {
      const list = this.probeOrder[p];
      const start = d.listOffsets[list];
      const end = d.listOffsets[list + 1];
      for (let s = start; s < end; s++) {
        if (nc >= this.maxCandidates) break outer;   // hard ceiling => bounded P100
        let dist = 0;
        const off = s * cw;
        for (let w = 0; w < cw; w++) dist += popcnt((codes[off + w] ^ qc[w]) >>> 0);
        this.candSlot[nc] = s;
        this.candDist[nc] = dist;
        nc++;
      }
    }
    if (nc === 0) { this.result.n = 0; return this.result; }

    // ---- stage 2: collapse to top-K distinct passages ----------------------
    // Partial selection sort to K*4 rather than a full sort: only enough of the
    // head is needed to find K distinct parents, and bounded beats optimal here.
    const need = Math.min(nc, K * 4);
    for (let i = 0; i < need; i++) {
      let best = i;
      for (let j = i + 1; j < nc; j++) if (this.candDist[j] < this.candDist[best]) best = j;
      let t = this.candSlot[i]; this.candSlot[i] = this.candSlot[best]; this.candSlot[best] = t;
      t = this.candDist[i]; this.candDist[i] = this.candDist[best]; this.candDist[best] = t;
    }

    const seen = this.seenParent;
    const ep = this.epoch;
    let n = 0;
    for (let i = 0; i < need && n < K; i++) {
      const parent = d.slotParent[this.candSlot[i]];
      if (seen[parent] === ep) continue;     // best chunk of this passage already taken
      seen[parent] = ep;
      this.outParents[n] = parent;
      this.outScores[n] = 1 - this.candDist[i] / dim;
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
 * similarity over the few dozen passages that survived fusion, and that value is
 * what gate 2 thresholds on.
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
