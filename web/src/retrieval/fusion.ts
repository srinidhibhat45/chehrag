/**
 * Reciprocal Rank Fusion across the chunking strategies.
 *
 * RRF rather than a score blend because the indices are not score comparable: a
 * `sentence` chunk averages ~111 chars against a `document` chunk's ~925, and
 * cosine against a short query runs systematically higher on short chunks.
 * Averaging would hand the sentence index a permanent advantage.
 *
 *     score(passage) = SUM over strategies of  weight / (k + rank)
 *
 * k=60 is the value from Cormack et al. Fusion is at passage level: several
 * chunks of one passage should reinforce it rather than crowd the list with
 * near-duplicates, so the best rank per (passage, strategy) wins.
 *
 * `Fuser` holds its working set. Fusing was the largest source of per-query
 * garbage in the pipeline (a Map, a Set per strategy and an object per
 * candidate), and minor GC is what sets P100.
 */

import { contentTokens } from "./tokens";

export interface StrategyHits {
  strategy: string;
  /** Passage ordinals, best first. */
  parentIds: Int32Array;
  scores: Float32Array;
  n: number;
}

export interface FusedHit {
  passageId: number;
  fusedScore: number;
  /** How many distinct strategies surfaced this passage - feeds guardrail gate 2. */
  agreement: number;
  /** Best raw similarity seen for this passage across strategies. */
  bestRawScore: number;
  bestStrategy: string;
  /** Bit i set when the i'th strategy of the current run surfaced this passage. */
  strategyMask: number;
}

/**
 * Per-strategy weights, encoding what each strategy is for - see
 * `pipeline/src/chunking/strategies.py`:
 *
 *   whole/contextual  precise, self-contained; trusted most
 *   sentence          precise but context-poor
 *   semantic          low yield on this corpus (passages are ~3 sentences)
 *   sliding           noisy alone, valuable as a tie-breaker
 *   document          recall instrument, dilute vectors; never precision
 *
 * Hand-set and modest in spread. Fitting six weights on the same split the
 * metrics are reported on would overfit; RRF is robust to weight choice, and
 * the eval measures leave-one-out contribution instead.
 */
export const DEFAULT_WEIGHTS: Record<string, number> = {
  whole: 1.0,
  contextual: 1.0,
  sentence: 0.9,
  semantic: 0.7,
  sliding: 0.6,
  document: 0.5,
  lexical: 0.8,
};

const RRF_K = 60;

/** Ceiling on distinct passages held per query. 13 strategies x topK 48 = 624. */
const MAX_HITS = 1024;
/** Open-addressed id -> slot table. Power of two, comfortably over MAX_HITS. */
const TABLE_BITS = 12;
const TABLE_SIZE = 1 << TABLE_BITS;
const TABLE_MASK = TABLE_SIZE - 1;

export class Fuser {
  private readonly pool: FusedHit[] = [];
  private readonly out: FusedHit[] = [];
  private readonly tblKey = new Int32Array(TABLE_SIZE);
  private readonly tblSlot = new Int32Array(TABLE_SIZE);
  private readonly tblStamp = new Int32Array(TABLE_SIZE);
  /** Strategy name per bit position, for the current run. */
  private names: string[] = [];
  private epoch = 0;
  private n = 0;

  constructor() {
    for (let i = 0; i < MAX_HITS; i++) {
      this.pool.push({
        passageId: -1, fusedScore: 0, agreement: 0,
        bestRawScore: 0, bestStrategy: "", strategyMask: 0,
      });
    }
  }

  /** Set by `slotFor` when the slot it returned was created by that call. */
  private fresh = false;

  /** Slot for `id`, creating one if this is the first sighting. -1 when full. */
  private slotFor(id: number): number {
    let i = (Math.imul(id, 0x9e3779b1) >>> (32 - TABLE_BITS)) & TABLE_MASK;
    for (;;) {
      if (this.tblStamp[i] !== this.epoch) {
        if (this.n >= MAX_HITS) return -1;
        const slot = this.n++;
        this.tblStamp[i] = this.epoch;
        this.tblKey[i] = id;
        this.tblSlot[i] = slot;
        this.fresh = true;
        return slot;
      }
      if (this.tblKey[i] === id) { this.fresh = false; return this.tblSlot[i]; }
      i = (i + 1) & TABLE_MASK;
    }
  }

  /**
   * Fuse per-strategy hit lists into one ranked list.
   *
   * The returned array and the hits inside it are reused by the next call.
   * Callers hold them for the duration of one query and no longer.
   */
  fuse(hits: StrategyHits[], topN = 10, weights: Record<string, number> = DEFAULT_WEIGHTS): FusedHit[] {
    this.epoch++;
    this.n = 0;
    this.out.length = 0;
    this.names.length = 0;

    // 30 strategies is well past the 13 the app can produce, and keeps every
    // mask shift inside int32.
    for (let s = 0; s < hits.length && s < 30; s++) {
      const h = hits[s];
      this.names.push(h.strategy);
      const bit = 1 << s;
      const w = weights[h.strategy] ?? 1.0;

      for (let i = 0; i < h.n; i++) {
        const pid = h.parentIds[i];
        const slot = this.slotFor(pid);
        if (slot < 0) break;
        const e = this.pool[slot];
        if (this.fresh) {
          e.passageId = pid;
          e.fusedScore = 0;
          e.agreement = 0;
          e.bestRawScore = -Infinity;
          e.bestStrategy = h.strategy;
          e.strategyMask = 0;
          this.out.push(e);
        }
        // A passage contributes once per strategy however many chunks matched,
        // and the first sighting is its best rank because hits arrive ranked.
        if (e.strategyMask & bit) continue;
        e.strategyMask |= bit;
        e.fusedScore += w / (RRF_K + i + 1);
        e.agreement += 1;
        if (h.scores[i] > e.bestRawScore) {
          e.bestRawScore = h.scores[i];
          e.bestStrategy = h.strategy;
        }
      }
    }

    this.out.sort(byFusedScore);
    if (this.out.length > topN) this.out.length = topN;
    return this.out;
  }

  /** Names of the strategies that surfaced `h`, for display on a citation. */
  strategiesOf(h: FusedHit): string[] {
    const out: string[] = [];
    for (let i = 0; i < this.names.length; i++) {
      if (h.strategyMask & (1 << i)) out.push(this.names[i]);
    }
    return out;
  }
}

function byFusedScore(a: FusedHit, b: FusedHit): number {
  return b.fusedScore - a.fusedScore;
}

/**
 * Margin between the top hit and the runner-up, normalised by the top score.
 *
 * A flat distribution means retrieval found many equally mediocre things, the
 * signature of a query the corpus cannot answer. Gate 2 reads this alongside
 * the absolute score.
 */
export function fusionMargin(fused: FusedHit[]): number {
  if (fused.length < 2) return 1;
  const top = fused[0].fusedScore;
  if (top <= 0) return 0;
  return (top - fused[1].fusedScore) / top;
}

/**
 * Query-vs-passage token overlap. Cheap lexical sanity check for gate 2.
 *
 * Content words only. Including function words would make this a measure of how
 * tersely the question was phrased rather than of what it was about.
 */
export function lexicalOverlap(query: string, passage: string): number {
  const q = contentTokens(query);
  if (!q.size) return 0;
  const p = contentTokens(passage);
  let hit = 0;
  for (const w of q) if (p.has(w)) hit++;
  return hit / q.size;
}
