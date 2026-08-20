/**
 * Reciprocal Rank Fusion across the six chunking strategies.
 *
 * RRF rather than a score blend because the six indices are not score
 * comparable: a `sentence` chunk averages ~111 chars against a `document`
 * chunk's ~925, and cosine against a short query runs systematically higher on
 * short chunks. Averaging would hand the sentence index a permanent advantage.
 *
 * Rank is comparable across indices by construction, so RRF needs no per-index
 * normalisation and no tuning beyond k:
 *
 *   score(passage) = SUM over strategies of  weight / (k + rank)
 *
 * k=60 is the value from Cormack et al.; it damps the head so no single index
 * dominates.
 *
 * Fusion is at passage level, not chunk level: several chunks of one passage
 * should reinforce it rather than crowd the result list with near-duplicates.
 * Best rank per (passage, strategy) wins.
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
  /** How many distinct strategies surfaced this passage — feeds guardrail gate 2. */
  agreement: number;
  /** Best raw similarity seen for this passage across strategies. */
  bestRawScore: number;
  bestStrategy: string;
  perStrategyRank: Record<string, number>;
}

/**
 * Per-strategy weights, encoding what each strategy is for — see
 * `pipeline/src/chunking/strategies.py`:
 *
 *   whole/contextual  precise, self-contained; trusted most
 *   sentence          precise but context-poor
 *   semantic          low yield on this corpus (passages are ~3 sentences)
 *   sliding           noisy alone, valuable as a tie-breaker
 *   document          recall instrument, dilute vectors; never precision
 *
 * Hand-set and modest in spread. Fitting six weights on the same split the
 * metrics are reported on would overfit; RRF is robust to weight choice, and the
 * eval measures leave-one-out contribution instead.
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

export function fuse(
  hits: StrategyHits[],
  topN = 10,
  weights: Record<string, number> = DEFAULT_WEIGHTS,
): FusedHit[] {
  const acc = new Map<number, FusedHit>();

  for (const h of hits) {
    const w = weights[h.strategy] ?? 1.0;
    // Best rank per passage within this strategy — a passage contributes once
    // per strategy no matter how many of its chunks matched.
    const seen = new Set<number>();
    for (let i = 0; i < h.n; i++) {
      const pid = h.parentIds[i];
      if (seen.has(pid)) continue;
      seen.add(pid);

      const contrib = w / (RRF_K + i + 1);
      let e = acc.get(pid);
      if (!e) {
        e = {
          passageId: pid, fusedScore: 0, agreement: 0,
          bestRawScore: -Infinity, bestStrategy: h.strategy, perStrategyRank: {},
        };
        acc.set(pid, e);
      }
      e.fusedScore += contrib;
      e.agreement += 1;
      e.perStrategyRank[h.strategy] = i + 1;
      if (h.scores[i] > e.bestRawScore) {
        e.bestRawScore = h.scores[i];
        e.bestStrategy = h.strategy;
      }
    }
  }

  const out = [...acc.values()];
  out.sort((a, b) => b.fusedScore - a.fusedScore);
  return out.slice(0, topN);
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
 * tersely the question was phrased rather than of what it was about. See
 * `tokens.ts`.
 */
export function lexicalOverlap(query: string, passage: string): number {
  const q = contentTokens(query);
  if (!q.size) return 0;
  const p = contentTokens(passage);
  let hit = 0;
  for (const w of q) if (p.has(w)) hit++;
  return hit / q.size;
}
