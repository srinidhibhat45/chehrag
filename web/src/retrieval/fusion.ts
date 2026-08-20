/**
 * Reciprocal Rank Fusion across the six chunking strategies.
 *
 * Why RRF rather than blending scores: the six indices are not score-comparable.
 * A `sentence` chunk is ~111 chars and a `document` chunk is ~925, and cosine
 * similarity against a short query behaves differently at each length -- short
 * chunks score systematically higher. Averaging those scores would silently hand
 * the sentence index a permanent advantage.
 *
 * RRF throws the magnitudes away and keeps only *rank*, which is comparable
 * across indices by construction. It needs no per-index normalisation and no
 * tuning beyond k.
 *
 *   score(passage) = SUM over strategies of  weight / (k + rank)
 *
 * k=60 is the value from the original Cormack et al. formulation; it damps the
 * head so a single index cannot dominate on its own.
 *
 * Chunks are fused at *passage* level, not chunk level: several chunks from one
 * passage should reinforce that passage, not crowd the result list with near
 * duplicates of itself. Best rank per (passage, strategy) wins.
 */

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
 * Per-strategy weights.
 *
 * These encode what each strategy is *for*, from the design rationale in
 * `pipeline/src/chunking/strategies.py`:
 *   whole/contextual — precise, self-contained; trusted most
 *   sentence         — precise but context-poor
 *   semantic         — low yield on this corpus (passages are ~3 sentences)
 *   sliding          — noisy alone, valuable as a tie-breaker
 *   document         — recall instrument, dilute vectors; never precision
 *
 * Deliberately hand-set and modest in spread. Fitting six weights on the same
 * split we report metrics on would overfit; RRF is robust to weight choice, and
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
 * A flat distribution means retrieval found many equally-mediocre things, which
 * is the signature of a query the corpus cannot answer. Guardrail gate 2 reads
 * this alongside the absolute score.
 */
export function fusionMargin(fused: FusedHit[]): number {
  if (fused.length < 2) return 1;
  const top = fused[0].fusedScore;
  if (top <= 0) return 0;
  return (top - fused[1].fusedScore) / top;
}

/** Query-vs-passage token overlap. Cheap lexical sanity check for gate 2. */
export function lexicalOverlap(query: string, passage: string): number {
  const norm = (s: string) =>
    s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ").split(/\s+/).filter((w) => w.length > 1);
  const q = new Set(norm(query));
  if (!q.size) return 0;
  const p = new Set(norm(passage));
  let hit = 0;
  for (const w of q) if (p.has(w)) hit++;
  return hit / q.size;
}
