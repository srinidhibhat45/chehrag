/**
 * Deadline-aware sizing of retrieval work.
 *
 * A per-stage timeout reports a missed deadline, it does not prevent one. The
 * only way to guarantee the cap is for the expensive stage to read the clock
 * first and pick an amount of work it can finish.
 *
 * Embedding is the variable cost: a couple of ms here, 40-70ms for the same
 * WASM graph on a mid-range Android, past 100ms on a throttled laptop.
 * Retrieval and the gates are near-constant, so retrieval is told what is left
 * after embedding.
 *
 * Knobs, in the order they are worth giving up:
 *
 *   nprobe          clusters per index. Linear in cost, sublinear in recall.
 *   perStrategyK    candidates kept per index. Narrows what fusion sees.
 *   maxCandidates   scan ceiling per index. Falls with nprobe.
 *   rescoreTopN     int8 re-ranking depth. Cut last, and never to zero while
 *                   any budget remains: gate 2 reads the rescored score.
 *
 * `bench/deadline.ts` is what keeps this honest.
 */

import type { RagConfig } from "./rag";

export interface RetrievalPlan {
  nprobe: number;
  perStrategyK: number;
  rescoreTopN: number;
  /** Candidates each index may scan before stopping. */
  maxCandidates: number;
  /** True when the plan is smaller than the configured one. Surfaced in the UI,
   *  so a degraded answer is not presented as a full-quality one. */
  degraded: boolean;
  /** Why it degraded, for telemetry. */
  reason?: string;
}

/**
 * Cost constants, in ms.
 *
 * Measured by `bench/_cost` on this build (0.30 ms fixed, 0.013 ms per nprobe,
 * 0.05 ms for BM25) and then multiplied by three, because the browser runs the
 * same code on WASM against a colder cache than the Node harness they were
 * taken on. They only need to be right within a factor of two: they pick a
 * rung, and every rung is still enforced by the stage budget behind it.
 *
 * Ranking every centroid is the fixed part and does not shrink with nprobe,
 * which is why it is charged separately. A model that folded it into the
 * per-nprobe term would think the floor rung was free and overrun by the
 * difference.
 */
const MS_DENSE_FIXED = 0.9;
const MS_PER_NPROBE = 0.04;
/** Flat scan over user chunks, per thousand chunks. */
const MS_PER_1K_USER_CHUNKS = 0.014;
/** Fusion, both gates, extraction, and the pipeline's own bookkeeping. */
const FIXED_TAIL_MS = 0.6;
/**
 * The BM25 pass over the corpus.
 *
 * Near-constant rather than proportional to nprobe: it is bounded by the
 * postings cap in `retrieval/lexical.ts`, not by how many clusters the dense
 * indices visit, so it belongs in the fixed cost and not in the ladder.
 * Charged unconditionally, because the ablation says it is the last thing that
 * should be given up rather than the first.
 */
const MS_LEXICAL = 0.15;
/** int8 rescore of one passage. */
const MS_PER_RESCORE = 0.002;

/**
 * Reserve held back from whatever the clock says is left.
 *
 * Absorbs what `remaining()` cannot see: a GC pause landing between the
 * measurement and the work, and `performance.now()` being coarsened to ~100µs
 * in cross-origin-isolated contexts.
 */
const SAFETY_MS = 8;

/**
 * Choose how much retrieval to do in the time that remains.
 *
 * @param remainingMs  budget left, from the pipeline's own clock
 * @param cfg          the configured (full-quality) settings
 * @param userChunks   chunks in enabled user sources, which the flat scan pays for
 */
export function budgetPlan(remainingMs: number, cfg: RagConfig, userChunks: number): RetrievalPlan {
  const full: RetrievalPlan = {
    nprobe: cfg.nprobe,
    perStrategyK: cfg.perStrategyK,
    rescoreTopN: cfg.rescoreTopN,
    maxCandidates: cfg.maxCandidates,
    degraded: false,
  };

  const userCost = (userChunks / 1000) * MS_PER_1K_USER_CHUNKS;
  const budget = remainingMs - SAFETY_MS - FIXED_TAIL_MS - MS_LEXICAL - userCost;

  const costOf = (p: RetrievalPlan) =>
    MS_DENSE_FIXED + p.nprobe * MS_PER_NPROBE + p.rescoreTopN * MS_PER_RESCORE;

  // The common case by a wide margin: embedding finished quickly and there is
  // an order of magnitude more budget than retrieval needs.
  if (budget >= costOf(full)) return full;

  // Cheapest-to-lose first. Each rung is a real configuration rather than an
  // interpolation, so behaviour is reproducible.
  //
  // `maxCandidates` falls with nprobe rather than being a rung of its own. At
  // 12 clusters an index rarely reaches even 3072 candidates, so cutting the
  // cap below that would cost nothing to buy nothing; nprobe is what actually
  // bounds the scan once the ladder is in use.
  const ladder: RetrievalPlan[] = [
    { nprobe: 12, perStrategyK: cfg.perStrategyK, rescoreTopN: cfg.rescoreTopN, maxCandidates: 3072, degraded: true },
    { nprobe: 8, perStrategyK: cfg.perStrategyK, rescoreTopN: cfg.rescoreTopN, maxCandidates: 3072, degraded: true },
    { nprobe: 6, perStrategyK: 18, rescoreTopN: 16, maxCandidates: 3072, degraded: true },
    { nprobe: 4, perStrategyK: 14, rescoreTopN: 12, maxCandidates: 2048, degraded: true },
    { nprobe: 3, perStrategyK: 10, rescoreTopN: 10, maxCandidates: 2048, degraded: true },
    { nprobe: 2, perStrategyK: 8, rescoreTopN: 8, maxCandidates: 1536, degraded: true },
    { nprobe: 1, perStrategyK: 6, rescoreTopN: 6, maxCandidates: 1024, degraded: true },
  ];

  for (const rung of ladder) {
    if (budget >= costOf(rung)) {
      return { ...rung, reason: `${remainingMs.toFixed(1)}ms left after embedding` };
    }
  }

  // Below the floor nothing fits. Return the floor anyway: one cluster of one
  // index still finds something, where the alternative is a refusal caused by
  // the clock rather than by the corpus. The global deadline is the backstop.
  return {
    nprobe: 1, perStrategyK: 6, rescoreTopN: 4, maxCandidates: 1024, degraded: true,
    reason: `only ${Math.max(0, remainingMs).toFixed(1)}ms left - minimum plan`,
  };
}

/** Human-readable summary for the latency panel. */
export function describePlan(p: RetrievalPlan): string {
  return p.degraded
    ? `reduced - nprobe ${p.nprobe}, k ${p.perStrategyK}, rescore ${p.rescoreTopN}`
    : `full - nprobe ${p.nprobe}, k ${p.perStrategyK}, rescore ${p.rescoreTopN}`;
}
