/**
 * Deadline-aware sizing of retrieval work.
 *
 * A per-stage timeout reports that the deadline was missed; it does not prevent
 * the miss. Guaranteeing the cap means the expensive stage reads the clock
 * before it starts and picks an amount of work it can finish.
 *
 * The query path has one variable cost and several fixed ones. Embedding is the
 * variable: ~7ms here, 40-70ms for the same WASM graph on a mid-range Android
 * phone, past 100ms on a thermally-throttled laptop. Retrieval, fusion and the
 * gates are near-constant, so retrieval is told what remains after embedding.
 *
 * The knobs, in the order they are worth giving up:
 *
 *   nprobe        clusters visited per index. Linear in cost, sublinear in
 *                 recall, so this goes first.
 *   perStrategyK  candidates kept per index. Narrows what fusion sees.
 *   rescoreTopN   int8 re-ranking depth. Cut last and never to zero while any
 *                 budget remains: gate 2's threshold reads the rescored score.
 *
 * The costs below are measured by `bench/deadline.ts`. They only need to be
 * right within a factor of two — they pick a tier, and each tier is still
 * enforced by the stage budgets behind it.
 */

import type { RagConfig } from "./rag";

export interface RetrievalPlan {
  nprobe: number;
  perStrategyK: number;
  rescoreTopN: number;
  /** True when the plan is smaller than the configured one. Surfaced in the UI,
   *  so a degraded answer is not presented as a full-quality one. */
  degraded: boolean;
  /** Why it degraded, for telemetry. */
  reason?: string;
}

/**
 * Measured on an M3 Pro, Chrome, warm index (see README "Measured results").
 * Cost is dominated by the Hamming scan, which is linear in nprobe across the
 * six corpus indices.
 */
const MS_PER_NPROBE = 0.17;
/** Flat scan over user chunks, per thousand chunks. */
const MS_PER_1K_USER_CHUNKS = 0.014;
/** Fusion, both gates, extraction, and the pipeline's own bookkeeping. */
const FIXED_TAIL_MS = 1.2;
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
    degraded: false,
  };

  const userCost = (userChunks / 1000) * MS_PER_1K_USER_CHUNKS;
  const budget = remainingMs - SAFETY_MS - FIXED_TAIL_MS - userCost;

  const costOf = (p: RetrievalPlan) =>
    p.nprobe * MS_PER_NPROBE + p.rescoreTopN * MS_PER_RESCORE;

  // The common case by a wide margin: embedding finished quickly and there is
  // an order of magnitude more budget than retrieval needs.
  if (budget >= costOf(full)) return full;

  // Cheapest-to-lose first. Each rung is a real configuration rather than an
  // interpolation, so behaviour is reproducible.
  const ladder: RetrievalPlan[] = [
    { nprobe: 8, perStrategyK: cfg.perStrategyK, rescoreTopN: cfg.rescoreTopN, degraded: true },
    { nprobe: 6, perStrategyK: 18, rescoreTopN: 16, degraded: true },
    { nprobe: 4, perStrategyK: 14, rescoreTopN: 12, degraded: true },
    { nprobe: 3, perStrategyK: 10, rescoreTopN: 10, degraded: true },
    { nprobe: 2, perStrategyK: 8, rescoreTopN: 8, degraded: true },
    { nprobe: 1, perStrategyK: 6, rescoreTopN: 6, degraded: true },
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
    nprobe: 1, perStrategyK: 6, rescoreTopN: 4, degraded: true,
    reason: `only ${Math.max(0, remainingMs).toFixed(1)}ms left — minimum plan`,
  };
}

/** Human-readable summary for the latency panel. */
export function describePlan(p: RetrievalPlan): string {
  return p.degraded
    ? `reduced — nprobe ${p.nprobe}, k ${p.perStrategyK}, rescore ${p.rescoreTopN}`
    : `full — nprobe ${p.nprobe}, k ${p.perStrategyK}, rescore ${p.rescoreTopN}`;
}
