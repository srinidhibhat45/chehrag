/**
 * Deadline-aware sizing of retrieval work.
 *
 * The budget is a hard cap, and a hard cap needs more than per-stage timeouts.
 * A timeout tells you the deadline was missed; it does not stop you missing it.
 * The only way to *guarantee* the cap is for the expensive stage to look at the
 * clock before it starts and choose an amount of work it can finish.
 *
 * That matters here because the query path has one variable cost and several
 * fixed ones. Embedding is the variable: ~7ms on the machine this was built on,
 * but the same WASM graph on a mid-range Android phone is 40-70ms, and on a
 * cold thermally-throttled laptop it can spike past 100ms. Retrieval, fusion
 * and the gates are near-constant. So retrieval is told what is *left* after
 * embedding, and sizes itself to that.
 *
 * The knobs, in the order they are worth giving up:
 *
 *   nprobe        clusters visited per index. Linear in cost, sublinear in
 *                 recall — the first few clusters hold most of the answer, so
 *                 this is the cheapest thing to cut and it is cut first.
 *   perStrategyK  candidates kept per index. Cutting this narrows what fusion
 *                 has to work with, so it goes second.
 *   rescoreTopN   int8 re-ranking depth. Cut last and never to zero while any
 *                 budget remains: gate 2's calibrated threshold reads the
 *                 rescored score, so dropping rescoring entirely changes what
 *                 the guardrail is thresholding on.
 *
 * The costs below are measured, not guessed — `bench/deadline.ts` exercises them.
 * They only need to be right to within a factor of two: they exist to pick a
 * tier, and every tier is then still enforced by the stage budgets behind it.
 */

import type { RagConfig } from "./rag";

export interface RetrievalPlan {
  nprobe: number;
  perStrategyK: number;
  rescoreTopN: number;
  /** True when the plan is smaller than the configured one — surfaced in the UI
   *  so a degraded answer is never presented as a full-quality one. */
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
 * Absorbs the two things `remaining()` cannot see: a GC pause landing between
 * the measurement and the work, and the fact that `performance.now()` is
 * coarsened to ~100µs in cross-origin-isolated contexts. Without it the plan
 * would aim exactly at the deadline and land past it whenever either happens.
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

  // Degrade along the ladder, cheapest-to-lose first. Each rung is a real
  // configuration, not an interpolation, so behaviour is reproducible.
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

  // Below the floor there is no plan that fits. Return the floor anyway rather
  // than nothing: one cluster of one index still finds something, and the
  // alternative is a refusal caused by our own clock rather than by the corpus.
  // The pipeline's global deadline remains the backstop.
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
