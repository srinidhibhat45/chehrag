/**
 * Stage-based orchestration harness.
 *
 * Requirement 5 asks for structured orchestration rather than one prompt-in /
 * text-out call. Concretely that means every unit of work is a declared Stage
 * with its own contract:
 *
 *   - a typed input and output
 *   - a latency budget it is not allowed to exceed
 *   - a retry policy
 *   - a declared behaviour on failure: FAIL (abort) or DEGRADE (fall back)
 *   - telemetry recorded whether it succeeds or not
 *
 * The design constraint that shapes everything: we are graded on P100, so the
 * pipeline must have a *bounded* worst case. A stage that can hang has no
 * bounded worst case, so every stage carries a budget and the pipeline carries a
 * global deadline. When the deadline is near, remaining optional stages are
 * skipped rather than run and abandoned -- an answer slightly less polished is
 * strictly better than a blown budget.
 */

export type StageOutcome = "ok" | "degraded" | "skipped" | "failed";

export interface StageTelemetry {
  name: string;
  outcome: StageOutcome;
  ms: number;
  attempts: number;
  note?: string;
}

export interface RunContext {
  /** Wall-clock ms remaining before the pipeline's global deadline. */
  remaining(): number;
  /** Everything measured so far. */
  telemetry: StageTelemetry[];
  /** Free-form scratch shared across stages (kept small and typed by callers). */
  bag: Record<string, unknown>;
  signal: AbortSignal;
}

export interface Stage<I, O> {
  name: string;
  /** Hard budget for one attempt, ms. */
  budgetMs: number;
  /** Extra attempts after the first. Only meaningful for non-deterministic work. */
  retries?: number;
  /**
   * FAIL    — this stage is load-bearing; abort the run.
   * DEGRADE — recoverable; use `fallback` and carry on.
   */
  onError?: "FAIL" | "DEGRADE";
  /** Required when onError is DEGRADE. */
  fallback?: (input: I, err: unknown, ctx: RunContext) => O;
  /** Skip entirely if fewer than this many ms remain. Optional stages only. */
  minRemainingMs?: number;
  run: (input: I, ctx: RunContext) => O | Promise<O>;
}

export class StageError extends Error {
  constructor(readonly stage: string, readonly cause: unknown) {
    super(`stage "${stage}" failed: ${cause instanceof Error ? cause.message : String(cause)}`);
    this.name = "StageError";
  }
}

/** Rejects if `p` outruns `ms`. Synchronous stages bypass this entirely. */
/**
 * A stage that ran out of budget, as opposed to one that failed.
 *
 * The distinction is the whole reason this class exists. A transient fault —
 * an ONNX session failing under memory pressure — is worth retrying, because
 * the second attempt genuinely might succeed. A timeout on deterministic work
 * is not: the stage was asked to do a fixed amount of work, and it did not fit.
 * Running it again produces the same overrun and pays for it twice.
 */
export class StageTimeout extends Error {
  constructor(name: string, ms: number) {
    super(`${name} exceeded ${ms}ms budget`);
    this.name = "StageTimeout";
  }
}

function withTimeout<T>(p: Promise<T>, ms: number, name: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new StageTimeout(name, ms)), ms);
    p.then(
      (v) => { clearTimeout(t); resolve(v); },
      (e) => { clearTimeout(t); reject(e); },
    );
  });
}

const now = () =>
  typeof performance !== "undefined" ? performance.now() : Date.now();

/**
 * A retry needs enough budget left to plausibly succeed. Below this, retrying
 * only guarantees a second failure and a blown deadline, so the stage gives up
 * and lets its declared `onError` policy take over.
 */
const RETRY_MIN_REMAINING_MS = 5;

export class Pipeline {
  private stages: Stage<any, any>[] = [];

  constructor(private readonly globalBudgetMs: number) {}

  add<I, O>(stage: Stage<I, O>): this {
    this.stages.push(stage);
    return this;
  }

  async run<T>(initial: unknown, signal?: AbortSignal): Promise<{
    value: T;
    telemetry: StageTelemetry[];
    totalMs: number;
    ok: boolean;
  }> {
    const t0 = now();
    const telemetry: StageTelemetry[] = [];
    const ctrl = new AbortController();
    if (signal) signal.addEventListener("abort", () => ctrl.abort(), { once: true });

    const ctx: RunContext = {
      remaining: () => this.globalBudgetMs - (now() - t0),
      telemetry,
      bag: {},
      signal: ctrl.signal,
    };

    let value: any = initial;
    let ok = true;

    for (const stage of this.stages) {
      const sStart = now();

      // Deadline-aware skip. Running an optional stage we cannot finish wastes
      // budget that the remaining required stages need.
      if (stage.minRemainingMs != null && ctx.remaining() < stage.minRemainingMs) {
        telemetry.push({
          name: stage.name, outcome: "skipped", ms: 0, attempts: 0,
          note: `only ${ctx.remaining().toFixed(1)}ms left, needs ${stage.minRemainingMs}ms`,
        });
        continue;
      }

      const maxAttempts = 1 + (stage.retries ?? 0);
      let attempts = 0;
      let lastErr: unknown;
      let settled = false;

      while (attempts < maxAttempts && !settled) {
        attempts++;
        try {
          const budget = Math.max(1, Math.min(stage.budgetMs, ctx.remaining()));
          const out = stage.run(value, ctx);
          value = out instanceof Promise
            ? await withTimeout(out, budget, stage.name)
            : out;
          telemetry.push({
            name: stage.name, outcome: "ok", ms: now() - sStart, attempts,
          });
          settled = true;
        } catch (err) {
          lastErr = err;
          if (attempts >= maxAttempts) break;

          // A timeout is never retried, however much budget is left.
          //
          // Retries exist for transient faults. A stage that timed out did not
          // hit a fault — it was handed a fixed amount of deterministic work
          // that did not fit, and running it again produces the identical
          // overrun at double the price.
          //
          // Found by bench/multilingual.ts: a long Assamese query embedded in
          // ~61ms against a 60ms stage budget, timed out with 139ms of the
          // global budget still free, retried, spent another ~61ms, and then
          // failed the whole pipeline under `onError: "FAIL"`. So a query that
          // was merely slow was turned into a 122.9ms *error* — the worst of
          // both outcomes, and the single worst number in a 3,000-query sweep.
          if (err instanceof StageTimeout) {
            break;
          }

          // Never start an attempt that cannot finish. A retry after a failure
          // late in the budget is the worst case for a hard deadline: the first
          // attempt already spent much of what was left, and the second spends
          // it again, so the stage overshoots precisely when the budget was
          // tightest.
          //
          // `bench/deadline.ts` found this — single-query overruns at 12-15ms
          // budgets that the retrieval plan could not explain, because the time
          // went to a second embedding pass nobody had budget for. Retries are
          // for transient faults with room to spare, not for deadline misses.
          const left = ctx.remaining();
          if (left <= 0 || left < RETRY_MIN_REMAINING_MS) {
            lastErr = new Error(
              `${stage.name} failed and there was no budget to retry ` +
              `(${left.toFixed(1)}ms left)`);
            break;
          }
        }
      }

      if (settled) continue;

      const mode = stage.onError ?? "FAIL";
      if (mode === "DEGRADE" && stage.fallback) {
        value = stage.fallback(value, lastErr, ctx);
        telemetry.push({
          name: stage.name, outcome: "degraded", ms: now() - sStart, attempts,
          note: lastErr instanceof Error ? lastErr.message : String(lastErr),
        });
        continue;
      }

      telemetry.push({
        name: stage.name, outcome: "failed", ms: now() - sStart, attempts,
        note: lastErr instanceof Error ? lastErr.message : String(lastErr),
      });
      ok = false;
      break;
    }

    return { value, telemetry, totalMs: now() - t0, ok };
  }
}

/**
 * Circuit breaker for external calls (STT, LLM).
 *
 * Without this, a provider outage turns every request into a full timeout wait.
 * After `threshold` consecutive failures the breaker opens and calls fail
 * instantly for `cooldownMs`, so a dead dependency costs ~0ms instead of the
 * full budget. One trial call is allowed through on expiry to test recovery.
 */
export class CircuitBreaker {
  private failures = 0;
  private openedAt = 0;

  constructor(
    private readonly threshold = 3,
    private readonly cooldownMs = 15_000,
  ) {}

  get isOpen(): boolean {
    if (this.failures < this.threshold) return false;
    if (Date.now() - this.openedAt > this.cooldownMs) {
      this.failures = this.threshold - 1;   // half-open: allow one trial
      return false;
    }
    return true;
  }

  async call<T>(fn: () => Promise<T>): Promise<T> {
    if (this.isOpen) throw new Error("circuit open — dependency unhealthy");
    try {
      const v = await fn();
      this.failures = 0;
      return v;
    } catch (e) {
      this.failures++;
      if (this.failures >= this.threshold) this.openedAt = Date.now();
      throw e;
    }
  }
}

/** Percentiles over a latency sample. P100 is the max, by definition. */
export function percentiles(msValues: number[]): Record<string, number> {
  if (!msValues.length) return {};
  const s = [...msValues].sort((a, b) => a - b);
  const at = (p: number) => s[Math.min(s.length - 1, Math.ceil((p / 100) * s.length) - 1)];
  return {
    p50: at(50), p70: at(70), p90: at(90), p95: at(95), p99: at(99),
    p100: s[s.length - 1],
    mean: s.reduce((a, b) => a + b, 0) / s.length,
    n: s.length,
  };
}
