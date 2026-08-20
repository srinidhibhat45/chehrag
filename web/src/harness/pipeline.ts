/**
 * Stage-based orchestration harness.
 *
 * Every unit of work is a declared Stage with its own contract: a typed input
 * and output, a latency budget, a retry policy, a declared failure behaviour
 * (FAIL aborts, DEGRADE falls back), and telemetry either way.
 *
 * The constraint that shapes all of it is a bounded worst case. A stage that can
 * hang has none, so every stage carries a budget and the pipeline carries a
 * global deadline. Near the deadline, remaining optional stages are skipped
 * rather than started and abandoned.
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

/**
 * A stage that ran out of budget, as distinct from one that failed.
 *
 * The distinction drives the retry policy. A transient fault — an ONNX session
 * failing under memory pressure — is worth retrying. A timeout on deterministic
 * work is not: the stage was given a fixed amount of work that did not fit, and
 * a second attempt overruns identically at double the price.
 */
export class StageTimeout extends Error {
  constructor(name: string, ms: number) {
    super(`${name} exceeded ${ms}ms budget`);
    this.name = "StageTimeout";
  }
}

/** Rejects if `p` outruns `ms`. Synchronous stages bypass this entirely. */
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

      // Deadline-aware skip. An optional stage that cannot finish would spend
      // budget the remaining required stages need.
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

          // A timeout is never retried, however much budget is left. Retries
          // are for transient faults; deterministic work that did not fit will
          // not fit the second time either, and the retry turns a merely slow
          // query into a doubly-slow error.
          if (err instanceof StageTimeout) {
            break;
          }

          // Never start an attempt that cannot finish. A retry late in the
          // budget is the worst case for a hard deadline: the first attempt
          // already spent most of what was left and the second spends it again,
          // overshooting exactly when the budget is tightest.
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
