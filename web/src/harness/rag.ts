/**
 * The RAG pipeline itself, expressed as harness stages.
 *
 * Everything here is inside the 200ms budget. The LLM synthesis path is
 * deliberately NOT here — it runs afterwards, off the clock, and its result
 * replaces the extractive answer in the UI only once it arrives and passes the
 * grounding gate.
 *
 * Stage budgets are set well above measured cost so that a slow machine
 * degrades gracefully rather than failing. The global budget is what actually
 * enforces the requirement, and `deadline.ts` is what makes it a *hard* cap
 * rather than an aspiration: when time runs short the retrieval stage reduces
 * its own work instead of overrunning.
 *
 * Two corpora, one ranking. The shipped MS MARCO index and the user's own
 * sources are searched separately — they need different structures, IVF versus
 * flat — but their hits are fused in a single RRF pass and thresholded by a
 * single calibrated confidence. That only holds because both sides quantise
 * identically; see sources/quantise.ts.
 */

import { Pipeline, type StageTelemetry } from "./pipeline";
import { budgetPlan, type RetrievalPlan } from "./deadline";
import type { LoadedIndex } from "../retrieval/loader";
import { fuse, fusionMargin, lexicalOverlap, type FusedHit, type StrategyHits } from "../retrieval/fusion";
import type { SourceStore } from "../sources/store";
import {
  gateInput, gateRetrieval, gateGrounding, DEFAULT_THRESHOLDS,
  type ConfidenceThresholds, type GateResult, type Refusal,
} from "../guardrails/gates";

export interface Encoder {
  /** Returns a raw 384-d L2-normalised embedding for a query string. */
  encodeQuery(text: string): Promise<Float32Array>;
}

/**
 * User passages live above this ordinal in the shared id space.
 *
 * Fusion, rescoring and citation all key on one integer, so corpus and user
 * passages must not collide. The corpus is ~99k passages and the format caps
 * far below 10M, so a fixed offset is simpler and faster than a tagged pair —
 * one comparison rather than an object dereference in the hot loop.
 */
export const USER_BASE = 10_000_000;

export interface CitationSource {
  kind: "corpus" | "user";
  /** Present for user sources: the document's title. */
  title?: string;
  id?: number;
}

export interface Citation {
  passageId: number;
  text: string;
  score: number;
  strategies: string[];
  source: CitationSource;
}

export interface RagAnswer {
  status: "answered" | "refused";
  answer: string;
  refusal?: Refusal;
  citations: Citation[];
  confidence: number;
  telemetry: StageTelemetry[];
  totalMs: number;
  /** What retrieval was actually allowed to do, after deadline pressure. */
  plan: RetrievalPlan;
  /** Set when the answer came from cache — reported separately so cached runs
   *  never flatter the latency numbers. */
  cached?: boolean;
  /**
   * Fused passage ids in rank order, populated whether or not the answer
   * survived the guardrails.
   *
   * `citations` is deliberately empty on a refusal — the user must not be shown
   * evidence for an answer the system declined to give. But evaluation needs
   * the opposite: "retrieval found the right passage and the gate rejected it"
   * and "retrieval never found it" are completely different failures, and with
   * only `citations` they are indistinguishable. This field is what makes the
   * multilingual stress test able to tell them apart.
   */
  retrieved?: number[];
}

export interface RagConfig {
  nprobe: number;
  perStrategyK: number;
  fuseTopN: number;
  rescoreTopN: number;
  thresholds: ConfidenceThresholds;
  globalBudgetMs: number;
}

export const DEFAULT_CONFIG: RagConfig = {
  nprobe: 12,
  perStrategyK: 24,
  fuseTopN: 24,
  rescoreTopN: 16,
  thresholds: DEFAULT_THRESHOLDS,
  globalBudgetMs: 200,
};

/**
 * Warm-up set: both scripts, short and long, question and fragment.
 * Deliberately not real corpus queries — the cache is cleared afterwards, but
 * warming on real questions would make the first genuine ask suspiciously fast.
 */
const WARMUP_QUERIES = [
  "what is a corporation",
  "भारत की राजधानी क्या है?",
  "how does a professional certification process work in practice",
  "किसी संगठन के वित्तीय विवरण में क्या शामिल होता है",
  "define",
  "who",
  "a somewhat longer question with a number 1947 and a Latin acronym NATO in it",
  "एक लंबा प्रश्न जिसमें संख्या 2024 और अंग्रेज़ी शब्द hospital शामिल है",
];

/**
 * Hard ceiling on the text handed to the encoder.
 *
 * Embedding is the only stage whose cost is set by the *input* rather than by
 * the corpus, and until this existed nothing bounded it. Measured on the
 * parallel query set (`bench/multilingual.ts`):
 *
 *     30 chars ->   2.2 ms embed        <- p50 query
 *     55 chars ->   3.6 ms
 *   2834 chars ->  78.4 ms
 *   6594 chars -> 219.7 ms              <- P100 216 ms, budget blown
 *
 * A single pathological query took the whole system over its 200 ms guarantee.
 * The deadline planner could not save it: the planner degrades *retrieval*,
 * which costs about a millisecond, and by the time it runs the embedding has
 * already been paid for. A budget that only holds for well-formed input is not
 * a budget, so the bound has to be applied before the cost is incurred.
 *
 * The value is set from measurement, not from the model's 512-token limit,
 * because characters and tokens are not the same thing across scripts. e5's
 * vocabulary covers Devanagari far better than Assamese or Kannada, so the same
 * character count becomes very different token counts — and attention cost is
 * quadratic in tokens. Embed cost at the cap, worst script measured:
 *
 *       chars    Node    in-browser (~4x, WASM)
 *         256   7.6 ms          ~30 ms
 *         320   9.3 ms          ~37 ms
 *         512  15.3 ms          ~60 ms   <- brushes the 60 ms stage budget
 *
 * 512 characters put the slowest script right on the embed stage's budget, so
 * it timed out, and the answer failed rather than merely being slow. 320 leaves
 * roughly a third of the stage budget spare on this machine while still being
 * four and a half times the p99 real query (71 characters) and about fifteen
 * seconds of continuous speech.
 *
 * Truncating rather than refusing is deliberate: this is a voice interface, and
 * answering the first part of a rambling question is far more useful than
 * declining all of it. When it fires it is reported, because a silently
 * shortened question is a wrong answer waiting to happen.
 */
export const MAX_QUERY_CHARS = 320;

/** Trim to the cap at a word boundary, so the encoder never sees half a word. */
function clampQuery(q: string): string {
  if (q.length <= MAX_QUERY_CHARS) return q;
  const cut = q.slice(0, MAX_QUERY_CHARS);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > MAX_QUERY_CHARS * 0.6 ? cut.slice(0, lastSpace) : cut).trim();
}

/** Split on Devanagari danda and ASCII terminators. */
function sentences(text: string): string[] {
  return text.split(/(?<=[।.!?])\s+/).map((s) => s.trim()).filter(Boolean);
}

/**
 * Extractive answer: the sentences of the winning passage that best overlap the
 * query, in original order.
 *
 * Grounded by construction — every character is copied from a retrieved
 * passage — which is why the fast path needs no hallucination check. Gate 3
 * exists for the LLM path, where text is generated rather than copied.
 */
function extractAnswer(query: string, passage: string, maxSentences = 2): string {
  const sents = sentences(passage);
  if (sents.length <= maxSentences) return passage;
  const scored = sents.map((s, i) => ({ s, i, score: lexicalOverlap(query, s) }));
  scored.sort((a, b) => b.score - a.score);
  const picked = scored.slice(0, maxSentences).sort((a, b) => a.i - b.i);
  // If nothing overlaps at all, the lead sentences are a safer summary than a
  // confidently-wrong "best match".
  if (picked.every((p) => p.score === 0)) return sents.slice(0, maxSentences).join(" ");
  return picked.map((p) => p.s).join(" ");
}

export class RagEngine {
  private queryVec: Float32Array;
  private cache = new Map<string, RagAnswer>();
  private store: SourceStore | null = null;

  constructor(
    private readonly index: LoadedIndex,
    private readonly encoder: Encoder,
    private readonly cfg: RagConfig = DEFAULT_CONFIG,
  ) {
    this.queryVec = new Float32Array(index.manifest.dim);
  }

  /** User sources are attached after boot, once the encoder is up. */
  attachSources(store: SourceStore): void { this.store = store; }

  /** Passage text for either corpus. */
  textOf(passageId: number): string {
    return passageId >= USER_BASE
      ? (this.store?.passages[passageId - USER_BASE] ?? "")
      : (this.index.passages[passageId] ?? "");
  }

  private sourceOf(passageId: number): CitationSource {
    if (passageId < USER_BASE) return { kind: "corpus" };
    const s = this.store?.sourceOfPassage(passageId - USER_BASE);
    return { kind: "user", title: s?.title, id: s?.id };
  }

  /**
   * Warm every code path with throwaway queries.
   *
   * This exists specifically for P100. The first execution of any JS function
   * runs in the interpreter before the JIT promotes it, and the first typed-array
   * touch faults pages in. Measured here: the first query costs ~20ms of embed
   * against ~6ms once warm, so an under-warmed engine sets P100 with its own
   * first request rather than with anything about the corpus.
   *
   * Two scripts and a range of lengths, because tokenisation and the resulting
   * sequence length are what the graph specialises on — warming only on a short
   * ASCII string leaves the Devanagari path cold. The cost is ~150ms of a
   * multi-second load, which is the cheapest tail reduction available anywhere
   * in this system.
   */
  async warmup(samples: string[] = WARMUP_QUERIES): Promise<void> {
    for (const s of samples) {
      try { await this.ask(s, { skipCache: true }); } catch { /* warmup must never throw */ }
    }
    this.cache.clear();
  }

  async ask(query: string, opts: { skipCache?: boolean } = {}): Promise<RagAnswer> {
    const key = query.trim().toLowerCase();
    if (!opts.skipCache) {
      const hit = this.cache.get(key);
      if (hit) return { ...hit, cached: true };
    }

    const pipe = new Pipeline(this.cfg.globalBudgetMs);
    const idx = this.index;
    const cfg = this.cfg;
    const qv = this.queryVec;
    const store = this.store;

    let refusal: GateResult | null = null;
    /** Original length, when the query had to be clamped before embedding. */
    let truncatedFrom = 0;
    let fused: FusedHit[] = [];
    let citations: Citation[] = [];
    let confidence = 0;
    // Whether gate 2 is about to read a calibrated cosine or a raw binary proxy.
    let rescored = false;
    let plan: RetrievalPlan = {
      nprobe: cfg.nprobe, perStrategyK: cfg.perStrategyK,
      rescoreTopN: cfg.rescoreTopN, degraded: false,
    };

    pipe
      // ---- gate 1 --------------------------------------------------------
      .add<string, string>({
        name: "guard:input",
        budgetMs: 5,
        run: (q) => {
          const g = gateInput(q);
          if (!g.pass) refusal = g;
          // Clamped here rather than in `gateInput`, which answers yes/no and
          // has no way to hand back a modified query. The clamped text is what
          // every later stage sees, so the answer, the citations and the
          // lexical overlap all describe the question that was actually asked.
          const clamped = clampQuery(q);
          if (clamped.length !== q.length) {
            truncatedFrom = q.length;
          }
          return clamped;
        },
      })
      // ---- embed ---------------------------------------------------------
      .add<string, string>({
        name: "embed",
        budgetMs: 60,
        retries: 1,           // ONNX session can transiently fail under memory pressure
        onError: "FAIL",      // load-bearing: no vector, no retrieval
        run: async (q) => {
          if (refusal) return q;
          const raw = await this.encoder.encodeQuery(q);
          idx.projectQuery(raw, qv);
          return q;
        },
      })
      // ---- retrieve across all strategies, both corpora -------------------
      .add<string, string>({
        name: "retrieve",
        budgetMs: 80,
        run: (q, ctx) => {
          if (refusal) return q;
          // The hard cap in practice. Embedding is the variable cost — on a slow
          // phone it can take 10x what it takes here — so retrieval sizes itself
          // against what is *left*, not against what it would like.
          plan = budgetPlan(ctx.remaining(), cfg, store?.activeChunks ?? 0);

          const hits: StrategyHits[] = [];
          for (const [name, ix] of idx.indices) {
            const r = ix.search(qv, plan.nprobe, plan.perStrategyK);
            if (r.n > 0) {
              hits.push({ strategy: name, parentIds: r.parentIds, scores: r.scores, n: r.n });
            }
          }
          // User sources join the same fusion, offset into the shared id space.
          if (store) {
            for (const b of store.search(qv, plan.perStrategyK)) {
              if (b.n === 0) continue;
              const ids = new Int32Array(b.n);
              for (let i = 0; i < b.n; i++) ids[i] = USER_BASE + b.parentIds[i];
              hits.push({ strategy: b.strategy, parentIds: ids, scores: b.scores, n: b.n });
            }
          }
          fused = fuse(hits, cfg.fuseTopN);
          return q;
        },
      })
      // ---- rescore -------------------------------------------------------
      .add<string, string>({
        name: "rescore",
        budgetMs: 30,
        // This stage is NOT the optional one it looks like. It converts the
        // binary Hamming proxy into real cosine similarity, and gate 2's
        // threshold was calibrated on that cosine. Skipping it does not just
        // lose a little ranking quality — it leaves the gate comparing 0.4788
        // against a number on a different scale entirely.
        //
        // It also costs ~0.03ms for 16 passages. `bench/deadline.ts` caught an
        // earlier version reserving 12ms for it, which under a tight budget
        // skipped it and refused nearly everything. The reservation is now
        // proportionate to what it actually costs; depth is controlled by
        // `plan.rescoreTopN`, which is the knob that belongs to the deadline.
        minRemainingMs: 2,
        onError: "DEGRADE",
        fallback: (q) => q,
        run: (q) => {
          if (refusal || !fused.length) return q;
          idx.rescorer.prepare(qv);
          store?.prepareRescore(qv);
          const n = Math.min(plan.rescoreTopN, fused.length);
          for (let i = 0; i < n; i++) {
            const pid = fused[i].passageId;
            fused[i].bestRawScore = pid >= USER_BASE
              ? (store?.rescore(pid - USER_BASE) ?? 0)
              : idx.rescorer.score(pid);
          }
          fused = fused.slice(0, n).sort((a, b) => b.bestRawScore - a.bestRawScore);
          rescored = true;
          return q;
        },
      })
      // ---- gate 2 --------------------------------------------------------
      .add<string, string>({
        name: "guard:retrieval",
        budgetMs: 5,
        run: (q) => {
          if (refusal) return q;
          if (!fused.length) {
            refusal = {
              pass: false, reason: "LOW_CONFIDENCE",
              message: "I couldn't find anything relevant in my sources.",
            };
            return q;
          }
          const top = fused[0];
          confidence = top.bestRawScore;

          // If rescoring did not run, `bestRawScore` is `1 - hamming/dim`, not
          // cosine, and there is no calibrated threshold for that scale. Rather
          // than compare against a number fitted on a different quantity — which
          // silently refuses almost everything — the score test is dropped and
          // the structural signals (cross-strategy agreement, fusion margin,
          // lexical overlap) carry the decision alone. That is a weaker gate,
          // and it is reported as one rather than passed off as the calibrated
          // path.
          const thresholds = rescored
            ? cfg.thresholds
            : { ...cfg.thresholds, minTopScore: -Infinity };
          if (!rescored) {
            plan = { ...plan, degraded: true,
                     reason: "rescore skipped — confidence is uncalibrated" };
          }

          const g = gateRetrieval({
            topScore: top.bestRawScore,
            margin: fusionMargin(fused),
            strategyAgreement: top.agreement,
            lexicalOverlap: lexicalOverlap(q, this.textOf(top.passageId)),
          }, thresholds);
          if (!g.pass) refusal = g;
          return q;
        },
      })
      // ---- extract -------------------------------------------------------
      .add<string, string>({
        name: "answer:extract",
        budgetMs: 15,
        run: (q) => {
          if (refusal) return q;
          citations = fused.slice(0, 3).map((f) => ({
            passageId: f.passageId,
            text: this.textOf(f.passageId),
            score: f.bestRawScore,
            strategies: Object.keys(f.perStrategyRank),
            source: this.sourceOf(f.passageId),
          }));
          return q;
        },
      });

    const run = await pipe.run<string>(query);

    // A truncated question is a different question, so it is surfaced with the
    // answer rather than left as a silent difference between what was asked and
    // what was searched for.
    if (truncatedFrom) {
      plan = {
        ...plan,
        degraded: true,
        reason: `question truncated from ${truncatedFrom} to ${MAX_QUERY_CHARS} characters ` +
                `before embedding — only the first part was searched`,
      };
    }

    // Captured before the branch below, so a refusal reports what retrieval
    // actually ranked rather than losing it along with the citations.
    const retrieved = fused.slice(0, 10).map((f) => f.passageId);

    let out: RagAnswer;
    if (refusal) {
      out = {
        status: "refused",
        answer: (refusal as GateResult).message ?? "I can't answer that.",
        refusal: (refusal as GateResult).reason,
        citations: [],
        confidence,
        telemetry: run.telemetry,
        totalMs: run.totalMs,
        plan,
        retrieved,
      };
    } else if (!run.ok || !citations.length) {
      out = {
        status: "refused",
        answer: "Something went wrong while answering. Please try again.",
        refusal: "LOW_CONFIDENCE",
        citations: [],
        confidence,
        telemetry: run.telemetry,
        totalMs: run.totalMs,
        plan,
        retrieved,
      };
    } else {
      out = {
        status: "answered",
        answer: extractAnswer(query, citations[0].text),
        citations,
        confidence,
        telemetry: run.telemetry,
        totalMs: run.totalMs,
        plan,
        retrieved,
      };
    }

    // Bounded cache: a demo session asking the same question repeatedly should
    // not grow memory without limit.
    if (this.cache.size > 256) this.cache.clear();
    this.cache.set(key, out);
    return out;
  }

  /**
   * Adding or removing a source changes what the right answer is, so every
   * cached answer from before the change is now potentially wrong.
   */
  invalidate(): void { this.cache.clear(); }

  /**
   * Verify an LLM-synthesised answer before it is allowed to replace the
   * extractive one. Runs off the fast path.
   */
  verifySynthesis(answer: string, citations: Citation[]): GateResult {
    return gateGrounding(answer, citations.map((c) => c.text));
  }
}
