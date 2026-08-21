/**
 * The RAG pipeline as harness stages.
 *
 * Everything here is inside the 200ms budget. Synthesis is not: it runs
 * afterwards and replaces the extractive answer only once it arrives and passes
 * the grounding gate.
 *
 * Stage budgets sit well above measured cost so a slow machine degrades instead
 * of failing, and `deadline.ts` makes the global budget hard rather than
 * aspirational by shrinking the retrieval plan when time runs short.
 *
 * The shipped index and the user's sources are searched separately, IVF against
 * flat, but their hits fuse in one RRF pass under one calibrated confidence.
 * That only works because both sides quantise identically (`sources/quantise.ts`).
 */

import { Pipeline, type StageTelemetry } from "./pipeline";
import { budgetPlan, type RetrievalPlan } from "./deadline";
import type { LoadedIndex } from "../retrieval/loader";
import { Fuser, fusionMargin, lexicalOverlap, type FusedHit, type StrategyHits } from "../retrieval/fusion";
import type { SourceStore } from "../sources/store";
import {
  gateInput, gateRetrieval, gateGrounding, gateCitations, DEFAULT_THRESHOLDS,
  type ConfidenceThresholds, type GateResult, type Refusal, type RetrievalSignals,
} from "../guardrails/gates";
import { foldRomanisedHindi } from "../retrieval/translit";
import { contentTokens } from "../retrieval/tokens";

export interface Encoder {
  /** Returns a raw 384-d L2-normalised embedding for a query string. */
  encodeQuery(text: string): Promise<Float32Array>;
}

/**
 * User passages live above this ordinal in the shared id space.
 *
 * Fusion, rescoring and citation all key on one integer, so corpus and user
 * passages must not collide. A fixed offset is one comparison in the hot loop
 * where a tagged pair would be an object dereference.
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
  /** Set when the answer came from cache, so cached runs can be excluded from
   *  the latency numbers. */
  cached?: boolean;
  /**
   * Fused passage ids in rank order, populated whether or not the answer
   * survived the guardrails.
   *
   * `citations` is empty on a refusal - the user must not be shown evidence for
   * an answer that was declined. Evaluation needs the opposite: "found it and
   * the gate rejected it" and "never found it" are different failures, and
   * `citations` alone cannot distinguish them.
   */
  retrieved?: number[];
  /** The four signals gate 2 decided on. Populated on refusals too, which is
   *  where they matter. */
  signals?: RetrievalSignals;
}

export interface RagConfig {
  nprobe: number;
  perStrategyK: number;
  fuseTopN: number;
  rescoreTopN: number;
  /**
   * Candidates each index may scan before it stops.
   *
   * The ceiling on work per query, and the reason P100 is flat: without it,
   * cost is set by how densely populated the clusters a query happens to land
   * in are. Swept against retrieval quality - at nprobe 24 raising it from
   * 3072 to 6144 moved hit@5 from 62.4% to 64.1%, and 9216 added nothing.
   */
  maxCandidates: number;
  thresholds: ConfidenceThresholds;
  globalBudgetMs: number;
  /**
   * Which indices take part, or null for all of them.
   *
   * Exists for the leave-one-out ablation (`bench/ablation.ts`), which is the
   * only way to say what a strategy is worth rather than asserting it. Left
   * null everywhere in the product: a strategy that is not worth its place
   * should be deleted from the build, not switched off at query time.
   */
  enabledStrategies?: string[] | null;
}

/**
 * Swept against retrieval quality on 415 answerable queries, graded on rank
 * before the guardrails so a setting cannot flatter itself by moving queries
 * across gate 2's threshold. nprobe saturates at 24 and the candidate cap at
 * 6144; both were raised from the point where the cost curve is still flat,
 * because after the search was rewritten there was budget to spend on recall.
 */
export const DEFAULT_CONFIG: RagConfig = {
  nprobe: 24,
  perStrategyK: 24,
  fuseTopN: 24,
  rescoreTopN: 16,
  maxCandidates: 6144,
  thresholds: DEFAULT_THRESHOLDS,
  globalBudgetMs: 200,
  enabledStrategies: null,
};

/**
 * Warm-up set: both scripts, short and long, question and fragment.
 *
 * Not real corpus queries. The cache is cleared afterwards, but warming on a
 * real question would make the first genuine ask artificially fast.
 *
 * The last two run at the clamp length on purpose. The ONNX graph sizes its
 * buffers to the longest sequence it has seen, so without them the first long
 * question of a session pays for that allocation and sets P100 on its own.
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
  "a deliberately long question that runs to the clamp so the encoder allocates " +
  "its widest buffers during warm-up rather than during somebody's first real " +
  "question about a certification process, a hospital admission in 1947, or the " +
  "financial statements of an organisation with a NATO contract attached",
  "यह एक जानबूझकर लंबा प्रश्न है जो अधिकतम लंबाई तक जाता है ताकि एनकोडर अपने सबसे " +
  "बड़े बफ़र वार्म-अप के दौरान आवंटित करे, न कि किसी उपयोगकर्ता के पहले वास्तविक " +
  "प्रश्न के दौरान, जैसे किसी संगठन के वित्तीय विवरण, अस्पताल में भर्ती, या 1947 " +
  "के किसी ऐतिहासिक अनुबंध के बारे में",
];

/**
 * Hard ceiling on the text handed to the encoder.
 *
 * Embedding is the one stage whose cost comes from the input rather than the
 * corpus, and it is paid before the deadline planner runs, so a long enough
 * query blows the budget with nothing left to degrade.
 *
 * Set by measurement, not by the model's 512-token limit: characters and tokens
 * diverge across scripts (e5 tokenises Devanagari far better than Assamese or
 * Kannada) and attention is quadratic in tokens. At 512 characters the slowest
 * script sits on the embed budget and times out. 320 leaves a third spare and
 * is still 4.5x the p99 real query.
 *
 * Truncate rather than refuse: answering the first part of a rambling spoken
 * question beats declining all of it. Reported when it fires.
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
 * Grounded by construction, since every character is copied from a retrieved
 * passage. Gate 3 exists for the LLM path, where text is generated instead.
 */
function extractAnswer(query: string, passage: string, maxSentences = 2): string {
  const sents = sentences(passage);
  if (sents.length <= maxSentences) return passage;
  const scored = sents.map((s, i) => ({ s, i, score: lexicalOverlap(query, s) }));
  scored.sort((a, b) => b.score - a.score);
  const picked = scored.slice(0, maxSentences).sort((a, b) => a.i - b.i);
  // With no overlap at all, the lead sentences summarise better than an
  // arbitrary "best match".
  if (picked.every((p) => p.score === 0)) return sents.slice(0, maxSentences).join(" ");
  return picked.map((p) => p.s).join(" ");
}

export class RagEngine {
  private queryVec: Float32Array;
  private cache = new Map<string, RagAnswer>();
  private store: SourceStore | null = null;

  /**
   * Whether the shipped MS MARCO corpus takes part in retrieval. On by default,
   * so the app answers from the first question rather than starting empty.
   *
   * Provenance is handled by attribution instead: every answer names its source
   * and a corpus passage says so, so it cannot be mistaken for the visitor's own
   * document. Bench scripts set this explicitly, so a change here cannot move a
   * reported number.
   */
  private corpusEnabled = true;

  /** Resolves when the query currently running finishes. See `ask`. */
  private inFlight: Promise<void> | null = null;

  /** Holds its own working set; see `retrieval/fusion.ts`. */
  private readonly fuser = new Fuser();

  constructor(
    private readonly index: LoadedIndex,
    private readonly encoder: Encoder,
    private readonly cfg: RagConfig = DEFAULT_CONFIG,
  ) {
    this.queryVec = new Float32Array(index.manifest.dim);
  }

  /** User sources are attached after boot, once the encoder is up. */
  attachSources(store: SourceStore): void { this.store = store; }

  /**
   * The projected query vector from the most recent `ask`.
   *
   * Exposed so the precomputed-answer store can match on it instead of
   * embedding the question a second time. Valid only until the next call, and
   * only when that call reached the embed stage.
   */
  get lastQueryVector(): Float32Array { return this.queryVec; }

  get corpusOn(): boolean { return this.corpusEnabled; }

  /** Turning the corpus on or off changes what the right answer is. */
  setCorpusEnabled(on: boolean): void {
    if (this.corpusEnabled === on) return;
    this.corpusEnabled = on;
    this.invalidate();
  }

  /**
   * Is there anything at all to search?
   *
   * False with the corpus off and no sources added. Distinguished from
   * LOW_CONFIDENCE, which claims a search happened and found nothing good
   * enough.
   */
  private searchable(): boolean {
    return this.corpusEnabled || !!this.store?.hasEnabled;
  }

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
   * For P100. The first execution of a JS function runs interpreted before the
   * JIT promotes it, and the first typed-array touch faults pages in: an
   * unwarmed first query costs ~20ms of embed against ~6ms warm.
   *
   * Two scripts and a range of lengths, because the graph specialises on
   * tokenisation and sequence length - warming only on short ASCII leaves the
   * Devanagari path cold. Costs ~150ms of a multi-second load.
   */
  async warmup(samples: string[] = WARMUP_QUERIES): Promise<void> {
    // Forced on for the duration, whatever the user's setting. With the corpus
    // off and no sources yet, every warm-up query would stop at gate 1 and
    // leave retrieve, rescore and fuse cold.
    const restore = this.corpusEnabled;
    this.corpusEnabled = true;
    try {
      for (const s of samples) {
        try { await this.ask(s, { skipCache: true }); } catch { /* warmup must never throw */ }
      }
    } finally {
      this.corpusEnabled = restore;
      this.cache.clear();
    }
  }

  async ask(query: string, opts: { skipCache?: boolean } = {}): Promise<RagAnswer> {
    const key = query.trim().toLowerCase();
    if (!opts.skipCache) {
      const hit = this.cache.get(key);
      if (hit) return { ...hit, cached: true };
    }

    // One query at a time. Scratch buffers are shared across stages and across
    // the indices, and speculative retrieval on a partial transcript is often
    // still running when the final transcript arrives. Queueing costs a
    // microtask; overlapping would let one query embed into another's vector.
    const prev = this.inFlight;
    let release!: () => void;
    this.inFlight = new Promise<void>((r) => { release = r; });
    if (prev) {
      try { await prev; } catch { /* the previous query's failure is its own */ }
    }
    try {
      return await this.runQuery(query, key);
    } finally {
      release();
    }
  }

  private async runQuery(query: string, key: string): Promise<RagAnswer> {
    const pipe = new Pipeline(this.cfg.globalBudgetMs);
    const idx = this.index;
    const cfg = this.cfg;
    const qv = this.queryVec;
    const store = this.store;

    let refusal: GateResult | null = null;
    /** Original length, when the query had to be clamped before embedding. */
    let truncatedFrom = 0;
    /** Set when romanised Hindi was folded to Devanagari before embedding. */
    let folded = "";
    let fused: FusedHit[] = [];
    let citations: Citation[] = [];
    let confidence = 0;
    let signals: RetrievalSignals | undefined;
    // Whether gate 2 is about to read a calibrated cosine or a raw binary proxy.
    let rescored = false;
    let plan: RetrievalPlan = {
      nprobe: cfg.nprobe, perStrategyK: cfg.perStrategyK,
      rescoreTopN: cfg.rescoreTopN, maxCandidates: cfg.maxCandidates, degraded: false,
    };

    pipe
      // ---- gate 1 --------------------------------------------------------
      .add<string, string>({
        name: "guard:input",
        budgetMs: 5,
        run: (q) => {
          const g = gateInput(q);
          if (!g.pass) refusal = g;
          // After gateInput, so an unsafe or injected query gets its own message
          // rather than being told to add a source.
          else if (!this.searchable()) {
            refusal = {
              pass: false,
              reason: "NO_SOURCES",
              message: "There's nothing to search yet. Add a source - paste some " +
                       "text, drop a file, or add a link - and I'll answer from it.",
            };
          }
          // Clamped here rather than in `gateInput`, which answers yes/no and
          // cannot hand back a modified query. Every later stage sees the
          // clamped text, so answer, citations and overlap all describe the
          // question that was actually searched.
          const clamped = clampQuery(q);
          if (clamped.length !== q.length) {
            truncatedFrom = q.length;
          }
          // Romanised Hindi is folded to Devanagari before it reaches the
          // encoder. The corpus is Devanagari, and the folded form is what the
          // embedder and the lexical-overlap signal can both read; see
          // `retrieval/translit.ts`. A no-op for every other kind of query.
          const hindi = foldRomanisedHindi(clamped);
          if (hindi !== clamped) folded = hindi;
          return hindi;
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
          // Embedding is the variable cost - up to 10x on a slow phone - so
          // retrieval sizes itself against what is left rather than against
          // what it would prefer.
          plan = budgetPlan(ctx.remaining(), cfg, store?.activeChunks ?? 0);

          const hits: StrategyHits[] = [];
          const enabled = cfg.enabledStrategies;
          const on = (name: string) => !enabled || enabled.includes(name);
          if (this.corpusEnabled) {
            for (const [name, ix] of idx.indices) {
              if (!on(name)) continue;
              const r = ix.search(qv, plan.nprobe, plan.perStrategyK, plan.maxCandidates);
              if (r.n > 0) {
                hits.push({ strategy: name, parentIds: r.parentIds, scores: r.scores, n: r.n });
              }
            }
            // BM25 over the same passages. It takes the query's text rather
            // than its vector - it is the one index that never sees an
            // embedding - and it takes the folded text, so a romanised Hindi
            // question looks up Devanagari terms like any other.
            if (idx.lexical && on("lexical")) {
              const r = idx.lexical.search(contentTokens(q), plan.perStrategyK);
              if (r.n > 0) {
                hits.push({ strategy: "lexical", parentIds: r.parentIds, scores: r.scores, n: r.n });
              }
            }
          }
          // User sources join the same fusion, offset into the shared id space.
          // They get the same seven strategies as the corpus: the tokens are
          // what adds BM25, and on a document of names, dates and amounts it is
          // usually the only one that can match at all.
          if (store) {
            for (const b of store.search(qv, plan.perStrategyK,
                                         on("lexical") ? contentTokens(q) : undefined)) {
              if (b.n === 0) continue;
              const ids = new Int32Array(b.n);
              for (let i = 0; i < b.n; i++) ids[i] = USER_BASE + b.parentIds[i];
              hits.push({ strategy: b.strategy, parentIds: ids, scores: b.scores, n: b.n });
            }
          }
          fused = this.fuser.fuse(hits, cfg.fuseTopN);
          return q;
        },
      })
      // ---- rescore -------------------------------------------------------
      .add<string, string>({
        name: "rescore",
        budgetMs: 30,
        // Not optional despite the DEGRADE policy: this converts the binary
        // Hamming proxy into real cosine, and gate 2's threshold is calibrated
        // on that cosine. Skipping it leaves the gate comparing against a
        // different scale entirely.
        //
        // The reservation is small because the work is: ~0.03ms for 16
        // passages. Depth belongs to the deadline, via `plan.rescoreTopN`.
        minRemainingMs: 2,
        onError: "DEGRADE",
        fallback: (q) => q,
        run: (q) => {
          if (refusal || !fused.length) return q;
          // With the corpus off, nothing in `fused` can be a corpus id, so
          // preparing its rescorer would be work for a lookup never made.
          if (this.corpusEnabled) idx.rescorer.prepare(qv);
          store?.prepareRescore(qv);
          const n = Math.min(plan.rescoreTopN, fused.length);
          for (let i = 0; i < n; i++) {
            const pid = fused[i].passageId;
            fused[i].bestRawScore = pid >= USER_BASE
              ? (store?.rescore(pid - USER_BASE) ?? 0)
              : idx.rescorer.score(pid);
          }
          fused.length = n;
          fused.sort((a, b) => b.bestRawScore - a.bestRawScore);
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

          // Without rescoring, `bestRawScore` is `1 - hamming/dim` rather than
          // cosine, and no threshold is calibrated for that scale. The score
          // test is dropped rather than compared against a number fitted on a
          // different quantity, leaving agreement, margin and lexical overlap
          // to carry the decision. That is a weaker gate and is reported as one.
          //
          // The mid-band rescue corrects a corpus mismatch, so it is applied
          // only where the mismatch is: `minTopScore` was fitted on the shipped
          // corpus, so for corpus hits the calibration already holds. Enabling
          // it there raised coverage 88.5% -> 93.1% but cut abstention recall
          // 0.245 -> 0.172.
          const fromUserSource = top.passageId >= USER_BASE;
          const thresholds = rescored
            ? (fromUserSource
                ? cfg.thresholds
                : { ...cfg.thresholds, rescueMinScore: Infinity })
            : { ...cfg.thresholds, minTopScore: -Infinity, rescueMinScore: Infinity };
          if (!rescored) {
            plan = { ...plan, degraded: true,
                     reason: "rescore skipped - confidence is uncalibrated" };
          }

          signals = {
            topScore: top.bestRawScore,
            margin: fusionMargin(fused),
            strategyAgreement: top.agreement,
            lexicalOverlap: lexicalOverlap(q, this.textOf(top.passageId)),
          };
          const g = gateRetrieval(signals, thresholds);
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
            strategies: this.fuser.strategiesOf(f),
            source: this.sourceOf(f.passageId),
          }));
          return q;
        },
      });

    const run = await pipe.run<string>(query);

    // A truncated question is a different question, so the difference between
    // what was asked and what was searched is surfaced with the answer.
    if (truncatedFrom) {
      plan = {
        ...plan,
        degraded: true,
        reason: `question truncated from ${truncatedFrom} to ${MAX_QUERY_CHARS} characters ` +
                `before embedding - only the first part was searched`,
      };
    }

    // Captured before the branch below, so a refusal still reports what
    // retrieval ranked instead of losing it with the citations.
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
        signals,
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
        signals,
      };
    } else {
      out = {
        status: "answered",
        // The folded query where there is one: picking which sentence of a
        // Devanagari passage answers the question is a lexical-overlap
        // comparison, and a romanised query shares no tokens with either.
        answer: extractAnswer(folded || query, citations[0].text),
        citations,
        confidence,
        telemetry: run.telemetry,
        totalMs: run.totalMs,
        plan,
        retrieved,
        signals,
      };
    }

    // Bounded so a long session cannot grow memory without limit.
    if (this.cache.size > 256) this.cache.clear();
    this.cache.set(key, out);
    return out;
  }

  /** Adding or removing a source changes what the right answer is, so every
   *  answer cached before the change is potentially wrong. */
  invalidate(): void { this.cache.clear(); }

  /**
   * Retrieval for the model's `search_corpus` tool.
   *
   * The ordinary pipeline, then `retrieved` rather than `citations`. Gate 2
   * decides whether an answer reaches the user; this is the model asking to
   * look something up, and what it gets back is checked by gate 3 anyway.
   *
   * Passages already in front of the model are excluded, so a rephrasing that
   * finds the same thing costs a round trip rather than duplicate context.
   */
  async searchAgain(query: string, exclude: Set<number>, k = 3): Promise<Citation[]> {
    const r = await this.ask(query, { skipCache: true });
    const out: Citation[] = [];
    for (const pid of r.retrieved ?? []) {
      if (exclude.has(pid)) continue;
      const text = this.textOf(pid);
      if (!text) continue;
      out.push({ passageId: pid, text, score: 0, strategies: [], source: this.sourceOf(pid) });
      if (out.length >= k) break;
    }
    return out;
  }

  /**
   * Verify an LLM-synthesised answer before it replaces the extractive one.
   * Runs off the fast path.
   */
  verifySynthesis(answer: string, citations: Citation[]): GateResult {
    return gateGrounding(answer, citations.map((c) => c.text));
  }

  /**
   * Gate 3 over a tool-call answer, where the model named its sources.
   *
   * Citations first: an index outside the supplied set is a fabricated
   * citation and fails whatever the prose says. Then grounding, measured
   * against those excerpts only, which is stricter than checking the union - an
   * answer written from excerpt 1 while citing excerpt 3 no longer passes.
   *
   * A provider that ignores the field degrades to checking everything supplied
   * and records that in `checkedAgainst` rather than doing it silently.
   */
  verifyGenerated(
    answer: string,
    sources: Array<{ text: string }>,
    cited: number[],
  ): GateResult {
    const citations = gateCitations(cited, sources.length);
    if (!citations.pass) return citations;
    const used = cited.length
      ? cited.map((i) => sources[i - 1]?.text ?? "")
      : sources.map((s) => s.text);
    const g = gateGrounding(answer, used.filter(Boolean));
    return g.pass
      ? { ...g, detail: { ...(g.detail ?? {}), checkedAgainst: cited.length || sources.length } }
      : g;
  }
}
