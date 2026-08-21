/**
 * Cross-lingual stress test: the same question in 15 languages.
 *
 * The corpus is Hindi, the embedder is not, so "it retrieves cross-lingually"
 * is a claim that needs a number against it.
 *
 * The queries are MSMARCO-XI's own professional translations, joined on
 * `query_id` by `pipeline/src/parallel_queries.py`. Translating them ourselves
 * would fold translation error into the measurement, where it is
 * indistinguishable from retrieval error.
 *
 * Two numbers per language, because they fail for different reasons:
 *
 *   hit@k     did retrieval rank the gold passage? Read from
 *             `RagAnswer.retrieved`, which is populated even on a refusal.
 *   answered  did the guardrails let it through? The threshold is calibrated
 *             on Hindi, so a language sitting at lower cosine gets refused more
 *             often while still ranking correctly. One number would hide that,
 *             and hide it flatteringly.
 *
 * Latency is per language too: heavier grapheme clusters make more tokens for
 * the same sentence, and the forward pass dominates the budget.
 */

import type { RagAnswer } from "../harness/rag";

export interface LangSpec { code: string; name: string; tag: string }

export interface ParallelSet {
  query_id: number;
  eng_query: string;
  query_type: string | null;
  /** Gold passage ordinals in the shipped index. */
  gold: number[];
  by_lang: Record<string, string>;
}

export interface MultilingualData {
  languages: LangSpec[];
  n: number;
  queries: ParallelSet[];
}

export interface LangResult {
  code: string;
  name: string;
  tag: string;
  n: number;
  p50: number;
  p70: number;
  p95: number;
  p100: number;
  mean: number;
  /** Share of queries where a gold passage was ranked first / in the top 5. */
  hit1: number;
  hit5: number;
  /** Share the guardrails allowed through. */
  answered: number;
  /** Share over the 200 ms requirement. Expected to be zero. */
  overBudget: number;
  /** Mean confidence on answered queries - the number the gate thresholds. */
  meanConfidence: number;
}

export interface MultilingualReport {
  perLang: LangResult[];
  /** Every language, pooled - the headline latency claim. */
  overall: { n: number; p50: number; p70: number; p100: number; overBudget: number };
  /** hit@5 of the corpus's own language, as the ceiling everything else is read against. */
  baselineHit5: number;
}

export type AskFn = (q: string) => Promise<RagAnswer>;

export const ENGLISH: LangSpec = { code: "eng", name: "English", tag: "en-IN" };

/**
 * Run the sweep.
 *
 * `onProgress` is called between languages rather than between queries: in the
 * browser this loop must yield to let the UI paint, and yielding per query
 * would add scheduler noise to the very timings being collected.
 *
 * The yield happens immediately after the progress callback and before any
 * timing starts, which is not incidental. `totalMs` is measured on the main
 * thread across an `await` on the encoder worker, so anything the browser
 * decides to do during that await - including painting the progress line this
 * function just wrote - is counted as query time. Measured: with the progress
 * update left to paint whenever it liked, the in-browser sweep reported a P100
 * of 122 ms against 13 ms for the identical queries run without it. The
 * reporting UI was nine tenths of the number it was reporting.
 *
 * So: draw, let the frame land, then start the clock.
 */
export async function runMultilingual(
  data: MultilingualData,
  ask: AskFn,
  opts: {
    perLang?: number;
    onProgress?: (done: number, total: number, label: string) => void;
    /**
     * Awaited after each progress update, before that language's queries are
     * timed. A browser caller should wait for a frame here, not just a
     * macrotask - see the note above on what a stray paint does to `totalMs`.
     */
    yieldFn?: () => Promise<void>;
  } = {},
): Promise<MultilingualReport> {
  const perLang = Math.min(opts.perLang ?? 60, data.queries.length);
  const langs: LangSpec[] = [...data.languages, ENGLISH];
  const perLangResults: LangResult[] = [];
  const pooled: number[] = [];
  let pooledOver = 0;

  for (let li = 0; li < langs.length; li++) {
    const lang = langs[li];
    opts.onProgress?.(li, langs.length, lang.name);
    // Let the progress line actually paint before the stopwatch starts.
    if (opts.yieldFn) await opts.yieldFn();

    const times: number[] = [];
    let hit1 = 0, hit5 = 0, answered = 0, over = 0, confSum = 0;

    for (let i = 0; i < perLang; i++) {
      const set = data.queries[i];
      const text = lang.code === "eng" ? set.eng_query : set.by_lang[lang.code];
      if (!text) continue;

      // `skipCache` is not optional here. The same query id appears once per
      // language, but repeated runs of the sweep would otherwise be timing
      // the cache rather than the pipeline.
      const r = await ask(text);
      times.push(r.totalMs);
      if (r.totalMs >= 200) over++;
      if (r.status === "answered") { answered++; confSum += r.confidence; }

      const ranked = r.retrieved ?? [];
      const gold = new Set(set.gold);
      if (ranked.length && gold.has(ranked[0])) hit1++;
      if (ranked.slice(0, 5).some((p) => gold.has(p))) hit5++;
    }

    const n = times.length || 1;
    const p = percentiles(times);
    perLangResults.push({
      code: lang.code, name: lang.name, tag: lang.tag,
      n: times.length,
      p50: p.p50, p70: p.p70, p95: p.p95, p100: p.p100, mean: p.mean,
      hit1: hit1 / n, hit5: hit5 / n,
      answered: answered / n,
      overBudget: over / n,
      meanConfidence: answered ? confSum / answered : 0,
    });
    pooled.push(...times);
    pooledOver += over;
  }

  const pp = percentiles(pooled);
  const hindi = perLangResults.find((r) => r.code === "hin");
  return {
    perLang: perLangResults,
    overall: {
      n: pooled.length,
      p50: pp.p50, p70: pp.p70, p100: pp.p100,
      overBudget: pooled.length ? pooledOver / pooled.length : 0,
    },
    baselineHit5: hindi?.hit5 ?? 0,
  };
}

export function percentiles(xs: number[]): {
  p50: number; p70: number; p95: number; p100: number; mean: number;
} {
  if (!xs.length) return { p50: 0, p70: 0, p95: 0, p100: 0, mean: 0 };
  const s = [...xs].sort((a, b) => a - b);
  // Nearest-rank, not interpolated. P100 must be an observed measurement - an
  // interpolated maximum is not a query anyone actually ran.
  const at = (q: number) => s[Math.min(s.length - 1, Math.ceil(q * s.length) - 1)];
  return {
    p50: at(0.5), p70: at(0.7), p95: at(0.95), p100: s[s.length - 1],
    mean: s.reduce((a, b) => a + b, 0) / s.length,
  };
}

/** Fixed-width report, shared by the browser drawer and the Node harness. */
export function formatReport(rep: MultilingualReport): string {
  const pad = (s: string, n: number) => s.padEnd(n);
  const num = (v: number, n = 6, d = 2) => v.toFixed(d).padStart(n);
  const pct = (v: number) => `${(v * 100).toFixed(1).padStart(5)}%`;

  let s = "MULTILINGUAL STRESS TEST - same questions, 14 Indian languages + English\n";
  s += "corpus is Hindi; every other row is cross-lingual retrieval into it\n";
  s += "-".repeat(74) + "\n";
  s += `${pad("language", 12)}${pad("n", 5)}${pad("p50", 8)}${pad("p100", 8)}` +
       `${pad("hit@1", 8)}${pad("hit@5", 8)}${pad("answered", 10)}conf\n`;
  s += "-".repeat(74) + "\n";
  for (const r of rep.perLang) {
    s += `${pad(r.name, 12)}${pad(String(r.n), 5)}${num(r.p50)}  ${num(r.p100)}  ` +
         `${pct(r.hit1)}   ${pct(r.hit5)}   ${pct(r.answered)}     ${r.meanConfidence.toFixed(3)}\n`;
  }
  s += "-".repeat(74) + "\n";
  s += `pooled  n=${rep.overall.n}  P50 ${rep.overall.p50.toFixed(2)} ms  ` +
       `P70 ${rep.overall.p70.toFixed(2)} ms  P100 ${rep.overall.p100.toFixed(2)} ms\n`;
  s += `over 200 ms budget: ${(rep.overall.overBudget * 100).toFixed(2)}%  ` +
       `-> ${rep.overall.p100 < 200 ? "PASS" : "FAIL"}\n`;
  return s;
}
