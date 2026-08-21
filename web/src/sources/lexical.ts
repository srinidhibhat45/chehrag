/**
 * BM25 over the user's own sources.
 *
 * The corpus gets a lexical index because dense retrieval cannot hold an exact
 * token; a document someone adds needs it more, not less. The shipped corpus is
 * encyclopedia prose, where a paraphrase usually finds the passage. A CV, an
 * invoice, a config dump or a bank statement is mostly proper nouns, dates and
 * amounts — the exact tokens a 256-dimension average is worst at, and usually
 * the entire content of the question. "How old am I" against
 * `name: priya rao, age: 31` is a lexical match or it is nothing.
 *
 * Built here rather than shipped, because it is small and the source is not
 * known until someone adds it. Nothing is precomputed offline, nothing is
 * uploaded, and the build is a single pass over the passage text — a few
 * milliseconds for a document, against the seconds of embedding that follow it.
 *
 * The corpus version of this (`retrieval/lexical.ts`) reads a sorted hash
 * dictionary out of a shipped binary because it holds 99,703 terms and must not
 * parse anything at boot. Here a plain `Map` is the right structure: the
 * vocabulary is thousands of terms, it changes whenever a source is added, and
 * the whole thing is rebuilt more cheaply than a binary format could be parsed.
 */

import { contentTokens } from "../retrieval/tokens";

/** Same defaults as the corpus index, so the two behave alike where they meet. */
const K1 = 1.2;
const B = 0.75;

interface Posting { docs: number[]; tfs: number[] }

export class UserLexical {
  private postings = new Map<string, Posting>();
  /** Content-token count per global user-passage ordinal. */
  private len: number[] = [];
  /** Source id per passage, or -1 once its source is removed. */
  private source: number[] = [];
  private liveDocs = 0;
  private liveLen = 0;

  private scores = new Map<number, number>();

  /**
   * Index one source's passages.
   *
   * @param base   the global ordinal the first passage was granted
   * @param texts  passage text, in ordinal order
   */
  addPassages(base: number, texts: string[], sourceId: number): void {
    for (let i = 0; i < texts.length; i++) {
      const ord = base + i;
      // Duplicates are kept: BM25 needs term frequency, where the gates only
      // need presence. `contentTokens` de-duplicates, so tf is counted from the
      // raw list it was built from — approximated here by counting occurrences
      // of each content token in the text, which is what the corpus builder
      // does exactly.
      const tokens = tokenList(texts[i]);
      this.len[ord] = tokens.length;
      this.source[ord] = sourceId;
      this.liveDocs++;
      this.liveLen += tokens.length;

      const tf = new Map<string, number>();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const [term, n] of tf) {
        let p = this.postings.get(term);
        if (!p) { p = { docs: [], tfs: [] }; this.postings.set(term, p); }
        p.docs.push(ord);
        p.tfs.push(n);
      }
    }
  }

  /**
   * Drop a source.
   *
   * Its postings stay where they are and its passages are marked dead, matching
   * how `SourceStore` treats passage text: ordinals are permanent index keys and
   * renumbering would invalidate every surviving chunk's parent. Dead documents
   * are skipped at scoring time, so they cost a comparison and nothing else.
   *
   * Document frequency is left including them, which drifts a term's IDF
   * slightly after a removal. The alternative is a full rebuild on every
   * delete, and IDF over a few thousand passages is not sensitive enough to
   * justify it.
   */
  removeSource(sourceId: number): void {
    for (let i = 0; i < this.source.length; i++) {
      if (this.source[i] === sourceId) {
        this.source[i] = -1;
        this.liveDocs--;
        this.liveLen -= this.len[i];
        this.len[i] = 0;
      }
    }
  }

  clear(): void {
    this.postings.clear();
    this.len.length = 0;
    this.source.length = 0;
    this.liveDocs = 0;
    this.liveLen = 0;
  }

  get size(): number { return this.liveDocs; }

  /**
   * Rank user passages by BM25.
   *
   * @param tokens   query content tokens, from the same `contentTokens` the
   *                 index was built with
   * @param k        passages to return
   * @param disabled sources switched off in the rail; their passages are not
   *                 searched, exactly as in the dense scan
   */
  search(tokens: Iterable<string>, k: number, disabled: Set<number>): {
    parentIds: Int32Array; scores: Float32Array; n: number;
  } {
    const empty = { parentIds: new Int32Array(0), scores: new Float32Array(0), n: 0 };
    if (!this.liveDocs) return empty;

    const avgLen = this.liveLen / this.liveDocs || 1;
    const scores = this.scores;
    scores.clear();

    for (const term of tokens) {
      const p = this.postings.get(term);
      if (!p) continue;
      const df = p.docs.length;
      const idf = Math.log(1 + (this.liveDocs - df + 0.5) / (df + 0.5));
      for (let i = 0; i < df; i++) {
        const doc = p.docs[i];
        const src = this.source[doc];
        if (src < 0 || disabled.has(src)) continue;
        const f = p.tfs[i];
        const norm = K1 * (1 - B + (B * this.len[doc]) / avgLen);
        scores.set(doc, (scores.get(doc) ?? 0) + (idf * f * (K1 + 1)) / (f + norm));
      }
    }
    if (!scores.size) return empty;

    const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, k);
    const parentIds = new Int32Array(ranked.length);
    const out = new Float32Array(ranked.length);
    // Rescaled against this query's own best hit, for the same reason as the
    // corpus index: BM25 is unbounded, `FusedHit.bestRawScore` takes a max
    // across strategies, and an unbounded number there would dominate a field
    // of cosines in the degraded path where rescoring was skipped.
    const top = ranked[0][1] || 1;
    for (let i = 0; i < ranked.length; i++) {
      parentIds[i] = ranked[i][0];
      out[i] = ranked[i][1] / top;
    }
    return { parentIds, scores: out, n: ranked.length };
  }
}

/** Content tokens with duplicates kept, for term frequency. */
function tokenList(text: string): string[] {
  // `contentTokens` returns a set and applies the stopword rule and the
  // fallback for text that is entirely function words. Counting occurrences of
  // its members in the raw text reproduces the builder's list without a second
  // tokenisation that could drift from it.
  const keep = contentTokens(text);
  const out: string[] = [];
  for (const w of text.toLowerCase().replace(/[^\p{L}\p{N}\p{M}‌‍\s]/gu, " ").split(/\s+/)) {
    if (w.length > 1 && keep.has(w)) out.push(w);
  }
  return out;
}
