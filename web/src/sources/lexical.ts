/**
 * BM25 over the user's own sources.
 *
 * A personal document needs this more than the corpus does. Encyclopedia prose
 * usually survives a paraphrase; a CV, an invoice or a bank statement is proper
 * nouns, dates and amounts, which is what a 256-dim average is worst at and
 * usually the whole of the question. "How old am I" against
 * `name: priya rao, age: 31` is a lexical match or it is nothing.
 *
 * Built at ingest, in one pass over the passage text: a few ms per document
 * against the seconds of embedding that follow. A plain Map is right here,
 * where the corpus version reads a sorted hash dictionary out of a binary - the
 * vocabulary is thousands of terms and changes every time a source is added.
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
      // raw list it was built from - approximated here by counting occurrences
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
   * Drop a source: postings stay, passages are marked dead.
   *
   * Ordinals are permanent index keys, so renumbering would invalidate every
   * surviving chunk's parent. Dead documents cost one comparison at scoring
   * time. df still counts them, which drifts IDF slightly after a delete; a
   * full rebuild per delete is not worth it at a few thousand passages.
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
