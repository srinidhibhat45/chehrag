/**
 * Client-side chunking for user-added sources.
 *
 * A TypeScript port of `pipeline/src/chunking/strategies.py`, kept
 * strategy-for-strategy identical. User sources fuse with the shipped corpus in
 * the same RRF pass, and RRF compares ranks across strategies - so a user chunk
 * produced by a different rule than a corpus chunk with the same strategy label
 * would have the fusion weights applied to two different things.
 *
 * One substantive difference. MS MARCO passages arrive pre-cut at ~317
 * characters, already chunk-sized, which is why the pipeline's `whole` strategy
 * is a no-op splitter. A user's PDF is not, so this file adds a passage-forming
 * step in front: text is cut into passage-sized units, and the six strategies
 * then run over those units exactly as they do over MS MARCO passages.
 */

/** Sentence terminators: Devanagari danda plus ASCII. Matches `_SENT_RE`. */
const SENT_SPLIT = /(?<=[।.!?])\s+/;
const NUMERIC = /\d[\d,.\-/%]*/g;
/** Devanagari is caseless, so Latin runs inside Indic text carry the proper
 *  nouns, units and acronyms that dense vectors handle worst. */
const LATIN_RUN = /[A-Za-z][A-Za-z.\-]{2,}/g;

export type Strategy = "whole" | "sentence" | "sliding" | "contextual" | "semantic" | "document";

export interface UserChunk {
  strategy: Strategy;
  /** What the embedder sees - may carry a context header. */
  embedText: string;
  /** Index of the parent passage within this source. */
  parent: number;
}

export interface Passage {
  text: string;
  /** Section heading in force at this point, if the document had one. */
  heading?: string;
}

/** Mirrors the corpus statistics the pipeline measured: mean 317 chars. */
const TARGET_PASSAGE_CHARS = 340;
const MAX_PASSAGE_CHARS = 700;

export function splitSentences(text: string): string[] {
  return text.split(SENT_SPLIT).map((s) => s.trim()).filter(Boolean);
}

/** Markdown ATX/setext headings, and the ALL-CAPS/short-line shape that PDFs and
 *  Word documents use instead. */
function headingOf(line: string): string | null {
  const t = line.trim();
  if (!t) return null;
  const md = /^#{1,6}\s+(.{1,90})$/.exec(t);
  if (md) return md[1].trim();
  if (t.length <= 80 && !/[.!?।]$/.test(t)) {
    const letters = t.replace(/[^\p{L}]/gu, "");
    if (letters.length >= 3 && letters === letters.toUpperCase() && /[A-Z]/.test(letters)) return t;
    if (/^\d+(\.\d+)*\s+\S/.test(t)) return t;   // "3.2 Results"
  }
  return null;
}

/**
 * Cut raw document text into passage-sized units.
 *
 * Paragraph boundaries come first - they are semantic joints the author put
 * there. Only paragraphs that overshoot are cut on sentence boundaries, and only
 * sentences that overshoot are cut on whitespace. The ladder keeps a well-formed
 * document from being cut mid-thought while still terminating on a pathological
 * one (a 50 kB single paragraph, a minified file).
 */
export function toPassages(text: string): Passage[] {
  const out: Passage[] = [];
  let heading: string | undefined;
  let buf = "";

  const flush = () => {
    const t = buf.trim();
    if (t) out.push({ text: t, heading });
    buf = "";
  };

  for (const block of text.split(/\n{2,}/)) {
    const lines = block.split("\n");
    // A lone line that reads as a heading retitles what follows rather than
    // becoming a four-word passage of its own.
    if (lines.length === 1) {
      const h = headingOf(lines[0]);
      if (h) { flush(); heading = h; continue; }
    }
    const para = block.trim();
    if (!para) continue;

    if (buf.length + para.length + 1 <= TARGET_PASSAGE_CHARS) {
      buf = buf ? `${buf}\n${para}` : para;
      continue;
    }
    flush();

    if (para.length <= MAX_PASSAGE_CHARS) { out.push({ text: para, heading }); continue; }

    for (const sent of splitSentences(para)) {
      for (const piece of hardWrap(sent, MAX_PASSAGE_CHARS)) {
        if (buf.length + piece.length + 1 > TARGET_PASSAGE_CHARS) flush();
        buf = buf ? `${buf} ${piece}` : piece;
      }
    }
    flush();
  }
  flush();
  return out;
}

/** Last resort for text with no sentence boundaries at all (logs, CSV rows,
 *  minified code). Breaks on whitespace when it can, mid-token when it can't. */
function hardWrap(s: string, limit: number): string[] {
  if (s.length <= limit) return [s];
  const parts: string[] = [];
  let i = 0;
  while (i < s.length) {
    let end = Math.min(i + limit, s.length);
    if (end < s.length) {
      const sp = s.lastIndexOf(" ", end);
      if (sp > i + limit * 0.6) end = sp;
    }
    parts.push(s.slice(i, end).trim());
    i = end;
  }
  return parts.filter(Boolean);
}

// -- the six strategies -----------------------------------------------------

const SENTENCE_MIN_CHARS = 45;
const SLIDING_WINDOW = 2;
const SLIDING_STRIDE = 1;

/**
 * Produce every chunk for one source.
 *
 * Returns chunks tagged with the strategy that made them and the passage they
 * came from. Passage ordinals are local to the source; the store maps them into
 * global ids when the chunks are registered.
 */
export function chunkPassages(passages: Passage[], title: string): UserChunk[] {
  const chunks: UserChunk[] = [];
  const push = (strategy: Strategy, embedText: string, parent: number) => {
    const t = embedText.trim();
    if (t.length >= 12) chunks.push({ strategy, embedText: t, parent });
  };

  passages.forEach((p, i) => {
    const text = p.text;

    // 1. whole - the passage as-is. Strongest single index on this corpus shape.
    push("whole", text, i);

    const sents = splitSentences(text);

    // 2. sentence - precision. A one-sentence answer inside a long passage is
    //    otherwise averaged away by the surrounding text.
    if (sents.length > 1) {
      for (const s of sents) if (s.length >= SENTENCE_MIN_CHARS) push("sentence", s, i);
    }

    // 3. sliding - overlapping sentence windows. Catches answers that straddle a
    //    boundary, and survives the bad punctuation that machine-translated and
    //    PDF-extracted text is full of.
    if (sents.length > SLIDING_WINDOW) {
      for (let s = 0; s + SLIDING_WINDOW <= sents.length; s += SLIDING_STRIDE) {
        push("sliding", sents.slice(s, s + SLIDING_WINDOW).join(" "), i);
      }
    }

    // 4. contextual - same span, richer embedding text. Numerics and Latin-script
    //    tokens are restated in a header because dense vectors handle them worst.
    //    For a user document the section heading and title go in too, restoring
    //    the anaphora a bare passage lost ("it doubled in Q3" - what did?).
    const nums = uniq(text.match(NUMERIC) ?? []).slice(0, 8);
    const latin = uniq(text.match(LATIN_RUN) ?? []).slice(0, 8);
    const bits: string[] = [title];
    if (p.heading) bits.push(p.heading);
    if (latin.length) bits.push(latin.join(" "));
    if (nums.length) bits.push(nums.join(" "));
    push("contextual", `${bits.join(" | ")}\n${text}`, i);

    // 5. semantic - group adjacent sentences that share vocabulary, so a complete
    //    idea stays in one vector. Low yield on ~3-sentence passages; kept for
    //    parity and for the longer passages user documents contain.
    if (sents.length > 2) {
      for (const g of semanticGroups(sents)) if (g.length > 1) push("semantic", g.join(" "), i);
    }
  });

  // 6. document - neighbouring passages merged. A recall instrument: dilute
  //    vectors, but the only strategy that can find an answer spread across
  //    consecutive paragraphs.
  for (let i = 0; i < passages.length; i += 3) {
    const group = passages.slice(i, i + 3);
    if (group.length > 1) push("document", group.map((g) => g.text).join(" "), i);
  }

  return chunks;
}

function uniq(xs: string[]): string[] {
  return [...new Set(xs.map((x) => x.trim()).filter(Boolean))];
}

/**
 * Split a sentence list where consecutive sentences stop sharing vocabulary.
 *
 * Jaccard over content tokens rather than embeddings: an embedding-based split
 * would cost one forward pass per sentence before the chunks are even known,
 * doubling the expensive half of ingestion for a gain the pipeline measured as
 * small.
 */
function semanticGroups(sents: string[], threshold = 0.12): string[][] {
  const toks = sents.map((s) => new Set(
    // `\p{M}` keeps Indic matras attached to their base letter; see tokens.ts.
    s.toLowerCase().replace(/[^\p{L}\p{N}\p{M}\s]/gu, " ").split(/\s+/).filter((w) => w.length > 2),
  ));
  const groups: string[][] = [];
  let cur = [sents[0]];
  for (let i = 1; i < sents.length; i++) {
    if (jaccard(toks[i - 1], toks[i]) >= threshold) cur.push(sents[i]);
    else { groups.push(cur); cur = [sents[i]]; }
  }
  groups.push(cur);
  return groups;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter);
}
