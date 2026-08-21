/**
 * The user's own sources: add, index, persist, search, remove.
 *
 * One source runs ingest -> passages -> chunks -> embed -> quantise ->
 * UserIndex. Everything but the last two steps is cheap, so embedding drives
 * the design here:
 *
 *   - off-thread, in small batches, so a query never queues behind it
 *   - cancellable, because a 400-page PDF is minutes of work
 *   - persisted, so it happens exactly once per document
 *
 * Persistence stores the quantised form — binary codes and int8 vectors — not
 * the float embeddings: 32x smaller for the codes, 4x for the passage vectors,
 * and exactly what search needs, so a reload replays straight into the index.
 */

import { UserLexical } from "./lexical";
import { UserIndex, type StrategyBucket } from "./flat";
import { chunkPassages, toPassages, type Passage, type Strategy } from "./chunk";
import { binarise, project, quantiseInt8, STRATEGY_IDS, STRATEGY_NAMES } from "./quantise";
import type { BatchEncoder } from "../retrieval/encoder";
import { ingestFile, ingestPaste, ingestUrl, IngestError, type Extracted, type SourceKind } from "./ingest";

export type SourceStatus = "indexing" | "ready" | "error";

/** Shared empty result, so a query with no user sources allocates nothing. */
const EMPTY: StrategyBucket[] = [];

export interface Source {
  id: number;
  title: string;
  kind: SourceKind;
  note?: string;
  addedAt: number;
  chars: number;
  passages: number;
  chunks: number;
  enabled: boolean;
  status: SourceStatus;
  /** 0..1 while indexing. */
  progress: number;
  error?: string;
}

/** Beyond this a single document costs more than it is worth: ingestion takes
 *  minutes and the flat scan starts eating into the query budget. */
const MAX_CHUNKS_PER_SOURCE = 20_000;

const DB_NAME = "chehrag-sources";
const DB_VERSION = 1;
const STORE = "sources";

interface StoredSource {
  id: number;
  title: string;
  kind: SourceKind;
  note?: string;
  addedAt: number;
  chars: number;
  enabled: boolean;
  /** Guards against replaying vectors built by a different model or geometry. */
  model: string;
  dim: number;
  codeWords: number;
  passageText: string[];
  codes: ArrayBuffer;        // Uint32Array, chunks * codeWords
  chunkParent: ArrayBuffer;  // Int32Array, chunks — LOCAL passage index
  chunkStrategy: ArrayBuffer;// Uint8Array, chunks
  p8: ArrayBuffer;           // Int8Array, passages * dim
  pScale: ArrayBuffer;       // Float32Array, passages
}

export interface StoreDeps {
  encoder: BatchEncoder;
  pcaMean: Float32Array;
  pcaComp: Float32Array;
  dim: number;
  codeWords: number;
  model: string;
  workerBase: string;
}

export class SourceCancelled extends Error {
  constructor() { super("cancelled"); this.name = "SourceCancelled"; }
}

export class SourceStore {
  readonly index: UserIndex;
  /** BM25 over the same passages. See `sources/lexical.ts`. */
  readonly lexical = new UserLexical();
  /** Text of every user passage, indexed by global user-passage ordinal. */
  readonly passages: string[] = [];
  /** Source id per user passage, so a citation can name where it came from. */
  private passageSource: number[] = [];

  private sources = new Map<number, Source>();
  private nextId = 1;
  private db: IDBDatabase | null = null;
  private listeners = new Set<() => void>();
  private cancelled = new Set<number>();
  private projected: Float32Array;

  constructor(private readonly deps: StoreDeps) {
    this.index = new UserIndex(deps.dim, deps.codeWords);
    this.projected = new Float32Array(deps.dim);
  }

  // -- observation ---------------------------------------------------------

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
  private emit(): void { for (const fn of this.listeners) fn(); }

  list(): Source[] {
    return [...this.sources.values()].sort((a, b) => b.addedAt - a.addedAt);
  }
  get activeChunks(): number {
    let n = 0;
    for (const s of this.sources.values()) if (s.enabled && s.status === "ready") n += s.chunks;
    return n;
  }
  get hasEnabled(): boolean {
    for (const s of this.sources.values()) if (s.enabled && s.status === "ready") return true;
    return false;
  }
  sourceOfPassage(userPassageId: number): Source | undefined {
    return this.sources.get(this.passageSource[userPassageId]);
  }

  // -- search --------------------------------------------------------------

  /**
   * Bounded flat search, bucketed by strategy. Returns nothing when no source
   * is enabled, so the corpus-only path pays nothing for this.
   *
   * `tokens` adds the lexical bucket. Optional so a caller with only a vector
   * still gets the six dense strategies rather than an error — but the app
   * always passes them, because on a user's own document the exact-token match
   * is usually the whole question.
   */
  search(q: Float32Array, k: number, tokens?: Iterable<string>): StrategyBucket[] {
    if (!this.hasEnabled) return EMPTY;
    const buckets = this.index.search(q, k);
    if (!tokens) return buckets;
    const lex = this.lexical.search(tokens, k, this.index.disabled);
    if (lex.n === 0) return buckets;
    // Copied rather than pushed onto the index's own reused array: `UserIndex`
    // hands back internal scratch that it resets on the next search.
    return [...buckets, {
      strategy: "lexical" as StrategyBucket["strategy"],
      parentIds: lex.parentIds, scores: lex.scores, n: lex.n,
    }];
  }
  prepareRescore(q: Float32Array): void { this.index.prepare(q); }
  rescore(userPassageId: number): number { return this.index.score(userPassageId); }
  get truncated(): boolean { return this.index.truncated; }

  // -- adding --------------------------------------------------------------

  async addPaste(text: string, title?: string): Promise<Source> {
    return this.ingest("paste", () => Promise.resolve(ingestPaste(text, title)));
  }
  async addFile(file: File): Promise<Source> {
    return this.ingest("file", () => ingestFile(file));
  }
  async addUrl(url: string): Promise<Source> {
    return this.ingest("url", () => ingestUrl(url, this.deps.workerBase));
  }

  /**
   * Run one source end to end. The Source record is published as soon as
   * extraction succeeds, in `indexing` state, so the panel shows progress
   * instead of freezing on an empty list.
   */
  private async ingest(kind: SourceKind, extract: () => Promise<Extracted>): Promise<Source> {
    const ex = await extract();          // throws IngestError, surfaced by the caller
    const id = this.nextId++;
    const src: Source = {
      id, kind, title: ex.title, note: ex.note, addedAt: Date.now(),
      chars: ex.text.length, passages: 0, chunks: 0,
      enabled: true, status: "indexing", progress: 0,
    };
    this.sources.set(id, src);
    this.emit();

    try {
      await this.build(src, ex.text);
      src.status = "ready";
      src.progress = 1;
      this.emit();
      void this.persist(src);
    } catch (err) {
      if (err instanceof SourceCancelled) {
        this.sources.delete(id);
        this.cancelled.delete(id);
        this.emit();
        throw err;
      }
      src.status = "error";
      src.error = err instanceof Error ? err.message : String(err);
      this.emit();
      throw err;
    }
    return src;
  }

  /** Chunk, embed and index one source's text. */
  private async build(src: Source, text: string): Promise<void> {
    const passages: Passage[] = toPassages(text);
    if (!passages.length) throw new IngestError("Nothing to index in that source.");

    let chunks = chunkPassages(passages, src.title);
    if (chunks.length > MAX_CHUNKS_PER_SOURCE) {
      // Trimmed from the end. Chunks are emitted in passage order, so this
      // keeps a contiguous prefix rather than a random sample.
      chunks = chunks.slice(0, MAX_CHUNKS_PER_SOURCE);
      const lastParent = chunks[chunks.length - 1].parent;
      passages.length = lastParent + 1;
      src.note = `first ${passages.length} passages — the rest was too large to index`;
    }

    const { dim, codeWords } = this.deps;
    const nChunks = chunks.length;
    const nPassages = passages.length;

    const codes = new Uint32Array(nChunks * codeWords);
    const chunkParent = new Int32Array(nChunks);
    const chunkStrategy = new Uint8Array(nChunks);
    const p8 = new Int8Array(nPassages * dim);
    const pScale = new Float32Array(nPassages);
    // A passage's vector is its own `whole` chunk, so it is reused rather than
    // embedded a second time for the identical result.
    const wholeSeen = new Uint8Array(nPassages);

    const BATCH = 32;
    for (let start = 0; start < nChunks; start += BATCH) {
      if (this.cancelled.has(src.id)) throw new SourceCancelled();

      const slice = chunks.slice(start, start + BATCH);
      const vecs = await this.deps.encoder.encodePassages(slice.map((c) => c.embedText));

      for (let i = 0; i < slice.length; i++) {
        const c = slice[i];
        const g = start + i;
        const p = project(vecs[i], this.deps.pcaMean, this.deps.pcaComp, dim, this.projected);
        binarise(p, dim, codeWords, codes, g * codeWords);
        chunkParent[g] = c.parent;
        chunkStrategy[g] = STRATEGY_IDS[c.strategy];
        if (c.strategy === "whole" && !wholeSeen[c.parent]) {
          wholeSeen[c.parent] = 1;
          pScale[c.parent] = quantiseInt8(p, dim, p8, c.parent * dim);
        }
      }

      src.progress = Math.min(0.99, (start + slice.length) / nChunks);
      this.emit();
      // Yield so the panel repaints between batches.
      await new Promise<void>((r) => setTimeout(r, 0));
    }

    this.mount(src, passages.map((p) => p.text), codes, chunkParent, chunkStrategy, p8, pScale);
  }

  /**
   * Attach a built source to the live index.
   *
   * Local passage indices become global here. The index keeps one flat passage
   * space across all sources, so every stored `chunkParent` is offset by the
   * base ordinal this source was granted.
   */
  private mount(
    src: Source, passageText: string[],
    codes: Uint32Array, chunkParent: Int32Array, chunkStrategy: Uint8Array,
    p8: Int8Array, pScale: Float32Array,
  ): void {
    const base = this.index.addPassages(p8, pScale);
    for (const t of passageText) {
      this.passages.push(t);
      this.passageSource.push(src.id);
    }
    this.lexical.addPassages(base, passageText, src.id);
    const globalParents = new Int32Array(chunkParent.length);
    for (let i = 0; i < chunkParent.length; i++) globalParents[i] = base + chunkParent[i];
    this.index.addChunks(codes, globalParents, chunkStrategy, src.id, STRATEGY_NAMES);

    src.passages = passageText.length;
    src.chunks = chunkParent.length;
    if (!src.enabled) this.index.disabled.add(src.id);
  }

  // -- mutation ------------------------------------------------------------

  cancel(id: number): void {
    const s = this.sources.get(id);
    if (s?.status === "indexing") this.cancelled.add(id);
  }

  setEnabled(id: number, on: boolean): void {
    const s = this.sources.get(id);
    if (!s) return;
    s.enabled = on;
    if (on) this.index.disabled.delete(id); else this.index.disabled.add(id);
    this.emit();
    void this.persist(s);
  }

  remove(id: number): void {
    const s = this.sources.get(id);
    if (!s) return;
    if (s.status === "indexing") { this.cancel(id); return; }
    this.index.removeSource(id);
    this.lexical.removeSource(id);
    // Passage text stays in place: ordinals are permanent index keys, and
    // renumbering would invalidate every surviving chunk's parent. The orphans
    // are unreachable, since no chunk points at them any more.
    for (let i = 0; i < this.passageSource.length; i++) {
      if (this.passageSource[i] === id) this.passages[i] = "";
    }
    this.sources.delete(id);
    this.emit();
    void this.forget(id);
  }

  // -- persistence ---------------------------------------------------------

  private async openDb(): Promise<IDBDatabase | null> {
    if (this.db) return this.db;
    try {
      this.db = await new Promise<IDBDatabase>((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
          if (!req.result.objectStoreNames.contains(STORE)) {
            req.result.createObjectStore(STORE, { keyPath: "id" });
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
      return this.db;
    } catch {
      return null;   // private mode — sources work, they just don't survive reload
    }
  }

  /** Replay everything stored from previous sessions. Called once at boot. */
  async restore(): Promise<void> {
    const db = await this.openDb();
    if (!db) return;
    let rows: StoredSource[];
    try {
      rows = await new Promise<StoredSource[]>((resolve, reject) => {
        const req = db.transaction(STORE, "readonly").objectStore(STORE).getAll();
        req.onsuccess = () => resolve(req.result as StoredSource[]);
        req.onerror = () => reject(req.error);
      });
    } catch { return; }

    const { dim, codeWords, model } = this.deps;
    for (const row of rows.sort((a, b) => a.addedAt - b.addedAt)) {
      // Vectors from a different model or PCA geometry are not comparable with
      // the shipped index, and keeping them would corrupt ranking invisibly.
      if (row.model !== model || row.dim !== dim || row.codeWords !== codeWords) {
        void this.forget(row.id);
        continue;
      }
      const src: Source = {
        id: row.id, kind: row.kind, title: row.title, note: row.note,
        addedAt: row.addedAt, chars: row.chars, enabled: row.enabled,
        passages: row.passageText.length, chunks: 0,
        status: "ready", progress: 1,
      };
      this.sources.set(row.id, src);
      this.nextId = Math.max(this.nextId, row.id + 1);
      this.mount(
        src, row.passageText,
        new Uint32Array(row.codes), new Int32Array(row.chunkParent),
        new Uint8Array(row.chunkStrategy),
        new Int8Array(row.p8), new Float32Array(row.pScale),
      );
    }
    if (rows.length) this.emit();
  }

  private async persist(src: Source): Promise<void> {
    const db = await this.openDb();
    if (!db) return;
    // Re-derived from the live index rather than held as a second in-memory
    // copy for the lifetime of the session.
    const snap = this.snapshot(src.id);
    if (!snap) return;
    const row: StoredSource = {
      id: src.id, title: src.title, kind: src.kind, note: src.note,
      addedAt: src.addedAt, chars: src.chars, enabled: src.enabled,
      model: this.deps.model, dim: this.deps.dim, codeWords: this.deps.codeWords,
      ...snap,
    };
    try {
      await new Promise<void>((resolve, reject) => {
        const req = db.transaction(STORE, "readwrite").objectStore(STORE).put(row);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    } catch { /* quota exceeded — the source still works this session */ }
  }

  private async forget(id: number): Promise<void> {
    const db = await this.openDb();
    if (!db) return;
    try {
      await new Promise<void>((resolve, reject) => {
        const req = db.transaction(STORE, "readwrite").objectStore(STORE).delete(id);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    } catch { /* nothing to do */ }
  }

  /** Extract one source's arrays back out of the shared index for storage. */
  private snapshot(id: number): Pick<StoredSource,
    "passageText" | "codes" | "chunkParent" | "chunkStrategy" | "p8" | "pScale"> | null {
    const src = this.sources.get(id);
    if (!src) return null;
    const { dim, codeWords } = this.deps;

    const localOf = new Map<number, number>();
    const passageText: string[] = [];
    const p8 = new Int8Array(src.passages * dim);
    const pScale = new Float32Array(src.passages);
    for (let g = 0; g < this.passageSource.length; g++) {
      if (this.passageSource[g] !== id) continue;
      const local = passageText.length;
      localOf.set(g, local);
      passageText.push(this.passages[g]);
      this.index.copyPassage(g, p8, local * dim);
      pScale[local] = this.index.scaleOf(g);
    }

    const refs = this.index.refsOf(id);
    const codes = new Uint32Array(refs.length * codeWords);
    const chunkParent = new Int32Array(refs.length);
    const chunkStrategy = new Uint8Array(refs.length);
    refs.forEach((r, i) => {
      this.index.copyCode(r.slot, codes, i * codeWords);
      chunkParent[i] = localOf.get(r.passage) ?? 0;
      chunkStrategy[i] = STRATEGY_IDS[r.strategy as Strategy];
    });

    return {
      passageText,
      codes: codes.buffer, chunkParent: chunkParent.buffer,
      chunkStrategy: chunkStrategy.buffer,
      p8: p8.buffer, pScale: pScale.buffer,
    };
  }

  /** Drop everything, including from storage. */
  async clear(): Promise<void> {
    this.index.clear();
    this.lexical.clear();
    this.passages.length = 0;
    this.passageSource.length = 0;
    const ids = [...this.sources.keys()];
    this.sources.clear();
    this.emit();
    for (const id of ids) await this.forget(id);
  }
}
