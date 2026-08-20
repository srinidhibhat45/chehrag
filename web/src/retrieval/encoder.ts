/**
 * Query encoder — transformers.js over the same ONNX weights the index was built
 * with (Xenova/multilingual-e5-small, int8).
 *
 * Two implementations behind one interface:
 *
 *   E5Encoder      in-thread. Used by the Node benchmark, and as the browser
 *                  fallback when module workers are unavailable.
 *   WorkerEncoder  off-thread (encoder.worker.ts). What the app uses.
 *
 * Both run the identical graph with identical prefixes, so the benchmark
 * exercises the deployed code path.
 *
 * Two details are easy to get wrong:
 *   - dtype 'q8' must match the build. The Python builder used
 *     model_quantized.onnx; drift against fp32 measured cos=0.996 with 100%
 *     top-10 rank agreement, but only for that pairing.
 *   - e5 requires the "query: " / "passage: " prefixes. It was contrastively
 *     trained with them, and omitting them costs recall silently.
 */

import type { Encoder } from "../harness/rag";
import type { Req, Res } from "./encoder.worker";

const MODEL_ID = "Xenova/multilingual-e5-small";

/**
 * transformers.js types the pipeline factory as a union over every task, which
 * TypeScript cannot represent at this call site (TS2590). Narrowed to the
 * feature-extraction shape, which is the only one used here.
 */
type ExtractFn = (
  text: string | string[],
  opts: { pooling: "mean"; normalize: boolean },
) => Promise<{ data: Float32Array | ArrayLike<number>; dims: number[] }>;

/**
 * Cumulative bytes of the model download. Called only while the encoder is first
 * loading and only for files actually coming over the network — a warm cache
 * reports nothing, which the caller must read as "unknown", not "0%".
 */
export type EncoderProgress = (loaded: number, total: number) => void;

/** What ingestion needs on top of the query interface. */
export interface BatchEncoder extends Encoder {
  /** Embed documents with the "passage: " prefix. Returns one row per text. */
  encodePassages(texts: string[]): Promise<Float32Array[]>;
  /** Stop dispatching ingestion work; queries are unaffected. */
  pauseBulk(): void;
  resumeBulk(): void;
}

// ---------------------------------------------------------------------------
// in-thread
// ---------------------------------------------------------------------------

export class E5Encoder implements BatchEncoder {
  private extractor: ExtractFn | null = null;

  async init(onProgress?: EncoderProgress): Promise<void> {
    if (this.extractor) return;
    // Dynamic for bundle reasons: a static import pulls ~870 kB of
    // transformers.js into the main chunk to have a fallback ready, on top of
    // the copy already in the worker chunk that actually runs.
    const { pipeline } = await import("@huggingface/transformers");
    this.extractor = (await pipeline(
      "feature-extraction", MODEL_ID, {
        dtype: "q8",
        progress_callback: onProgress
          ? (i) => { if (i.status === "progress") onProgress(i.loaded, i.total); }
          : undefined,
      },
    )) as unknown as ExtractFn;
    // The first inference allocates buffers and triggers JIT; paying it here
    // keeps it out of the first measured query.
    await this.encodeQuery("warmup");
  }

  async encodeQuery(text: string): Promise<Float32Array> {
    if (!this.extractor) throw new Error("encoder not initialised — call init() first");
    const out = await this.extractor(`query: ${text}`, { pooling: "mean", normalize: true });
    return out.data instanceof Float32Array ? out.data : new Float32Array(out.data as ArrayLike<number>);
  }

  async encodePassages(texts: string[]): Promise<Float32Array[]> {
    if (!this.extractor) throw new Error("encoder not initialised — call init() first");
    if (!texts.length) return [];
    const out = await this.extractor(texts.map((t) => `passage: ${t}`),
      { pooling: "mean", normalize: true });
    const flat = out.data instanceof Float32Array
      ? out.data : new Float32Array(out.data as ArrayLike<number>);
    const d = flat.length / texts.length;
    return texts.map((_, i) => flat.subarray(i * d, (i + 1) * d));
  }

  pauseBulk(): void { /* nothing to schedule in-thread */ }
  resumeBulk(): void { /* nothing to schedule in-thread */ }
}

// ---------------------------------------------------------------------------
// off-thread
// ---------------------------------------------------------------------------

interface Pending {
  resolve: (r: Res) => void;
  reject: (e: Error) => void;
}

/**
 * `Omit` collapses a union into one object type, which would allow `{op:"query"}`
 * to carry a `texts` field. Distributing over the members keeps each variant's
 * own shape intact.
 */
type Message = Req extends infer T
  ? T extends { id: number } ? Omit<T, "id"> : never
  : never;

/**
 * Worker-backed encoder with query priority.
 *
 * The worker handles messages serially, so priority is enforced on this side:
 * ingestion batches queue here and dispatch one at a time, and dispatch stops
 * while a query is in flight. A query therefore waits for at most one running
 * batch, which is why `BULK_BATCH` is small — at 8 passages a batch costs ~25ms,
 * bounding what a query can inherit from ingestion.
 */
const BULK_BATCH = 8;

export class WorkerEncoder implements BatchEncoder {
  private worker: Worker | null = null;
  private pending = new Map<number, Pending>();
  private nextId = 1;

  private bulkQueue: Array<{
    texts: string[];
    resolve: (v: Float32Array[]) => void;
    reject: (e: Error) => void;
  }> = [];
  private bulkInFlight = false;
  private bulkPaused = false;
  private queriesInFlight = 0;

  /** Set when construction or init fails; the caller falls back in-thread. */
  private failed = false;

  async init(onProgress?: EncoderProgress): Promise<void> {
    if (this.worker) return;
    this.worker = new Worker(new URL("./encoder.worker.ts", import.meta.url), {
      type: "module",
      name: "chehrag-encoder",
    });
    this.worker.onmessage = (e: MessageEvent<Res>) => {
      // Routed on `op` first: progress answers no request, so a `pending`
      // lookup would drop it silently.
      if (e.data.ok && e.data.op === "progress") {
        onProgress?.(e.data.loaded, e.data.total);
        return;
      }
      const p = this.pending.get(e.data.id);
      if (!p) return;
      this.pending.delete(e.data.id);
      p.resolve(e.data);
    };
    this.worker.onerror = (e) => {
      this.failed = true;
      const err = new Error(`encoder worker failed: ${e.message}`);
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
    await this.send({ op: "init" });
  }

  get usable(): boolean { return !!this.worker && !this.failed; }

  private send(req: Message): Promise<Res> {
    return new Promise((resolve, reject) => {
      if (!this.worker || this.failed) { reject(new Error("encoder worker unavailable")); return; }
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...req, id } as Req);
    });
  }

  async encodeQuery(text: string): Promise<Float32Array> {
    this.queriesInFlight++;
    try {
      const res = await this.send({ op: "query", text });
      if (!res.ok) throw new Error(res.error);
      if (res.op !== "query") throw new Error("unexpected worker reply");
      return res.vec;
    } finally {
      this.queriesInFlight--;
      this.pumpBulk();
    }
  }

  encodePassages(texts: string[]): Promise<Float32Array[]> {
    if (!texts.length) return Promise.resolve([]);
    // Split up front so the queue is already at the granularity dispatch needs.
    const batches: Promise<Float32Array[]>[] = [];
    for (let i = 0; i < texts.length; i += BULK_BATCH) {
      const slice = texts.slice(i, i + BULK_BATCH);
      batches.push(new Promise((resolve, reject) => {
        this.bulkQueue.push({ texts: slice, resolve, reject });
      }));
    }
    this.pumpBulk();
    return Promise.all(batches).then((parts) => parts.flat());
  }

  pauseBulk(): void { this.bulkPaused = true; }
  resumeBulk(): void { this.bulkPaused = false; this.pumpBulk(); }

  private pumpBulk(): void {
    if (this.bulkInFlight || this.bulkPaused) return;
    if (this.queriesInFlight > 0) return;    // a query owns the worker
    const job = this.bulkQueue.shift();
    if (!job) return;

    this.bulkInFlight = true;
    this.send({ op: "passages", texts: job.texts })
      .then((res) => {
        if (!res.ok) throw new Error(res.error);
        if (res.op !== "passages") throw new Error("unexpected worker reply");
        const { vecs, n, dim } = res;
        job.resolve(Array.from({ length: n }, (_, i) => vecs.subarray(i * dim, (i + 1) * dim)));
      })
      .catch((e) => job.reject(e instanceof Error ? e : new Error(String(e))))
      .finally(() => { this.bulkInFlight = false; this.pumpBulk(); });
  }

  terminate(): void {
    this.worker?.terminate();
    this.worker = null;
  }
}

/**
 * Build the best encoder this environment supports.
 *
 * Worker construction can fail for external reasons — a restrictive CSP, an old
 * browser, a privacy extension — so this falls back in-thread rather than
 * failing outright.
 */
export async function createEncoder(onProgress?: EncoderProgress): Promise<BatchEncoder> {
  if (typeof Worker !== "undefined") {
    const w = new WorkerEncoder();
    try {
      await w.init(onProgress);
      if (w.usable) return w;
    } catch (err) {
      console.warn("encoder worker unavailable, running in-thread", err);
      w.terminate();
    }
  }
  const e = new E5Encoder();
  await e.init(onProgress);
  return e;
}
