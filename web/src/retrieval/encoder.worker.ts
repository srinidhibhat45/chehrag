/**
 * The embedding model, on its own thread.
 *
 * Embedding is the most expensive thing in the query path and also what
 * ingestion spends minutes doing. Share a thread and a question asked during a
 * PDF import queues behind hundreds of ingestion batches, which no per-stage
 * budget can rescue: the time is spent before the stage starts.
 *
 * Costs ~0.3ms of postMessage per query. Buys three things: ingestion never
 * blocks a query, the main thread never janks, and ONNX stops allocating in the
 * same GC arena as the UI, which is where P100 jitter came from.
 *
 * Scheduling stays on the main thread (`encoder.ts`).
 */

import { pipeline, env } from "@huggingface/transformers";

const MODEL_ID = "Xenova/multilingual-e5-small";

// Must match the index build. `pipeline/src/embedder.py` used
// `model_quantized.onnx` (q8); drift against fp32 measured cos=0.996 with 100%
// top-10 rank agreement, but only for that pairing.
const DTYPE = "q8";

type ExtractFn = (
  text: string | string[],
  opts: { pooling: "mean"; normalize: boolean },
) => Promise<{ data: Float32Array | ArrayLike<number>; dims: number[] }>;

let extractor: ExtractFn | null = null;
let dim = 384;

export type Req =
  | { id: number; op: "init" }
  | { id: number; op: "query"; text: string }
  | { id: number; op: "passages"; texts: string[] };

export type Res =
  | { id: number; ok: true; op: "init"; dim: number }
  /**
   * Unsolicited: model download progress, sent while `init` is still pending.
   * `id` is 0 because it answers no request, and the main thread routes on `op`
   * before looking a request up.
   */
  | { id: 0; ok: true; op: "progress"; loaded: number; total: number }
  | { id: number; ok: true; op: "query"; vec: Float32Array }
  | { id: number; ok: true; op: "passages"; vecs: Float32Array; n: number; dim: number }
  | { id: number; ok: false; error: string };

/**
 * Cumulative download progress across the model's files.
 *
 * transformers.js reports per file - config, tokenizer, ONNX weights - so the
 * per-file numbers are held and totalled rather than forwarded one at a time.
 *
 * Files already in the browser cache emit no `progress` at all, which is why
 * `total` can stay 0 through a warm boot. Nothing is sent in that case, and the
 * curtain leaves its row alone rather than showing 0%.
 */
const files = new Map<string, { loaded: number; total: number }>();
let lastPost = 0;

function onModelProgress(info: { status: string; file?: string; loaded?: number; total?: number }): void {
  if (info.status !== "progress" || !info.file) return;
  files.set(info.file, { loaded: info.loaded ?? 0, total: info.total ?? 0 });

  // Throttled: this fires per chunk per file, and one postMessage per chunk
  // would put the download's own bookkeeping on the main thread's queue.
  const now = performance.now();
  if (now - lastPost < 100) return;
  lastPost = now;

  let loaded = 0, total = 0;
  for (const f of files.values()) { loaded += f.loaded; total += f.total; }
  if (total > 0) reply({ id: 0, ok: true, op: "progress", loaded, total });
}

async function init(): Promise<number> {
  if (extractor) return dim;
  // Weights come from the HF CDN once, then from the browser cache. Nothing
  // here is on the query path.
  env.allowLocalModels = false;

  // onnxruntime-web runs the graph on a single WASM thread unless told
  // otherwise, and the forward pass is the dominant cost in the query budget.
  // Threads need SharedArrayBuffer, which needs cross-origin isolation, which
  // `public/_headers` sets - so this is where that header actually pays.
  //
  // Measured in-browser over 36 corpus queries: 1 thread embeds in 51 ms,
  // 4 in 16.3, 8 in 15.2. Four is where the curve flattens, and the page still
  // has a UI and a shader that want cores.
  const wasm = env.backends?.onnx?.wasm;
  if (wasm) {
    const cores = self.navigator?.hardwareConcurrency ?? 4;
    wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, cores - 1)) : 1;
  }
  extractor = (await pipeline("feature-extraction", MODEL_ID, {
    dtype: DTYPE,
    progress_callback: onModelProgress,
  })) as unknown as ExtractFn;

  // Warm the graph here rather than on the first real query. The first inference
  // allocates every buffer and triggers JIT, running 5-20x slower.
  const out = await extractor("query: warmup", { pooling: "mean", normalize: true });
  dim = out.dims[out.dims.length - 1] ?? 384;
  console.info(`[chehrag] encoder ready - wasm threads ${env.backends?.onnx?.wasm?.numThreads}, ` +
               `isolated ${self.crossOriginIsolated}`);
  return dim;
}

self.onmessage = async (e: MessageEvent<Req>) => {
  const msg = e.data;
  try {
    if (msg.op === "init") {
      const d = await init();
      reply({ id: msg.id, ok: true, op: "init", dim: d });
      return;
    }
    if (!extractor) await init();

    if (msg.op === "query") {
      // e5 was contrastively trained with the "query: " prefix; omitting it
      // costs recall silently.
      const out = await extractor!(`query: ${msg.text}`, { pooling: "mean", normalize: true });
      const vec = toF32(out.data);
      reply({ id: msg.id, ok: true, op: "query", vec }, [vec.buffer]);
      return;
    }

    // The ingestion path. Batched by the caller.
    const texts = msg.texts.map((t) => `passage: ${t}`);
    const out = await extractor!(texts, { pooling: "mean", normalize: true });
    const vecs = toF32(out.data);
    const n = msg.texts.length;
    reply({ id: msg.id, ok: true, op: "passages", vecs, n, dim: vecs.length / n },
      [vecs.buffer]);
  } catch (err) {
    reply({ id: msg.id, ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};

function toF32(d: Float32Array | ArrayLike<number>): Float32Array {
  // Transfer requires ownership of the buffer. A tensor's data can be a view
  // over a larger arena, so copy unless it is exactly its own buffer.
  if (d instanceof Float32Array && d.byteOffset === 0 && d.buffer.byteLength === d.byteLength) {
    return d;
  }
  return new Float32Array(d as ArrayLike<number>);
}

function reply(r: Res, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(r, transfer);
}
