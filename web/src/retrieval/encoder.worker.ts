/**
 * The embedding model, on its own thread.
 *
 * Why a worker at all. Embedding is the single most expensive thing in the
 * query path (~7ms of a ~10ms query) and it is also what ingestion spends
 * minutes doing. On one thread those two compete: add a PDF, ask a question
 * while it indexes, and the question queues behind hundreds of ingestion
 * batches. That is a latency failure the per-stage budgets cannot rescue,
 * because the time is spent before the stage even starts.
 *
 * Moving inference here costs ~0.3ms of postMessage per query and buys three
 * things: ingestion never blocks a query, the main thread never janks (so the
 * lamp animates at 60fps while a document indexes), and ONNX's allocations stop
 * landing in the same GC arena as the UI — which is where P100 jitter comes
 * from.
 *
 * Scheduling lives on the main thread (see encoder.ts): it holds the ingestion
 * queue and dispatches one small batch at a time, so a query never waits behind
 * more than a single in-flight batch.
 */

import { pipeline, env } from "@huggingface/transformers";

const MODEL_ID = "Xenova/multilingual-e5-small";

// Match the index build exactly. `model_quantized.onnx` (q8) is what
// pipeline/src/embedder.py used; drift against fp32 was measured at cos=0.996
// with 100% top-10 rank agreement, but only for that pairing.
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
  | { id: number; ok: true; op: "query"; vec: Float32Array }
  | { id: number; ok: true; op: "passages"; vecs: Float32Array; n: number; dim: number }
  | { id: number; ok: false; error: string };

async function init(): Promise<number> {
  if (extractor) return dim;
  // Model weights are fetched from the HF CDN once and served from the browser
  // cache after. Nothing here is on the query path.
  env.allowLocalModels = false;
  extractor = (await pipeline("feature-extraction", MODEL_ID, {
    dtype: DTYPE,
  })) as unknown as ExtractFn;

  // Warm the graph here, not on the first real query. The first inference
  // allocates every buffer and triggers JIT; un-warmed it is 5-20x slower and
  // would single-handedly set P100.
  const out = await extractor("query: warmup", { pooling: "mean", normalize: true });
  dim = out.dims[out.dims.length - 1] ?? 384;
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
      // e5 requires the "query: " prefix — it was contrastively trained with it,
      // and omitting it costs recall silently.
      const out = await extractor!(`query: ${msg.text}`, { pooling: "mean", normalize: true });
      const vec = toF32(out.data);
      reply({ id: msg.id, ok: true, op: "query", vec }, [vec.buffer]);
      return;
    }

    // passages — the ingestion path. Batched by the caller.
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
