"""Embed every chunk of every strategy to fp32, resumably.

Runtime is ~90 min, so this checkpoints per strategy: a crash or Ctrl-C resumes
from the last completed strategy rather than restarting.

Throughput notes (measured on M3 Pro, 5P/6E):
  int8 weights, unsorted batches  ->  97 texts/s
  int8 weights, length-sorted     -> 157 texts/s   <- 1.6x, free
  fp32 weights, length-sorted     ->  69 texts/s
  CoreML EP                       -> slower; graph splits into 74 partitions

Length-sorted batching matters because the tokenizer pads each batch to its
longest member. Mixing a 20-token sentence with a 190-token document means the
short one is 90% padding. Sorting first makes batches near-uniform.

Quantized weights are used for the build because the drift check showed
cos=0.996 vs fp32 with 100% top-10 rank agreement -- the ranking is unchanged,
so fp32 build time buys nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import orjson
from tokenizers import Tokenizer

MODEL_DIR = Path("data/model")
MAX_LEN = 192
DIM = 384

STRATEGIES = ["whole", "sentence", "sliding", "contextual", "semantic", "document"]


def build_session(quantized: bool = True) -> ort.InferenceSession:
    name = "model_quantized.onnx" if quantized else "model.onnx"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(MODEL_DIR / "onnx" / name), so,
                                providers=["CPUExecutionProvider"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks")
    ap.add_argument("--out", default="data/vectors")
    ap.add_argument("--lang", default="hin")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sess = build_session()
    needs_tt = "token_type_ids" in {i.name for i in sess.get_inputs()}
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(MAX_LEN)

    grand_t = time.perf_counter()
    grand_n = 0

    for strat in STRATEGIES:
        src = Path(args.chunks) / f"chunks.{args.lang}.{strat}.jsonl"
        vec_path = out / f"vec.{args.lang}.{strat}.f32.npy"
        meta_path = out / f"meta.{args.lang}.{strat}.jsonl"

        if not src.exists():
            print(f"[{strat}] missing {src}, skipping")
            continue
        if vec_path.exists() and meta_path.exists():
            n = np.load(vec_path, mmap_mode="r").shape[0]
            print(f"[{strat}] already done ({n:,} vectors) — skipping")
            grand_n += n
            continue

        rows = []
        with src.open("rb") as f:
            for line in f:
                r = orjson.loads(line)
                rows.append((r["chunk_id"], r["parent_id"], r["embed_text"],
                             r["span"], r.get("meta") or {}))
        n = len(rows)
        print(f"[{strat}] {n:,} chunks", flush=True)

        # e5 requires the passage prefix; chunks are all passage-side.
        texts = [f"passage: {r[2]}" for r in rows]
        order = sorted(range(n), key=lambda i: len(texts[i]))

        vecs = np.lib.format.open_memmap(
            vec_path, mode="w+", dtype=np.float32, shape=(n, DIM))

        t0 = time.perf_counter()
        done = 0
        for s in range(0, n, args.batch):
            idx = order[s:s + args.batch]
            tok.enable_padding(length=None)
            enc = tok.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if needs_tt:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = sess.run(None, feed)[0]

            m = mask[..., None].astype(np.float32)
            v = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
            # scatter back to original chunk order
            vecs[idx] = v.astype(np.float32)

            done += len(idx)
            if s % (args.batch * 200) == 0 and done:
                el = time.perf_counter() - t0
                rate = done / el
                print(f"[{strat}] {done:,}/{n:,}  {rate:.0f}/s  "
                      f"eta {(n - done) / rate / 60:.1f} min", flush=True)

        vecs.flush()
        del vecs

        with meta_path.open("wb") as f:
            for cid, pid, _txt, span, meta in rows:
                f.write(orjson.dumps({
                    "chunk_id": cid, "parent_id": pid,
                    "span": span, "meta": meta,
                }) + b"\n")

        el = time.perf_counter() - t0
        grand_n += n
        print(f"[{strat}] DONE {n:,} in {el/60:.1f} min ({n/el:.0f}/s)", flush=True)

    tot = time.perf_counter() - grand_t
    print(f"\nALL DONE: {grand_n:,} vectors in {tot/60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
