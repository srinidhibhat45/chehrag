"""Turn fp32 chunk vectors into the browser's shippable index.

Pipeline:
  1. fit ONE shared PCA 384 -> 192 across all strategies
  2. project + renormalise every chunk vector
  3. binary-quantise chunk vectors (sign bits) -> 192 bits = 24 B each
  4. k-means IVF per strategy so search work per query is bounded
  5. int8-quantise PASSAGE vectors only, for final rescoring
  6. emit flat binary blobs + a JSON manifest

One shared PCA rather than one per strategy: the query is projected once and
compared against all six indices, so six projections would put each index in a
different vector space and leave the query valid in only one of them. That is a
silent correctness bug, not a quality tradeoff.

PCA is not compression here, it is what makes binary quantisation work. Measured
R@10 on 500 gold-labelled queries, brute force over 98,867 passages:

    fp32 384      (ceiling)        0.7580
    PCA-256 fp32  (var 0.90)       0.7440
    -> binary-256                  0.6040
    binary-384, NO PCA             0.2280

Sign-bit quantisation of raw embedding dimensions is near-worthless because they
are correlated and not zero-centred; PCA decorrelates and centres them, which is
what makes the sign bit informative.

Dim: 192 retained only 0.80 variance (R@10 0.718). 256 retains 0.90 (R@10 0.744,
within 1.4 points of the fp32 ceiling) and packs into exactly 8 uint32 words.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import orjson

STRATEGIES = ["whole", "sentence", "sliding", "contextual", "semantic", "document"]
OUT_DIM = 256
SEED = 20260819


def fit_pca(sample: np.ndarray, out_dim: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (mean, components[out_dim, in_dim], retained_variance_ratio)."""
    mean = sample.mean(axis=0)
    x = sample - mean
    # economy SVD on the sample; n_sample >> in_dim so this is cheap
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    retained = float(var[:out_dim].sum() / var.sum())
    return mean.astype(np.float32), vt[:out_dim].astype(np.float32), retained


def project(vecs: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """Project and re-L2-normalise. PCA does not preserve unit norm."""
    y = (vecs - mean) @ comp.T
    n = np.linalg.norm(y, axis=1, keepdims=True)
    return (y / np.clip(n, 1e-9, None)).astype(np.float32)


def binarise(v: np.ndarray) -> np.ndarray:
    """Sign bits packed little-endian into uint32 words (matches ivf.ts)."""
    n, d = v.shape
    words = d // 32
    bits = (v > 0)
    out = np.zeros((n, words), dtype=np.uint32)
    for w in range(words):
        chunk = bits[:, w * 32:(w + 1) * 32]
        weights = (np.uint32(1) << np.arange(32, dtype=np.uint32))
        out[:, w] = (chunk * weights).sum(axis=1, dtype=np.uint32)
    return out


def minibatch_kmeans(x: np.ndarray, k: int, iters: int = 40,
                     batch: int = 16384, rng=None) -> np.ndarray:
    """Minibatch k-means on unit vectors (cosine == dot product here)."""
    rng = rng or np.random.default_rng(SEED)
    n = x.shape[0]
    idx = rng.choice(n, size=min(k, n), replace=False)
    cent = x[idx].copy()
    if cent.shape[0] < k:                       # tiny strategy: pad by duplication
        pad = rng.choice(cent.shape[0], k - cent.shape[0])
        cent = np.vstack([cent, cent[pad]])
    counts = np.zeros(k, dtype=np.int64)
    for _ in range(iters):
        b = x[rng.choice(n, size=min(batch, n), replace=False)]
        assign = (b @ cent.T).argmax(axis=1)
        for c in np.unique(assign):
            m = b[assign == c]
            counts[c] += m.shape[0]
            lr = m.shape[0] / counts[c]
            cent[c] += lr * (m.mean(axis=0) - cent[c])
        norm = np.linalg.norm(cent, axis=1, keepdims=True)
        cent /= np.clip(norm, 1e-9, None)
    return cent.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="data/vectors")
    ap.add_argument("--subset", default="data/subset")
    ap.add_argument("--out", default="../web/public/index")
    ap.add_argument("--lang", default="hin")
    ap.add_argument("--pca-sample", type=int, default=120_000)
    args = ap.parse_args()

    vdir = Path(args.vectors)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # passage ordinals: the browser refers to passages by index, not hash
    passages = []
    with (Path(args.subset) / f"corpus.{args.lang}.jsonl").open("rb") as f:
        for line in f:
            passages.append(orjson.loads(line))
    pid_to_ord = {p["id"]: i for i, p in enumerate(passages)}
    n_pass = len(passages)
    print(f"passages: {n_pass:,}")

    present = [s for s in STRATEGIES
               if (vdir / f"vec.{args.lang}.{s}.f32.npy").exists()]
    missing = [s for s in STRATEGIES if s not in present]
    if missing:
        print(f"WARNING missing vectors for: {missing}")
    print(f"strategies: {present}")

    # ---- 1. shared PCA ------------------------------------------------------
    t0 = time.perf_counter()
    per = max(1, args.pca_sample // len(present))
    parts = []
    for s in present:
        v = np.load(vdir / f"vec.{args.lang}.{s}.f32.npy", mmap_mode="r")
        take = min(per, v.shape[0])
        sel = np.sort(rng.choice(v.shape[0], size=take, replace=False))
        parts.append(np.asarray(v[sel], dtype=np.float32))
    sample = np.vstack(parts)
    del parts
    mean, comp, retained = fit_pca(sample, OUT_DIM)
    print(f"PCA 384->{OUT_DIM} on {sample.shape[0]:,} vectors | "
          f"retained variance {retained:.4f} | {time.perf_counter()-t0:.1f}s")
    if retained < 0.90:
        print("  !! retained variance below 0.90 — reduction is costing recall")
    del sample

    manifest = {
        "lang": args.lang,
        "dim": OUT_DIM,
        "codeWords": OUT_DIM // 32,
        "numPassages": n_pass,
        "strategies": {},
        "model": "Xenova/multilingual-e5-small",
        "builtAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (out / "pca_mean.f32.bin").write_bytes(mean.tobytes())
    (out / "pca_comp.f32.bin").write_bytes(comp.tobytes())

    # ---- 2. passage int8 vectors (rescore table) ---------------------------
    # Built from the `whole` strategy: one vector per passage, aligned to ordinals.
    whole_meta = [orjson.loads(l) for l in
                  (vdir / f"meta.{args.lang}.whole.jsonl").read_bytes().splitlines()]
    whole_vec = np.load(vdir / f"vec.{args.lang}.whole.f32.npy", mmap_mode="r")
    pvec = np.zeros((n_pass, OUT_DIM), dtype=np.float32)
    seen = np.zeros(n_pass, dtype=bool)
    B = 20000
    for s in range(0, len(whole_meta), B):
        block = np.asarray(whole_vec[s:s + B], dtype=np.float32)
        proj = project(block, mean, comp)
        for j, m in enumerate(whole_meta[s:s + B]):
            o = pid_to_ord.get(m["parent_id"])
            if o is not None and not seen[o]:
                pvec[o] = proj[j]
                seen[o] = True
    print(f"passage vectors filled: {seen.sum():,}/{n_pass:,}")

    scale = np.abs(pvec).max(axis=1)
    scale[scale == 0] = 1.0
    p_int8 = np.clip(np.round(pvec / scale[:, None] * 127), -128, 127).astype(np.int8)
    (out / "passage_int8.bin").write_bytes(p_int8.tobytes())
    (out / "passage_scale.f32.bin").write_bytes(scale.astype(np.float32).tobytes())
    print(f"passage rescore table: {p_int8.nbytes/1e6:.1f} MB int8 + "
          f"{scale.nbytes/1e6:.1f} MB scale")
    del pvec, p_int8

    # ---- 3. per-strategy IVF ------------------------------------------------
    total_bytes = 0
    for strat in present:
        t1 = time.perf_counter()
        vec = np.load(vdir / f"vec.{args.lang}.{strat}.f32.npy", mmap_mode="r")
        meta = [orjson.loads(l) for l in
                (vdir / f"meta.{args.lang}.{strat}.jsonl").read_bytes().splitlines()]
        n = vec.shape[0]
        assert len(meta) == n, f"{strat}: {len(meta)} meta vs {n} vectors"

        parent = np.array([pid_to_ord.get(m["parent_id"], -1) for m in meta],
                          dtype=np.int32)
        keep = parent >= 0
        if not keep.all():
            print(f"  [{strat}] dropping {(~keep).sum():,} chunks with unknown parent")

        proj = np.zeros((n, OUT_DIM), dtype=np.float32)
        for s in range(0, n, B):
            proj[s:s + B] = project(np.asarray(vec[s:s + B], dtype=np.float32), mean, comp)

        nlist = int(np.clip(round(2 * np.sqrt(n)), 128, 4096))
        km_sample = proj[rng.choice(n, size=min(n, 200_000), replace=False)]
        cent = minibatch_kmeans(km_sample, nlist, rng=rng)
        del km_sample

        assign = np.zeros(n, dtype=np.int32)
        for s in range(0, n, B):
            assign[s:s + B] = (proj[s:s + B] @ cent.T).argmax(axis=1)
        assign[~keep] = -1

        order = np.argsort(assign, kind="stable")
        order = order[assign[order] >= 0]
        sorted_assign = assign[order]
        counts = np.bincount(sorted_assign, minlength=nlist)
        offsets = np.zeros(nlist + 1, dtype=np.uint32)
        offsets[1:] = np.cumsum(counts)

        codes = binarise(proj[order])
        slot_parent = parent[order].astype(np.int32)

        base = out / f"{strat}"
        (base.with_suffix(".centroids.f32.bin")).write_bytes(cent.tobytes())
        (base.with_suffix(".offsets.u32.bin")).write_bytes(offsets.tobytes())
        (base.with_suffix(".codes.u32.bin")).write_bytes(codes.tobytes())
        (base.with_suffix(".parents.i32.bin")).write_bytes(slot_parent.tobytes())

        sz = cent.nbytes + offsets.nbytes + codes.nbytes + slot_parent.nbytes
        total_bytes += sz
        manifest["strategies"][strat] = {
            "count": int(order.shape[0]),
            "nlist": nlist,
            "avgListSize": float(order.shape[0] / nlist),
        }
        print(f"  [{strat:11s}] n={order.shape[0]:>7,} nlist={nlist:<5} "
              f"avg_list={order.shape[0]/nlist:6.1f} "
              f"{sz/1e6:6.1f} MB  ({time.perf_counter()-t1:.1f}s)")
        del proj, codes

    # ---- 4. passage text (sharded) -----------------------------------------
    # Cloudflare Pages rejects any single file over 25 MiB, and the full Hindi
    # passage text is ~91 MB (Devanagari is 3 bytes/char in UTF-8). Shard it.
    # All shards are fetched in parallel at boot and concatenated; none of this
    # is in the 200ms path.
    texts = [p["text"] for p in passages]
    SHARD_TARGET = 8_000_000            # ~8 MB raw, comfortably under the limit
    shards, cur, cur_bytes, first = [], [], 0, 0
    for i, s in enumerate(texts):
        cur.append(s)
        cur_bytes += len(s.encode("utf-8")) + 8
        if cur_bytes >= SHARD_TARGET:
            shards.append((first, cur)); first = i + 1
            cur, cur_bytes = [], 0
    if cur:
        shards.append((first, cur))

    tsz = 0
    for k, (start, items) in enumerate(shards):
        name = f"passages.{k:03d}.json"
        (out / name).write_bytes(orjson.dumps(items))
        sz = (out / name).stat().st_size
        tsz += sz
        assert sz < 25 * 1024 * 1024, f"{name} is {sz/1e6:.1f} MB — exceeds Pages limit"
    manifest["passageShards"] = [f"passages.{k:03d}.json" for k in range(len(shards))]
    print(f"\npassage text: {tsz/1e6:.1f} MB raw across {len(shards)} shards "
          f"(max {max((out / n).stat().st_size for n in manifest['passageShards'])/1e6:.1f} MB, "
          f"CDN gzip ~{tsz/4e6:.0f} MB)")

    (out / "manifest.json").write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    print(f"index blobs : {total_bytes/1e6:.1f} MB")
    print(f"TOTAL ship  : {(total_bytes + tsz)/1e6:.1f} MB raw")


if __name__ == "__main__":
    main()
