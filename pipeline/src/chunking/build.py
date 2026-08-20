"""Run every chunking strategy over the corpus and emit one chunk file per strategy.

Separate files per strategy is deliberate: each becomes its own index in the
browser, searched independently, then fused with Reciprocal Rank Fusion. Keeping
them separate (rather than one merged index) is what lets a passage be found by
whichever strategy suits the question, and lets us measure each strategy's
contribution in isolation.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunking.strategies import PASSAGE_STRATEGIES, strat_document  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/subset")
    ap.add_argument("--lang", default="hin")
    ap.add_argument("--out", default="data/chunks")
    args = ap.parse_args()

    sub = Path(args.subset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    corpus = {}
    with (sub / f"corpus.{args.lang}.jsonl").open("rb") as f:
        for line in f:
            r = orjson.loads(line)
            corpus[r["id"]] = r
    print(f"corpus: {len(corpus):,} passages")

    queries = []
    with (sub / f"queries.{args.lang}.jsonl").open("rb") as f:
        for line in f:
            queries.append(orjson.loads(line))
    print(f"queries: {len(queries):,}")

    # query_types each passage serves -> feeds the contextual strategy's header
    ptypes: dict[str, set[str]] = defaultdict(set)
    for q in queries:
        for pid in q["candidate_passage_ids"]:
            ptypes[pid].add(q["query_type"])

    stats = Counter()
    chars = Counter()

    for name, fn in PASSAGE_STRATEGIES.items():
        path = out / f"chunks.{args.lang}.{name}.jsonl"
        with path.open("wb") as f:
            for pid, rec in corpus.items():
                meta = {"query_types": sorted(ptypes.get(pid, ()))}
                for c in fn(pid, rec["text"], meta):
                    stats[name] += 1
                    chars[name] += len(c.embed_text)
                    f.write(orjson.dumps({
                        "chunk_id": c.chunk_id,
                        "parent_id": c.parent_id,
                        "strategy": c.strategy,
                        "embed_text": c.embed_text,
                        "span": list(c.span),
                        "meta": c.meta,
                    }) + b"\n")

    # document strategy operates on query groups, not single passages
    path = out / f"chunks.{args.lang}.document.jsonl"
    seen_groups = set()
    with path.open("wb") as f:
        for q in queries:
            members = [(pid, corpus[pid]["text"])
                       for pid in q["candidate_passage_ids"] if pid in corpus]
            if len(members) < 2:
                continue
            key = tuple(sorted(p for p, _ in members))
            if key in seen_groups:          # identical passage sets recur
                continue
            seen_groups.add(key)
            for c in strat_document(str(q["query_id"]), members):
                stats["document"] += 1
                chars["document"] += len(c.embed_text)
                f.write(orjson.dumps({
                    "chunk_id": c.chunk_id,
                    "parent_id": c.parent_id,
                    "strategy": c.strategy,
                    "embed_text": c.embed_text,
                    "span": list(c.span),
                    "meta": c.meta,
                }) + b"\n")

    total = sum(stats.values())
    print("\n%-12s %10s %8s %10s %12s" % ("strategy", "chunks", "x/psg", "avg chars", "embed cost"))
    print("-" * 56)
    for name in list(PASSAGE_STRATEGIES) + ["document"]:
        n = stats[name]
        if not n:
            print("%-12s %10s %8s %10s %12s" % (name, 0, "-", "-", "(no output)"))
            continue
        print("%-12s %10s %8.2f %10.0f %11.1f%%" % (
            name, f"{n:,}", n / len(corpus), chars[name] / n, 100 * n / total))
    print("-" * 56)
    print("%-12s %10s %8.2f" % ("TOTAL", f"{total:,}", total / len(corpus)))

    print(f"\nvector budget @384d:")
    print(f"  binary  (48 B/vec): {total * 48 / 1e6:7.1f} MB")
    print(f"  int8   (384 B/vec): {total * 384 / 1e6:7.1f} MB")
    print(f"  fp32  (1536 B/vec): {total * 1536 / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
