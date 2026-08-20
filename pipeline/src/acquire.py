"""Build a working subset of MSMARCO-XI.

Strategy: sample N *queries*, then keep every passage attached to them. That keeps
MS MARCO's own hard negatives (the ~9 non-answer passages per query) in the corpus,
so retrieval is measured against realistic distractors rather than random text.

Sampling is stratified two ways:
  - query_type  (DESCRIPTION / NUMERIC / ENTITY / PERSON / LOCATION)
  - answerable  (has >=1 is_selected passage) vs unanswerable (none)

The unanswerable slice is deliberately preserved: it is the abstention test set.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import orjson
import pyarrow.parquet as pq

# A handful of passages are corrupt/degenerate (one Hindi passage is 21k chars).
MIN_PASSAGE_CHARS = 40
MAX_PASSAGE_CHARS = 2000


def normalise(text: str) -> str:
    """NFC-normalise and collapse whitespace.

    Devanagari can encode the same grapheme multiple ways; without NFC the same
    passage dedupes as two distinct strings and the embedder sees different bytes.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def passage_id(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="data/subset")
    ap.add_argument("--queries", type=int, default=15000)
    ap.add_argument("--lang", default="hin")
    ap.add_argument("--seed", type=int, default=20260819)
    # Share of sampled queries that are unanswerable. The raw file is 45%
    # unanswerable; this keeps a useful slice without swamping the retrieval
    # quality metrics.
    ap.add_argument("--unanswerable-share", type=float, default=0.30)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(
        args.parquet,
        columns=["query_id", "query", "Eng_Query", "Answer", "Eng_Answer",
                 "query_type", "passages"],
    )
    n = table.num_rows
    print(f"loaded {n:,} rows from {args.parquet}")

    qid = table.column("query_id").to_pylist()
    query = table.column("query").to_pylist()
    eq = table.column("Eng_Query").to_pylist()
    ans = table.column("Answer").to_pylist()
    eans = table.column("Eng_Answer").to_pylist()
    qtype = table.column("query_type").to_pylist()
    passages = table.column("passages").to_pylist()

    # ---- stratify -----------------------------------------------------------
    buckets: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for i in range(n):
        answerable = sum(passages[i]["is_selected"]) > 0
        buckets[(qtype[i], answerable)].append(i)

    print("\nsource distribution:")
    for k in sorted(buckets):
        print(f"  {k[0]:12s} answerable={str(k[1]):5s} {len(buckets[k]):7,}")

    n_unans = int(args.queries * args.unanswerable_share)
    n_ans = args.queries - n_unans

    # Split each target proportionally across query_type, using the *answerable*
    # subpopulation's own type distribution so we don't distort it.
    def allocate(target: int, answerable: bool) -> list[int]:
        pool = {t: buckets[(t, answerable)] for t in
                {k[0] for k in buckets if k[1] == answerable}}
        total = sum(len(v) for v in pool.values())
        picked: list[int] = []
        for t, idxs in sorted(pool.items()):
            take = min(len(idxs), round(target * len(idxs) / total))
            picked += rng.sample(idxs, take)
        return picked

    selected = allocate(n_ans, True) + allocate(n_unans, False)
    rng.shuffle(selected)
    print(f"\nsampled {len(selected):,} queries "
          f"({n_ans:,} answerable / {n_unans:,} unanswerable target)")

    # ---- emit corpus + queries ---------------------------------------------
    corpus: dict[str, dict] = {}
    skipped_short = skipped_long = 0
    q_records = []

    for i in selected:
        p = passages[i]
        texts = p["Translated_passages"]
        eng = p["English_passages"]
        sel = p["is_selected"]
        gold: list[str] = []
        member: list[str] = []

        for j, raw in enumerate(texts):
            if raw is None:
                continue
            text = normalise(raw)
            if len(text) < MIN_PASSAGE_CHARS:
                skipped_short += 1
                continue
            if len(text) > MAX_PASSAGE_CHARS:
                skipped_long += 1
                continue
            pid = passage_id(text)
            if pid not in corpus:
                corpus[pid] = {
                    "id": pid,
                    "text": text,
                    "eng": normalise(eng[j]) if j < len(eng) and eng[j] else "",
                    "lang": args.lang,
                    # provenance: which queries this passage was attached to
                    "from_qids": [],
                }
            corpus[pid]["from_qids"].append(qid[i])
            member.append(pid)
            if j < len(sel) and sel[j] == 1:
                gold.append(pid)

        if not member:
            continue

        q_records.append({
            "query_id": qid[i],
            "query": normalise(query[i]),
            "eng_query": normalise(eq[i] or ""),
            "answer": normalise(ans[i] or ""),
            "eng_answer": normalise(eans[i] or ""),
            "query_type": qtype[i],
            "gold_passage_ids": gold,
            "candidate_passage_ids": member,
            "answerable": len(gold) > 0,
        })

    corpus_path = out / f"corpus.{args.lang}.jsonl"
    queries_path = out / f"queries.{args.lang}.jsonl"

    with corpus_path.open("wb") as f:
        for rec in corpus.values():
            f.write(orjson.dumps(rec) + b"\n")
    with queries_path.open("wb") as f:
        for rec in q_records:
            f.write(orjson.dumps(rec) + b"\n")

    n_ans_final = sum(1 for r in q_records if r["answerable"])
    chars = sum(len(c["text"]) for c in corpus.values())

    print(f"\ncorpus  : {len(corpus):,} unique passages -> {corpus_path}")
    print(f"queries : {len(q_records):,} ({n_ans_final:,} answerable, "
          f"{len(q_records) - n_ans_final:,} unanswerable) -> {queries_path}")
    print(f"skipped : {skipped_short:,} too short, {skipped_long:,} too long")
    print(f"text    : {chars / 1e6:.1f}M chars "
          f"(~{chars * 3 / 1e6:.0f} MB raw utf-8, ~{chars * 3 / 4e6:.0f} MB gzipped)")


if __name__ == "__main__":
    main()
