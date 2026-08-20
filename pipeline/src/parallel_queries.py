"""Pull parallel queries for the multilingual stress test.

MSMARCO-XI is the *same* MS MARCO queries translated into 14 Indic languages, so
`query_id` is a join key across language files. That means we don't have to
translate anything ourselves — the parallel corpus already exists, and machine
translation of our own would put translation error inside the thing we're
measuring.

We only need three columns (`query_id`, `query`, `Eng_Query`). Parquet is
columnar, so reading those over HTTP range requests skips the `passages` column
that makes up ~95% of each 460 MB file. Fourteen languages cost a few MB
instead of 6 GB.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/ai4bharat/MSMARCO-XI/validation"

# code -> (file stem, endonym, BCP-47ish tag used by Sarvam/ElevenLabs)
LANGS: dict[str, tuple[str, str, str]] = {
    "hin": ("hinval", "हिन्दी",     "hi-IN"),
    "ben": ("benval", "বাংলা",      "bn-IN"),
    "tam": ("tamval", "தமிழ்",      "ta-IN"),
    "tel": ("telval", "తెలుగు",     "te-IN"),
    "mar": ("marval", "मराठी",      "mr-IN"),
    "guj": ("gujval", "ગુજરાતી",    "gu-IN"),
    "kan": ("kanval", "ಕನ್ನಡ",      "kn-IN"),
    "mal": ("malval", "മലയാളം",     "ml-IN"),
    "pan": ("panval", "ਪੰਜਾਬੀ",     "pa-IN"),
    "ori": ("orival", "ଓଡ଼ିଆ",       "od-IN"),
    "asm": ("asmval", "অসমীয়া",    "as-IN"),
    "urd": ("urdval", "اردو",       "ur-IN"),
    "nep": ("nepval", "नेपाली",     "ne-NP"),
    "san": ("sanval", "संस्कृतम्",   "sa-IN"),
}


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wanted", default="data/subset/queries.hin.jsonl",
                    help="query ids we already have gold passages for")
    ap.add_argument("--corpus", default="data/subset/corpus.hin.jsonl",
                    help="passage order — line number IS the browser ordinal")
    ap.add_argument("--out", default="../web/bench/data/multilingual.json")
    ap.add_argument("--n", type=int, default=120, help="how many parallel query sets")
    ap.add_argument("--row-groups", type=int, default=6,
                    help="row groups to read per language file")
    args = ap.parse_args()

    wanted: dict[int, dict] = {}
    with open(args.wanted, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            # Only answerable queries: an unanswerable one refuses in every
            # language, which tells us nothing about cross-lingual retrieval.
            if r.get("answerable") and r.get("gold_passage_ids"):
                wanted[r["query_id"]] = r
    print(f"{len(wanted):,} answerable queries available as join keys")

    # The browser index refers to passages by ordinal, not by hash, and the
    # ordinal IS the line number in the corpus file (see build_index.py). The
    # stress test needs ordinals so it can ask the only question that matters
    # cross-lingually: did the Tamil query land on the same passage the Hindi
    # one did?
    pid_to_ord: dict[str, int] = {}
    with open(args.corpus, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            pid_to_ord[json.loads(line)["id"]] = i
    print(f"{len(pid_to_ord):,} passages in the corpus ordering")

    fs = HfFileSystem()
    # qid -> {lang: text}
    rows: dict[int, dict[str, str]] = {}
    english: dict[int, str] = {}

    for code, (stem, endonym, tag) in LANGS.items():
        path = f"{REPO}/{stem}.parquet"
        try:
            with fs.open(path, "rb") as fh:
                pf = pq.ParquetFile(fh)
                ngroups = min(args.row_groups, pf.num_row_groups)
                tbl = pf.read_row_groups(
                    list(range(ngroups)),
                    columns=["query_id", "query", "Eng_Query"],
                )
        except Exception as exc:                       # noqa: BLE001
            print(f"  {code}: SKIPPED ({type(exc).__name__}: {exc})")
            continue

        qids = tbl.column("query_id").to_pylist()
        qs = tbl.column("query").to_pylist()
        eqs = tbl.column("Eng_Query").to_pylist()
        hit = 0
        for qid, q, eq in zip(qids, qs, eqs):
            if qid not in wanted:
                continue
            rows.setdefault(qid, {})[code] = norm(q)
            english.setdefault(qid, norm(eq))
            hit += 1
        print(f"  {code:>3} {endonym:<10} {len(qids):>7,} rows read, {hit:>5,} join")

    # Keep only query ids present in EVERY language we managed to read, so each
    # row of the stress test is a genuine like-for-like comparison.
    got = sorted({c for m in rows.values() for c in m})
    full = [q for q, m in rows.items() if len(m) == len(got)]
    full.sort()
    print(f"\n{len(got)} languages; {len(full):,} query ids present in all of them")

    out = []
    for qid in full:
        w = wanted[qid]
        gold = [pid_to_ord[p] for p in w["gold_passage_ids"] if p in pid_to_ord]
        # A query whose gold passage never made it into the shipped subset
        # cannot be scored, and including it would drag every language's
        # accuracy down by the same amount for a reason unrelated to language.
        if not gold:
            continue
        out.append({
            "query_id": qid,
            "eng_query": english[qid],
            "query_type": w.get("query_type"),
            "gold": gold,
            "by_lang": rows[qid],
        })
        if len(out) >= args.n:
            break

    payload = {
        "languages": [
            {"code": c, "name": LANGS[c][1], "tag": LANGS[c][2]} for c in got
        ],
        "n": len(out),
        "queries": out,
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest} — {len(out)} parallel sets x {len(got)} languages")


if __name__ == "__main__":
    main()
