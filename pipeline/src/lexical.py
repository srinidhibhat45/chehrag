"""BM25 lexical index — the fourth axis from `chunking/strategies.py`.

The other six indices all change *what* gets embedded. This one changes the
matching rule: it does not embed anything at all. Same unit as the `whole`
strategy — one entry per passage — scored by term statistics instead of by
cosine over a learned vector.

It is here because dense retrieval has one well-known blind spot and this corpus
is full of it. A 384-dimension vector averaged over a passage cannot hold an
exact token: a year, a price, a drug name, an acronym transliterated into
Devanagari. 25% of MSMARCO-XI's queries are NUMERIC-type. The `contextual`
strategy already tries to help by hoisting numbers and Latin runs into a header,
but that is still the same lossy averaging — it makes the numbers *more present*
in the vector, not exact. BM25 matches them exactly or not at all.

The two failure modes are close to complementary, which is the whole argument
for fusing them rather than choosing:

    dense wins   paraphrase, cross-lingual, synonymy, "what is a corporation"
                 against a passage that never says "corporation"
    BM25 wins    rare exact tokens — "1947", "NATO", "एचआईवी", a surname

What this costs, and how it is kept small:

  df ceiling     a term in more than `--df-max-ratio` of passages carries under
                 1 bit of IDF and owns the longest postings list in the index.
                 Dropping those is not pruning for size, it is dropping terms
                 that were never going to change a ranking. It is a stoplist
                 derived from the corpus rather than written by hand, which is
                 what makes it work for fourteen languages we have no stoplist
                 for.
  tf as u8       term frequency saturates in BM25 (that is what k1 is for), so
                 a count above 255 and a count of 255 score identically to
                 four decimal places.
  hashed terms   the dictionary ships as 64-bit FNV-1a hashes, not strings.
                 A 300k-term string table is several MB of JSON to parse at
                 boot; a sorted hash array is binary-searched in place with no
                 parse at all. 64 bits makes a collision ~1 in 10^8 across this
                 vocabulary, against ~10 expected collisions at 32 bits.

Scoring is done at query time from tf, df and document length rather than baked
into a precomputed impact, so k1 and b can be re-tuned against the eval without
rebuilding the index.

Tokenisation is a deliberate port of `web/src/retrieval/tokens.ts`. If the two
disagree the index is silently wrong — the query would be looking up terms that
were never indexed under that spelling — so `bench/lexparity.ts` asserts the two
implementations agree token for token and hash for hash.
"""

from __future__ import annotations

import argparse
import json
import time
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import orjson
import regex as re

# -- tokenisation, mirroring web/src/retrieval/tokens.ts ---------------------

EN_STOP = {
    "a", "an", "the", "this", "that", "these", "those",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "we", "us", "our", "ours",
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "as", "into", "onto", "over", "under", "up", "down", "out", "off",
    "and", "or", "but", "if", "then", "than", "so", "because",
    "there", "here", "not", "no", "any", "some", "all", "each", "very",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "tell", "give", "show", "please", "kindly",
}

HI_STOP = {
    "का", "के", "की", "को", "में", "से", "पर", "है", "हैं", "था", "थे", "थी",
    "और", "या", "यह", "वह", "ये", "वे", "एक", "कि", "जो", "तो", "ही", "भी",
    "हूँ", "हूं", "हो", "होता", "होती", "होते", "किया", "करना", "करें", "कर",
    "क्या", "कौन", "कब", "कहाँ", "कहां", "क्यों", "कैसे", "कितना", "कितनी", "किस",
    "मेरा", "मेरी", "मेरे", "आपका", "आपकी", "तुम", "आप", "मैं", "हम", "उनका",
    "बताओ", "बताएं", "बताइए",
    # Marathi
    "चा", "ची", "चे", "मध्ये", "आहे", "आहेत", "होता", "आणि", "काय", "कोण",
    "कुठे", "कसे", "किती", "सांगा",
}

# `\p{M}` is load-bearing: Indic vowel signs are combining marks, and a
# `\p{L}`-only class cuts every Devanagari word at its first matra. ZWNJ/ZWJ are
# kept because Devanagari uses them to control conjuncts. See tokens.ts.
_WORD_BREAK = re.compile(r"[^\p{L}\p{N}\p{M}‌‍\s]")
_HAS_LATIN = re.compile(r"[a-z]", re.IGNORECASE)
_HAS_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def raw_tokens(s: str) -> list[str]:
    return [w for w in _WORD_BREAK.sub(" ", s.lower()).split() if len(w) > 1]


def _is_stopword(w: str) -> bool:
    if _HAS_LATIN.search(w):
        return w in EN_STOP
    if _HAS_DEVANAGARI.search(w):
        return w in HI_STOP
    return False                       # script with no list — never filtered


def content_tokens(s: str) -> list[str]:
    """Content words, in order, duplicates kept.

    Duplicates matter here and do not in `tokens.ts`: that returns a set because
    gate 2 asks "was this word present", where BM25 asks "how often". The
    fallback to unfiltered tokens is the same in both — a string of nothing but
    function words names no subject, and an empty token list would make every
    passage score zero.
    """
    raw = raw_tokens(s)
    content = [w for w in raw if not _is_stopword(w)]
    return content if content else raw


# -- term hashing -----------------------------------------------------------

_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a64(s: str) -> int:
    """FNV-1a over the UTF-8 bytes. Ported to TS in `retrieval/lexical.ts`.

    Chosen over anything stronger because it is four lines in both languages and
    has no endianness or block-padding behaviour to get subtly wrong across an
    implementation boundary. The hash is not adversarial input here — it indexes
    a fixed corpus we built ourselves.
    """
    h = _FNV64_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * _FNV64_PRIME) & _MASK64
    return h


# -- build ------------------------------------------------------------------

def build(passages: list[str], df_max_ratio: float = 0.15,
          verbose: bool = True) -> dict:
    """Return the arrays that make up the shipped lexical index."""
    t0 = time.perf_counter()

    vocab: dict[str, int] = {}
    doc_ids = array("I")
    term_ids = array("I")
    tfs = array("B")
    lengths = np.zeros(len(passages), dtype=np.uint32)

    for d, text in enumerate(passages):
        toks = content_tokens(text)
        lengths[d] = len(toks)
        for term, tf in Counter(toks).items():
            tid = vocab.get(term)
            if tid is None:
                tid = len(vocab)
                vocab[term] = tid
            doc_ids.append(d)
            term_ids.append(tid)
            tfs.append(min(tf, 255))
        if verbose and d and d % 20000 == 0:
            print(f"  tokenised {d:,}/{len(passages):,}")

    n_post_raw = len(doc_ids)
    if verbose:
        print(f"  vocabulary {len(vocab):,} terms | postings {n_post_raw:,} "
              f"| {time.perf_counter() - t0:.1f}s")

    tid_arr = np.frombuffer(term_ids, dtype=np.uint32)
    did_arr = np.frombuffer(doc_ids, dtype=np.uint32)
    tf_arr = np.frombuffer(tfs, dtype=np.uint8)

    df = np.bincount(tid_arr, minlength=len(vocab))
    df_ceiling = max(2, int(df_max_ratio * len(passages)))
    keep_term = df <= df_ceiling
    dropped = int((~keep_term).sum())

    # Sort by (term, doc) so each term's postings are contiguous and ascending.
    order = np.lexsort((did_arr, tid_arr))
    tid_s, did_s, tf_s = tid_arr[order], did_arr[order], tf_arr[order]
    keep_post = keep_term[tid_s]
    tid_s, did_s, tf_s = tid_s[keep_post], did_s[keep_post], tf_s[keep_post]

    # Surviving terms, renumbered densely, still in term-id order.
    kept_tids = np.flatnonzero(keep_term)
    remap = np.full(len(vocab), -1, dtype=np.int64)
    remap[kept_tids] = np.arange(len(kept_tids))
    counts = np.bincount(remap[tid_s], minlength=len(kept_tids))
    offsets = np.zeros(len(kept_tids) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(counts)

    # The dictionary ships sorted by hash so the browser can binary-search it.
    id_to_term = [""] * len(vocab)
    for term, tid in vocab.items():
        id_to_term[tid] = term
    hashes = np.array([fnv1a64(id_to_term[t]) for t in kept_tids], dtype=np.uint64)
    hsort = np.argsort(hashes, kind="stable")

    dup = int((np.diff(hashes[hsort]) == 0).sum())
    if dup and verbose:
        print(f"  !! {dup} hash collisions — postings for those terms merge")

    # Postings reordered to follow the hash-sorted dictionary.
    starts, ends = offsets[:-1][hsort], offsets[1:][hsort]
    sel = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) \
        if len(starts) else np.zeros(0, dtype=np.int64)
    post_doc = did_s[sel].astype(np.uint32)
    post_tf = tf_s[sel].astype(np.uint8)
    new_offsets = np.zeros(len(hsort) + 1, dtype=np.uint32)
    new_offsets[1:] = np.cumsum((ends - starts).astype(np.uint32))

    h = hashes[hsort]
    return {
        "hashHi": (h >> np.uint64(32)).astype(np.uint32),
        "hashLo": (h & np.uint64(0xFFFFFFFF)).astype(np.uint32),
        "offsets": new_offsets,
        "docs": post_doc,
        "tf": post_tf,
        "lengths": np.minimum(lengths, 65535).astype(np.uint16),
        "stats": {
            "terms": int(len(hsort)),
            "postings": int(len(post_doc)),
            "vocabRaw": len(vocab),
            "postingsRaw": n_post_raw,
            "droppedTerms": dropped,
            "dfCeiling": df_ceiling,
            "avgLen": float(lengths.mean()),
            "collisions": dup,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/subset")
    ap.add_argument("--out", default="../web/public/index")
    ap.add_argument("--lang", default="hin")
    ap.add_argument("--df-max-ratio", type=float, default=0.15)
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--parity-out", default="/tmp/lexparity.json",
                    help="fixture for bench/lexparity.ts — tokens and hashes "
                         "this build produced, for the browser to reproduce")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    passages = []
    with (Path(args.subset) / f"corpus.{args.lang}.jsonl").open("rb") as f:
        for line in f:
            passages.append(orjson.loads(line)["text"])
    print(f"passages: {len(passages):,}")

    ix = build(passages, args.df_max_ratio)
    s = ix["stats"]

    files = {
        "lexical.hashhi.u32.bin": ix["hashHi"],
        "lexical.hashlo.u32.bin": ix["hashLo"],
        "lexical.offsets.u32.bin": ix["offsets"],
        "lexical.docs.u32.bin": ix["docs"],
        "lexical.tf.u8.bin": ix["tf"],
        "lexical.lengths.u16.bin": ix["lengths"],
    }
    total = 0
    for name, arr in files.items():
        (out / name).write_bytes(arr.tobytes())
        total += arr.nbytes

    # Merged into the existing manifest rather than written over it: the IVF
    # blobs are built by a separate, far more expensive step and must survive a
    # lexical rebuild.
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest["lexical"] = {
        "terms": s["terms"],
        "postings": s["postings"],
        "avgLen": s["avgLen"],
        "k1": args.k1,
        "b": args.b,
        "dfCeiling": s["dfCeiling"],
        # Versioned independently of `builtAt` so rebuilding the lexical index
        # does not invalidate 131 MB of cached IVF blobs in every browser that
        # already holds them.
        "version": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    mpath.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    # Parity fixture. Sampled across the corpus rather than from the head: the
    # tokeniser's hard cases are combining marks and mixed scripts, and the
    # first hundred passages are not representative of either.
    step = max(1, len(passages) // 300)
    sample = [passages[i][:400] for i in range(0, len(passages), step)][:300]
    sample += [
        "निगम क्या है", "भारत की राजधानी क्या है", "कोरोनावायरस",
        "what is a corporation", "NATO 1947 में", "HIV/AIDS की दवा",
        "एचआईवी", "COVID-19 vaccine", "45%", "१९४७ का वर्ष",
        "तमिळ் மொழி", "ਪੰਜਾਬੀ ਭਾਸ਼ਾ", "کیا ہے", "what is the name",
    ]
    Path(args.parity_out).write_bytes(orjson.dumps([
        {"text": t, "tokens": content_tokens(t),
         "hashes": [[(fnv1a64(w) >> 32) & 0xFFFFFFFF, fnv1a64(w) & 0xFFFFFFFF]
                    for w in dict.fromkeys(content_tokens(t))]}
        for t in sample
    ]))
    print(f"parity     : {len(sample)} fixtures -> {args.parity_out}")

    print(f"\nterms      : {s['terms']:,} kept of {s['vocabRaw']:,} "
          f"({s['droppedTerms']:,} above df ceiling {s['dfCeiling']:,})")
    print(f"postings   : {s['postings']:,} of {s['postingsRaw']:,} "
          f"({100 * s['postings'] / max(1, s['postingsRaw']):.1f}% kept)")
    print(f"avg length : {s['avgLen']:.1f} content tokens/passage")
    print(f"collisions : {s['collisions']}")
    print(f"ship       : {total / 1e6:.1f} MB raw "
          f"({total / max(1, len(passages)) :.0f} B/passage)")


if __name__ == "__main__":
    main()
