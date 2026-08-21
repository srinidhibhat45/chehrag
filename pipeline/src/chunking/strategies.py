"""Chunking strategies.

Chosen from measured corpus statistics. MS MARCO passages here are mean 317 /
median 295 chars (~60-80 tokens), which is to say already chunk-sized: a
512-token fixed-size splitter over a 295-char passage emits one chunk identical
to its input, an expensive no-op.

So the axes that pay on this corpus are:
  (a) going finer than a passage   -> sentence / sliding window
  (b) going coarser than a passage -> document merge across a query group
  (c) changing what gets embedded  -> contextual enrichment
  (d) changing the matching rule   -> lexical BM25 alongside dense

(a) through (c) live in this file. (d) is not a chunking strategy at all — it
indexes the same unit as `whole` and changes how a match is decided — so it
lives in `pipeline/src/lexical.py` and is fused as a seventh index.

Each strategy below states what it wins and what it costs, and
`web/bench/ablation.ts` measures those claims by removing one index at a time.
That measurement is worth reading before trusting anything in this file: over
1,412 answerable queries, no single dense strategy is worth more than 0.4
points of hit@5, because they find largely the same passages by slightly
different routes. Removing the BM25 index costs 2.3 points of hit@5 and 3.7 of
hit@10. Changing the matching rule buys more than any amount of changing where
the cut falls.

Which is an argument for the shape of the ensemble, not against the dense
strategies: fusion needs several imperfect voters, and the dense six are what
make a rank agree across strategies — the signal guardrail gate 2 reads to
decide whether a hit is real.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Devanagari danda + ASCII terminators. Deliberately conservative: over-splitting
# is cheaper than under-splitting because the sentence-window strategy re-expands
# to the parent passage anyway.
_SENT_RE = re.compile(r"(?<=[।.!?])\s+")
_NUMERIC_RE = re.compile(r"\d[\d,.\-/%]*")
# Devanagari has no case, so capitalisation-based entity detection is useless.
# Latin-script runs inside Indic text are usually proper nouns / units / acronyms
# and are exactly the tokens dense retrieval handles worst.
_LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z.\-]{2,}")


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str          # passage this chunk belongs to
    strategy: str
    embed_text: str         # what the embedder sees
    span: tuple[int, int]   # char offsets into the parent passage (for display)
    meta: dict = field(default_factory=dict)


def _cid(strategy: str, parent: str, k: int) -> str:
    h = hashlib.blake2b(f"{strategy}:{parent}:{k}".encode(), digest_size=6)
    return h.hexdigest()


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char spans of sentences."""
    spans, pos = [], 0
    for part in _SENT_RE.split(text):
        if not part:
            continue
        start = text.find(part, pos)
        if start < 0:
            start = pos
        spans.append((start, start + len(part)))
        pos = start + len(part)
    return spans or [(0, len(text))]


# ---------------------------------------------------------------------------
# 1. passage-native
# ---------------------------------------------------------------------------
def strat_whole(pid: str, text: str, meta: dict) -> list[Chunk]:
    """The passage as-is.

    Wins:  self-contained factual answers; strongest single baseline on MS MARCO,
           because the corpus was *built* as retrieval-sized units.
    Costs: a long passage covering two topics dilutes into one averaged vector.
    """
    return [Chunk(_cid("whole", pid, 0), pid, "whole", text, (0, len(text)))]


# ---------------------------------------------------------------------------
# 2. sentence-window  (retrieve small, return large)
# ---------------------------------------------------------------------------
def strat_sentence(pid: str, text: str, meta: dict, min_chars: int = 45) -> list[Chunk]:
    """Index each sentence; at query time the parent passage is returned.

    Wins:  precision. A one-sentence answer buried in a passage about something
           else is found, because its vector isn't averaged away.
    Costs: ~3x the vectors, and bare sentences lose anaphora ("it", "उसका").
           Mitigated by returning the parent for display and generation.
    """
    out = []
    for k, (s, e) in enumerate(split_sentences(text)):
        sent = text[s:e].strip()
        if len(sent) < min_chars:
            continue
        out.append(Chunk(_cid("sent", pid, k), pid, "sentence", sent, (s, e)))
    return out


# ---------------------------------------------------------------------------
# 3. sliding window with overlap
# ---------------------------------------------------------------------------
def strat_sliding(pid: str, text: str, meta: dict,
                  size: int = 200, stride: int = 120) -> list[Chunk]:
    """Fixed-width overlapping character windows.

    Window 200 / stride 120 gives 40% overlap. Sized from the measured median
    passage (295 chars) so a typical passage yields 2 windows, not 1 -- if the
    window matched the passage length this strategy would collapse into #1.

    Wins:  answers straddling a sentence boundary; robustness to bad punctuation
           (this corpus is machine-translated, so punctuation is not reliable).
    Costs: cuts mid-word/mid-clause; individually the least accurate strategy.
           It earns its place through fusion, not standalone quality.
    """
    if len(text) <= size:
        return [Chunk(_cid("slide", pid, 0), pid, "sliding", text, (0, len(text)))]
    out, k, pos = [], 0, 0
    while pos < len(text):
        seg = text[pos:pos + size]
        if len(seg) < 60 and out:      # don't emit a runt tail
            break
        out.append(Chunk(_cid("slide", pid, k), pid, "sliding", seg,
                         (pos, min(pos + size, len(text)))))
        k += 1
        pos += stride
    return out


# ---------------------------------------------------------------------------
# 4. contextual enrichment  (metadata-aware)
# ---------------------------------------------------------------------------
def strat_contextual(pid: str, text: str, meta: dict) -> list[Chunk]:
    """Prepend a derived context header to the passage before embedding.

    The header carries signals a bare passage vector represents poorly:
      - numeric spans      : 25% of this corpus is NUMERIC-type queries, and
                             dense embeddings are notoriously weak on exact figures
      - Latin-script runs  : in Devanagari text these are proper nouns, units and
                             acronyms -- high-value, low-frequency query terms
      - answer-type hint   : derived from which query_types this passage serves

    This is retrieval-time enrichment only; the header is never shown to the user
    and never enters the generated answer.

    Wins:  large gains on numeric/entity lookups.
    Costs: header tokens dilute the vector slightly for pure prose questions;
           the strategy is fused, not used alone, so the dilution is absorbed.
    """
    nums = _NUMERIC_RE.findall(text)[:8]
    lat = _LATIN_RUN_RE.findall(text)[:8]
    types = meta.get("query_types") or []
    bits = []
    if types:
        bits.append("प्रकार: " + " ".join(sorted(set(types))[:3]))
    if nums:
        bits.append("संख्या: " + " ".join(nums))
    if lat:
        bits.append("नाम: " + " ".join(dict.fromkeys(lat)))
    header = " | ".join(bits)
    embed = f"[{header}] {text}" if header else text
    return [Chunk(_cid("ctx", pid, 0), pid, "contextual", embed, (0, len(text)),
                  {"header": header})]


# ---------------------------------------------------------------------------
# 5. semantic (topic-shift) splitting
# ---------------------------------------------------------------------------
def strat_semantic(pid: str, text: str, meta: dict,
                   min_chars: int = 90, jaccard_cut: float = 0.08) -> list[Chunk]:
    """Split where adjacent sentences stop sharing vocabulary.

    Token-Jaccard between neighbouring sentences as a topic-shift proxy, rather
    than a second embedding pass: on 3-sentence passages that costs ~3x the
    build time to move a boundary fusion would find anyway.

    Expect low yield. The median passage is ~3 sentences, so there is often no
    boundary to find. Included because it is cheap and because the eval measures
    its contribution rather than assuming it.
    """
    spans = split_sentences(text)
    if len(spans) < 3:
        return []                      # nothing to gain; don't pay for a duplicate
    groups, cur = [], [spans[0]]
    for prev, nxt in zip(spans, spans[1:]):
        a = set(text[prev[0]:prev[1]].split())
        b = set(text[nxt[0]:nxt[1]].split())
        j = len(a & b) / len(a | b) if (a | b) else 0.0
        if j < jaccard_cut and sum(e - s for s, e in cur) >= min_chars:
            groups.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    groups.append(cur)
    if len(groups) < 2:
        return []                      # collapsed to the whole passage -> skip
    out = []
    for k, g in enumerate(groups):
        s, e = g[0][0], g[-1][1]
        seg = text[s:e].strip()
        if len(seg) >= min_chars:
            out.append(Chunk(_cid("sem", pid, k), pid, "semantic", seg, (s, e)))
    return out if len(out) >= 2 else []


# ---------------------------------------------------------------------------
# 6. document merge  (parent expansion, coarser than a passage)
# ---------------------------------------------------------------------------
def strat_document(group_id: str, passages: list[tuple[str, str]],
                   max_chars: int = 1200) -> list[Chunk]:
    """Concatenate a query group's passages into one pseudo-document.

    MS MARCO gives ~10 passages per query, retrieved together by the original
    search engine -- so they are topically coherent by construction. Some answers
    need two of them at once ("compare X and Y", multi-hop numerics).

    parent_id points at the *first* passage; the full member list rides in meta
    so the harness can expand to all of them.

    Wins:  answers spread across sibling passages.
    Costs: coarse vectors, high dilution. Contributes recall, never precision --
           and is weighted accordingly in fusion.
    """
    out, buf, members, k = [], [], [], 0
    for pid, text in passages:
        if buf and sum(len(b) for b in buf) + len(text) > max_chars:
            out.append(Chunk(_cid("doc", group_id, k), members[0], "document",
                             " ".join(buf), (0, 0), {"members": list(members)}))
            k += 1
            buf, members = [], []
        buf.append(text)
        members.append(pid)
    if buf:
        out.append(Chunk(_cid("doc", group_id, k), members[0], "document",
                         " ".join(buf), (0, 0), {"members": list(members)}))
    return out


PASSAGE_STRATEGIES = {
    "whole": strat_whole,
    "sentence": strat_sentence,
    "sliding": strat_sliding,
    "contextual": strat_contextual,
    "semantic": strat_semantic,
}
