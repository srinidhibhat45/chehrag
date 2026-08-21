# Chehrag

**चिराग़** *(chirāġ)*, a lamp, crossed with **RAG**. Speak a question, get an
answer grounded in sources, in under 200 milliseconds.

HH Goa 2026, Task 2. Live at **https://chehrag.pages.dev**.

The retrieval pipeline runs entirely in the browser. The measured path contains
no network at all.

---

## The decision everything else follows from

The brief asks for chunking, vector retrieval and everything through to final
output in under 200 ms. We costed the obvious architecture before writing any of
it:

| Component | Cost |
|---|---|
| One network round trip | 50-200 ms |
| Fastest streaming STT available (ElevenLabs Scribe v2) | ~150 ms typical |
| An LLM writing an answer | 500 ms+ |

The fastest speech-to-text on the market spends 150 of the 200 ms on its own.
So we moved the work instead of trying to outrun the network: the index and the
search engine are shipped to the browser once, and every question after that is
answered on the user's own machine.

That buys four things we wanted anyway:

- no round trip inside the measured span
- static hosting, global CDN, no cold starts (a sleeping free tier would put a
  30-second first request straight into P100)
- the system gets faster as more people use it, because each brings a CPU
- a document you add is parsed, chunked, embedded and indexed in your browser
  and never uploaded

---

## What the 200 ms covers

Precisely, so there is nothing to infer:

**Inside the budget**, measured, in the browser: query embedding, PCA
projection, search across seven indices, reciprocal-rank fusion, int8 rescoring,
input and retrieval guardrails, and an extractive answer drawn from the winning
passage. That is the full query path over 809,607 chunks and it is what the
timer under every answer reports.

**Outside the budget**, and labelled as such in the interface: speech-to-text
before the question exists, an optional model that rewrites the retrieved
passages into prose, and speech afterwards. Chunking, embedding and index
construction for the shipped corpus happen at build time, once.

Retrieval is the guarantee. Generation is an upgrade on top of it, timed
separately, and the app answers without it.

| | P50 | P70 | P100 | under 200 ms |
|---|---|---|---|---|
| **Browser**, quiet laptop (n=36) | **21.5 ms** | **22.9 ms** | **27.6 ms** | 36/36 |
| **Browser**, same build with Teams and Figma running (n=16) | **53.1 ms** | **62.6 ms** | **97.8 ms** | 16/16 |
| **Node harness**, quiet (n=500, three runs) | **2.6 ms** | **2.9 ms** | 15.0-17.5 ms | 100% |
| **Node harness**, busy (n=500) | 4.5 ms | 5.4 ms | 27.9 ms | 100% |

Both browser rows are the same build. The second is the one worth reading: with
the laptop genuinely busy, the worst query of sixteen was 97.8 ms, still half
the budget.

---

## The six requirements

| # | Requirement | Where |
|---|---|---|
| 1 | Speech-to-text via Sarvam or ElevenLabs | **Sarvam** `saaras:v3-realtime`, streaming over a WebSocket the Worker authenticates, with batch REST as a fallback (`web/src/stt/sarvam.ts`). Answers are spoken back with Sarvam `bulbul:v2` |
| 2 | Chunking must be vast, not one naive split | **Seven indices**: six chunking strategies plus a BM25 lexical index that changes the matching rule rather than the cut. Built separately, fused with RRF, applied to the corpus and to anything you add. Each one's contribution measured by leave-one-out ablation (`pipeline/src/chunking/strategies.py`, `web/bench/ablation.ts`) |
| 3 | Full pipeline under 200 ms | Retrieval runs in-browser and the cap is **enforced by a deadline-aware planner**, not hoped for. `bench/deadline.ts` holds it with zero overruns down to an 8 ms budget |
| 4 | P50 / P70 / P100 over many queries | `bench/run.ts` (500), `bench/multilingual.ts` (3,000 across 15 languages), `bench/deadline.ts` (9 budgets), `bench/voice.ts` (spoken, end to end) |
| 5 | A real harness | Typed stages with budgets, retries, degradation and a circuit breaker (`web/src/harness/pipeline.ts`), plus a bounded tool-call loop where the model names the excerpts it used and can request one more search that **the browser executes** (`worker/src/synthesize.ts`, `web/src/answer/generate.ts`) |
| 6 | Guardrails | Four checks: input, retrieval confidence (threshold fitted on 3,012 labelled unanswerable queries), fabricated citations, and grounding. 90 unit cases (`web/src/guardrails/gates.ts`) |

---

## Chunking: shaped by the data

We measured the corpus first. MS MARCO passages here are **mean 317 / median 295
characters** - already chunk-sized. A 512-token splitter over a 295-character
passage emits one chunk identical to its input, so on this corpus naive
fixed-size chunking is not merely naive, it is inert.

The axes that pay are different:

| Strategy | Direction | What it wins | What it costs |
|---|---|---|---|
| `whole` | baseline | self-contained answers; strongest single index | a topic-mixed passage averages into one vector |
| `sentence` | finer | precision; a one-sentence answer is not averaged away | 2.6x vectors, and bare sentences lose anaphora |
| `sliding` | finer | answers straddling boundaries; survives bad punctuation, and this corpus is machine-translated | cuts mid-clause; weakest alone |
| `contextual` | same size, richer | numerics and Latin-script proper nouns, which dense vectors handle worst | header tokens dilute pure-prose queries |
| `semantic` | finer | keeps complete ideas together | low yield here; the median passage is ~3 sentences |
| `document` | coarser | answers spread across sibling passages | dilute vectors; contributes recall, never precision |
| `lexical` (BM25) | **same unit, different rule** | exact tokens: a year, an acronym, a surname. 25% of this corpus's queries are numeric | no semantic match at all; blind across languages |

**809,607 chunks from 98,867 passages (8.19x)**, plus 2,995,581 BM25 postings.

Fused with Reciprocal Rank Fusion, which uses *rank* rather than score. That is
necessary rather than stylistic: a 111-character sentence and a 925-character
document are not score-comparable, and averaging their cosines would hand the
sentence index a permanent, invisible advantage.

### What each index is actually worth

`npm run bench:ablation` removes one index at a time and re-measures over 1,385
answerable queries, graded on rank **before** the guardrails - otherwise
removing an index could flatter itself by shifting queries across the
confidence threshold.

| config | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | Δ hit@5 |
|---|---|---|---|---|---|---|
| **all seven** | 28.7% | 50.6% | **59.1%** | **65.4%** | 0.4114 | - |
| – whole | 28.4% | 50.6% | 59.0% | 65.1% | 0.4088 | -0.1 |
| – sentence | 27.8% | 49.7% | 58.4% | 64.9% | 0.4043 | -0.6 |
| – sliding | 28.7% | 50.8% | 59.2% | 65.3% | 0.4122 | +0.1 |
| – contextual | 28.3% | 49.9% | 58.3% | 64.5% | 0.4067 | -0.7 |
| – semantic | 28.4% | 50.6% | 59.2% | 65.8% | 0.4114 | +0.1 |
| – document | 28.7% | 50.5% | 58.4% | 64.6% | 0.4098 | -0.6 |
| **– lexical** | 28.4% | 49.3% | **57.6%** | **62.5%** | 0.4026 | **−1.4** |
| lexical alone | 26.1% | 44.8% | 52.1% | 56.8% | 0.3662 | -7.0 |

No single dense strategy is worth more than 0.7 points of hit@5. **The lexical
index is worth 1.4, and 2.9 at hit@10** - the largest single contribution, and
not the answer we expected after building six chunkers.

The dense six are largely redundant with each other: they find the same passages
by slightly different routes, so removing any one leaves the others to cover it.
Changing the *matching rule* is worth more than changing where the cut falls,
which is what the corpus statistics predicted and we did not follow far enough
at first.

All seven stay, and not out of sentiment. Fusion needs several imperfect voters
that fail independently, and **cross-strategy agreement is a signal gate 2
reads** - a passage found by one index with no lexical support is the classic
shape of a spurious dense hit. Six overlapping voters are what make that signal
mean anything.

Cost of the seventh index: **+16.4 MB raw, ~7.7 MB over the wire, +0.1 ms** at
P50.

---

## The retrieval engine

Two-stage, and the shape is set by the worst case rather than the average.

1. **Binary codes and Hamming distance** over IVF clusters: a wide net with a
   hard candidate ceiling.
2. **int8 dot product** on the fused passage candidates only: real cosine for
   final ranking and for the confidence threshold.

Not a brute-force scan, which costs tens of milliseconds over 810k chunks in JS
- affordable on average and lethal at P100 with a GC pause on top. Not HNSW,
whose graph is large to ship and whose traversal has a long, data-dependent
tail. IVF bounds the work per query, so latency goes near-constant and the tail
flattens.

Every buffer is allocated at construction. The steady-state search allocates
nothing, which is the only reason the tail is flat: **minor GC is what sets
P100** in this pipeline, and we measured that directly - with a larger young
generation the same run's P100 falls from 12.3 ms to 6.8 ms while P50 does not
move.

Three things in `retrieval/ivf.ts` are worth naming because each replaced
something asymptotically worse:

- **Candidate selection is a counting sort.** Hamming distance is a small
  integer, so ordering the head of 6,144 candidates is O(n) where the partial
  selection sort it replaced was O(need × n) and cost more than the scan that
  produced the candidates.
- **Cluster ranking uses a PCA prefix.** PCA orders components by variance, so
  the first 96 of 256 dimensions carry most of what separates one cluster from
  another. A prefix pass shortlists 3× nprobe clusters, then the shortlist is
  scored on the full vector. Measured against full-dimension ranking it picks
  the same cluster set 99.7% of the time and the same nearest cluster 100% of
  the time, for 57% of the cost.
- **The dot products use four accumulators**, so the additions do not form one
  dependency chain.

Together these took retrieval from 2.07 ms to 0.80 ms at P50 (Node, all seven
indices) with hit@1 and hit@3 unchanged.

That headroom was then spent back on recall rather than banked. `nprobe` and the
candidate cap were swept against retrieval quality on 415 answerable queries,
again graded pre-guardrail:

| | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | P50 |
|---|---|---|---|---|---|---|
| nprobe 12, cap 3072 (before) | 31.6% | 52.0% | 61.9% | 67.2% | 0.4352 | 2.34 ms |
| **nprobe 24, cap 6144 (now)** | **32.3%** | **53.3%** | **64.1%** | **69.4%** | **0.4463** | 2.41 ms |

nprobe saturates at 24 and the cap at 6144; 9216 adds nothing. **+2.2 points of
hit@5 for 0.07 ms** - a larger retrieval gain than the entire lexical index
gave, bought with time the rewrite had already freed.

### The embedding graph gets four threads

`public/_headers` sets `Cross-Origin-Opener-Policy` and
`Cross-Origin-Embedder-Policy` so that onnxruntime can use `SharedArrayBuffer`.
Setting the headers is not enough on its own: onnxruntime-web still runs the
graph on **one** WASM thread unless it is told a number.

Measured in-browser over 36 corpus queries, embedding only:

| WASM threads | embed, median |
|---|---|
| 1 (the default) | 51.0 ms |
| **4** | **16.3 ms** |
| 8 | 15.2 ms |

Four is where the curve flattens, and the page still has a UI and a shader that
want cores. This one line is worth more than every other latency change in the
project combined: it took the browser P50 from ~56 ms to 21.5 ms.

---

## The 200 ms cap is enforced

Per-stage timeouts tell you a deadline was missed. They do not stop you missing
it. The only way to guarantee a cap is for the expensive stage to look at the
clock before it starts and choose an amount of work it can finish.

Embedding is the variable cost - a couple of ms here, 40-70 ms for the same WASM
graph on a mid-range Android, past 100 ms on a thermally throttled laptop. So
retrieval is told what is *left* after embedding and sizes itself to it, giving
up capability in a fixed order (`web/src/harness/deadline.ts`):

| Knob | Given up | Why in this order |
|---|---|---|
| `nprobe` | first | linear in cost, sublinear in recall; the first clusters hold most of the answer |
| `maxCandidates` | with nprobe | at 12 clusters an index rarely reaches even 3072 candidates, so nprobe already bounds the scan |
| `perStrategyK` | second | narrows what fusion has to work with |
| `rescoreTopN` | last | gate 2's calibrated threshold reads the rescored score |

**Testing it without a slow phone: shrink the budget instead.** Squeezing the
pipeline into 20 ms on this machine exercises the code path a 10x slower device
hits at 200 ms. `npm run bench:deadline` runs the full query set at nine budgets
and reports both whether the deadline held and what recall it cost, because a
cap met by returning nothing useful is not a cap worth having.

```
 budget      p50     p100   over  degraded   kept@3     n  worst plan
  200ms     2.58     5.03      0        0%   100.0%   169  nprobe 24, k 24, rescore 16
  100ms     2.66     4.57      0        0%   100.0%   169  nprobe 24, k 24, rescore 16
   50ms     2.66     8.47      0        0%   100.0%   169  nprobe 24, k 24, rescore 16
   30ms     2.63     5.51      0        0%   100.0%   169  nprobe 24, k 24, rescore 16
   20ms     2.80     8.91      0        0%   100.0%   169  nprobe 24, k 24, rescore 16
   15ms     3.83    11.13      0        7%    99.4%   169  nprobe 1, k 6, rescore 4
   12ms     2.43     4.90      0       87%    89.9%   169  nprobe 1, k 6, rescore 4
   10ms     2.20     4.45      0      100%    66.3%   169  nprobe 1, k 6, rescore 4
    8ms     2.19     5.57      0      100%    66.3%   169  nprobe 1, k 6, rescore 4
```

**Zero overruns from 200 ms down to an 8 ms budget**, and full retrieval quality
is retained down to 20 ms - the system tolerates a machine roughly 10x slower
than this one before it gives up anything at all. Below ~10 ms the budget is
smaller than a single embedding forward pass, which no retrieval plan can
recover; that is the floor of the technique and it is reported rather than
trimmed off the bottom of the table.

This benchmark has earned its keep three times. It found a `rescore` stage
reserving 12 ms for work costing 0.03 ms - skipped under pressure, after which
gate 2 compared its calibrated cosine threshold against a raw Hamming proxy and
refused almost everything. It found `embed` retrying on timeout, doubling the
spend at exactly the moment the budget was tightest. And when BM25 was added it
immediately broke, because `budgetPlan` was costing work it no longer knew the
shape of: the lexical scan is bounded by a postings cap rather than by `nprobe`,
so it is a fixed cost the ladder cannot reduce. A planner that does not know
about a stage will overrun by exactly that stage's cost.

---

## The harness

Requirement 5 asks for structured orchestration rather than a single raw
prompt-in, text-out call. There are two halves here and they run in different
places for a reason.

### Typed stages, under a deadline

Every unit of retrieval work is a declared `Stage` with a typed input and
output, a latency budget, a retry policy, a declared failure behaviour and
telemetry either way (`web/src/harness/pipeline.ts`). `FAIL` aborts the run;
`DEGRADE` falls back and carries on. Optional stages are skipped rather than
started when too little budget remains, because a stage abandoned halfway has
already spent the budget the required stages needed.

Two rules in there were written by measurement rather than taste:

- **A timeout is never retried.** Retries exist for transient faults. A stage
  that timed out was handed deterministic work that did not fit, and running it
  again produces the identical overrun at double the price.
- **A retry never starts if it cannot finish.**

The engine also runs **one query at a time**. Speculative retrieval fires on
partial transcripts while the user is still speaking, and the scratch buffers
that make the search allocation-free are shared, so an overlapping query would
embed into the vector a live one was still using. Queueing costs a microtask.

### The model calls tools, and the browser runs them

The model does not return prose. It returns a **tool call** against a typed
schema (`worker/src/synthesize.ts`):

| Tool | Purpose |
|---|---|
| `answer` | the answer **plus `excerpt_indices`**, naming which excerpts it used |
| `insufficient_context` | the excerpts do not answer the question |
| `search_corpus` | search again with different words. At most once |

`search_corpus` is the interesting one because of *where* it executes. The index
is in the browser and the Worker holds nothing but an API key, so the Worker
streams the tool call back, and the page runs it against the same sub-millisecond
engine that produced the first excerpts, appends what it finds, and posts the
transcript back (`web/src/answer/generate.ts`). The tool runs where the data is.
No passage is shipped to the Worker to make it possible, and the Worker stays
stateless, which is what lets a retry land on a different edge instance.

It exists because the first retrieval used the words the *user* said. A model
that has read the excerpts and can see they are about the wrong sense of a word
knows something the retriever did not.

**The loop is bounded structurally rather than by a counter.** Once the
transcript shows a search has run, the Worker does not offer the tool again, so
the model cannot spend a round trip asking for something that would be refused.
The browser independently stops after two rounds. Neither end depends on the
other to terminate.

**Answers still stream.** Once the answer is a tool call, its text arrives as
fragments of JSON that is not yet valid, and waiting for the closing brace would
replace a word-by-word answer with a one-second pause and a finished paragraph.
The stream handler pulls the partial `answer` string out of the incomplete JSON,
honouring escapes so a half-arrived `ह` never reaches the page as
backslash-u. Ten unit cases cover that extractor, including the split-escape
one.

**Retries and error recovery**, on both sides of the network:

- the client retries `/synthesize` on transport errors and 429/5xx only, never a
  4xx, with exponential backoff and jitter, because otherwise every visitor's
  page retries in the same millisecond and turns a provider's recovery into a
  second outage
- the Worker does the same for its outbound provider calls, rebuilding the
  `FormData` per attempt - a body that has been sent once cannot be sent again,
  and reusing it makes the retry fail in a way that looks like the upstream
  rejecting the audio
- a **circuit breaker** in front of the generator: after 3 consecutive failures
  it opens for 20 s and calls fail in ~0 ms instead of waiting out a full
  timeout, with one trial call allowed through on expiry
- every failure path still leaves a grounded extractive answer on screen

**How we know the loop works.** `npm run bench:tools` runs it twice over.

Against a scripted provider (no API calls, no key): turn one asks for a search,
turn two answers citing an excerpt that exists *only because the search ran*.
That is the path a well-behaved model correctly never takes on well-retrieved
queries, so without scripting it the most important code would go untested
precisely when everything is working. All 7 assertions pass: the tool is
dispatched exactly once, turn 2 replays the call and its result, the appended
excerpt arrives, and the citation index resolves to it.

Against Groq live, on 15 real corpus queries: 13 turns, 10 answered and 3
`insufficient_context` where the model read the passages and declined. **Every
answer named the excerpts it used, zero fabricated citations, zero transport
failures, and gate 3 passed all ten.** `search_corpus` fired zero times, which
is the correct behaviour on queries the first retrieval already answered; its
value is on the ones it does not, and that path is what the scripted test
covers.

---

## Guardrails

| Gate | Question | Method |
|---|---|---|
| 1 - input | Should we process this at all? | unsafe content, prompt injection (EN + HI), gibberish by character entropy, commands rather than questions, and whether there is anything loaded to search |
| 2 - retrieval | Did we actually find anything? | confidence threshold plus cross-strategy agreement |
| 3a - citation | Do the sources it cites exist? | the model names its excerpts; indices outside the supplied set are refused |
| 3b - grounding | Is every claim traceable to those sources? | token-level support, with an exact-match rule for numbers and names |

Gate 3a exists because the model answers by *calling a tool* and naming the
excerpts it used, which makes the claim checkable instead of inferred. An answer
citing excerpt 7 of 5 has stopped reading and started composing, and its prose
can still overlap the real passages well enough to pass a coverage test. It is
refused as `FABRICATED_CITATION`, distinct from `UNGROUNDED`: one asserts
something the sources do not support, the other is wrong about where it came
from.

That also makes 3b strictly harder, because grounding is measured against the
excerpts the model *claims* to have used rather than everything retrieved. An
answer written from excerpt 1 while citing excerpt 3 used to pass on the union
and no longer does.

Gate 1 refuses a question asked with nothing loaded in 0.1 ms, before any vector
is computed, and says *that* rather than "I couldn't find anything good enough".
The two are different facts.

**Gate 2 is calibrated, not guessed.** The corpus ships 3,012 queries with no
relevant passage - genuine cases where refusing is correct. The threshold is
fitted on a calibration split and reported on a holdout that never influenced
selection (`npm run calibrate`). The objective is deliberately not accuracy:
answering confidently when you should not is worse than declining too often, so
selection targets a coverage policy (answer at least 85% of answerable queries)
and reports the abstention it buys. Maximising F1 instead chooses a threshold of
0.68 and produces a system that refuses 71% of real questions.

| Gate 2, n=500 | |
|---|---|
| coverage of answerable | 88.3% |
| abstention precision | 0.512 |
| abstention recall | 0.285 |
| F1 | 0.366 |
| fitted threshold | minTopScore 0.4995, minAgreement 1 |

Holdout after refitting, never used for selection: coverage 85.4%, precision
0.487, recall 0.303, F1 0.374.

MS MARCO's "unanswerable" queries still carry ten topically relevant passages
retrieved by the original search engine - they are on topic but non-answering,
and measured separability on that split is AUC 0.65. The production job is
easier and gate 2 does it well: on genuinely out-of-corpus questions ("who won
the 2029 world cup on mars", "what is my bank balance") the fitted threshold
blocks 7 of 8 while keeping 84% of in-corpus queries answerable.

Gates 1, 3a and 3b: **90/90 unit cases pass** (`npm run bench:gates`), including
the false-positive guards - "How do vaccines work?", "what caused the second
world war" and "how to treat a snake bite" all correctly pass. The filter blocks
operational harm requests, not subject matter. Self-harm routes to a support
line rather than a bare refusal.

### The threshold does not transfer to your own documents

`minTopScore` is an absolute cosine fitted on MS MARCO, whose passages are
search results: dense, on topic, and selected *because* they answered a query. A
user's HR policy or a page of prose is not written that way, and its genuinely
correct passages score where MS MARCO's mediocre ones do. Measured on 43
labelled questions over two English documents, the single fitted threshold
refused 25% of answerable questions - including "what equipment does the company
provide", which the document answers word for word.

The mid-band turns out to be separable, just not by score. Every false positive
in it had weak word overlap with the winning passage (0.0-0.4): the embedder had
found the right *subject* and the wrong *question*. Every true positive had
strong overlap (0.5-1.0). So a hit scoring in the mid-band whose winning passage
shares at least half the query's content words is answered rather than refused.

`npm run bench:usersource` fits the floor on the split that governs it, 39
labelled questions over one field-structured document and one prose one:

| rescue floor | coverage | abstention |
|---|---|---|
| none | 40.0% (10/25) | 100% (14/14) |
| 0.40 | 52.0% (13/25) | 100% (14/14) |
| 0.30 | 76.0% (19/25) | 92.9% (13/14) |
| **0.20** | **88.0% (22/25)** | 92.9% (13/14) |
| 0.15 / 0.00 | 88.0% (22/25) | 92.9% (13/14) |

0.20 is the *highest* value on the coverage plateau, so the floor still excludes
the genuinely unrelated band ("what is the capital of Peru" scores 0.123 here)
rather than being switched off.

**It applies to user sources only.** Enabling it on the corpus was measured and
rejected: coverage rose 88.5% → 93.1% but abstention recall fell 0.245 → 0.172,
buying 16 more correct answers with 11 more wrong ones on the one split where
3,012 labels say which is which.

---

## Measured results

### Latency

**Browser**, corpus questions asked one after another with a pause between them,
which is how a person actually uses it. Chromium, cross-origin isolated, warm
index. Two machine states, same build:

| | quiet (n=36) | busy (n=16) |
|---|---|---|
| P50 | **21.5 ms** | 53.1 ms |
| P70 | **22.9 ms** | 62.6 ms |
| P100 | **27.6 ms** | 97.8 ms |
| under 200 ms | 36/36 | 16/16 |
| embed, median | 16.3 ms | 35.1 ms |
| retrieve, median | 4.9 ms | 15.1 ms |

"Busy" is an ordinary working laptop: a video-call client and a design tool in
the background, load average around 4. Both halves of the query slow by the same
factor, which is what tells you it is the machine and not the code — retrieval
is plain JavaScript over typed arrays and shares nothing with the WASM forward
pass. We report both because a judge's laptop will not be idle, and the cap
holds either way.

Embedding is 76% of the query. It is a WASM forward pass through a
33M-parameter model, and it is what any further work would have to attack.

**Node harness** (n=500, same modules, native onnxruntime), three consecutive
runs on a quiet machine because one run misrepresents the tail:

| | ms |
|---|---|
| P50 | 2.6 |
| P70 | 2.9 |
| P90 / P95 / P99 | 3.4-3.5 / 3.7-4.2 / 5.2-6.8 |
| P100 | 15.0 - 17.5 |
| mean | 2.7 - 2.9 |

The same command on the busy machine reads P50 4.5, P70 5.4, P100 27.9.

Per stage (P50 / P100 ms): `embed 1.7 / 15.0` · `retrieve 0.8 / 2.6` ·
`guard:input 0.01 / 2.5` · `rescore 0.01 / 0.05` · `guard:retrieval 0.03 / 0.14`
· `answer:extract 0.00 / 0.02`.

Searching all seven indices - six IVF scans plus a BM25 pass over three million
postings - costs 0.8 ms. Query embedding dominates the budget and owns the tail.

P50 is stable to within 0.1 ms across runs; P100 varies, and every bit of that
spread is a single outlier inside the embed stage. That is a GC event landing on
one query out of five hundred, not a code path, and the honest way to report it
is a range rather than the prettiest run. The worst P100 recorded is 9% of the
budget.

Refused queries short-circuit: an injection attempt is rejected at 0.1 ms
without spending an embedding.

The browser and Node numbers differ because Node uses native onnxruntime where
the browser uses WASM. **The browser number is the one that counts**, since that
is where the system runs.

### Retrieval quality

Brute-force ceiling on 500 gold-labelled queries: fp32-384 R@10 = 0.758.

Quantisation ladder, measured rather than assumed:

| config | R@10 | codes |
|---|---|---|
| fp32 384 (ceiling) | 0.758 | - |
| PCA-256 fp32 | 0.744 | 25.9 MB |
| → binary-256 | 0.604 | 25.9 MB |
| binary-384, no PCA | 0.228 | 38.9 MB |

**PCA is not compression here, it is what makes binary quantisation work at
all.** Sign bits over raw embedding dimensions are near-worthless because those
dimensions are correlated and off-centre. Dropping PCA to "keep more
information" would have destroyed retrieval.

End to end on the held-out query set (n=500): **hit@1 0.354, hit@3 0.581** over
308 graded answerable queries.

### Spoken, end to end

The architecture argument above is only an argument until someone measures it,
so `npm run bench:voice` does. Sarvam `bulbul:v2` speaks each Hindi query, that
audio goes back through Sarvam speech-to-text, and the transcript runs through
the shipped pipeline. Microphone to grounded answer, nothing excluded:

| n=30 | P50 | P70 | P100 | mean |
|---|---|---|---|---|
| Sarvam speech-to-text (network) | 451.2 ms | 593.1 ms | 1497.5 ms | 550.4 ms |
| **Retrieval pipeline (ours)** | **8.9 ms** | **9.9 ms** | **23.2 ms** | 9.9 ms |
| Total, voice in to answer out | 461.1 ms | 600.8 ms | 1506.1 ms | 560.3 ms |

**Speech-to-text is 97.8% of the wall clock at P50.** That is the whole argument
for the architecture, now as a measurement rather than a claim: our half of the
end-to-end path is 2% of it, and 100% of those spoken questions retrieved inside
200 ms with an order of magnitude to spare. No arrangement of our code moves the
other 98%, because it belongs to a vendor on the other side of a network, and
the brief requires that vendor.

Transcription accuracy is a sanity check rather than a claim: the audio is
synthesised, so it is cleaner than a person in a room with traffic outside.
Median content-word overlap between the spoken query and its transcript was
100%, and 25 of 30 transcripts still cleared the guardrails. The latency column
is not flattered the same way - Sarvam is transcribing a real audio file of a
real length over a real network, and that is what the clock measures. Batch STT
is the honest choice for a measurement and it is also the slower one: in the app
the socket is already open and partial transcripts arrive during speech, so the
wait after someone stops talking is shorter than this.

---

## Fifteen languages, asked and measured

The corpus is Hindi. The embedder (`multilingual-e5-small`) is not. That is a
claim about cross-lingual retrieval, so we measured it.

MSMARCO-XI is one set of MS MARCO queries professionally translated into
fourteen Indian languages, keyed by `query_id`. Joining on that key gives
genuine parallel text: the same question, fifteen ways. Machine-translating our
own would have put translation error inside the measurement, where it is
indistinguishable from retrieval error. `pipeline/src/parallel_queries.py`
builds the join by reading three columns over HTTP range requests, which is why
fourteen 460 MB files cost a few MB instead of 6 GB.

`npm run bench:multilingual`, 200 queries per language:

| language | P50 ms | P100 ms | hit@1 | hit@5 | answered |
|---|---|---|---|---|---|
| **हिन्दी** (corpus language) | 2.44 | 12.47 | 27.0% | **59.5%** | 89.0% |
| English | 2.24 | 4.77 | 26.0% | 56.0% | 67.5% |
| नेपाली | 2.39 | 8.65 | 20.0% | 51.5% | 68.0% |
| मराठी | 2.38 | 4.15 | 16.0% | 44.5% | 60.0% |
| മലയാളം | 2.61 | 8.09 | 17.0% | 43.5% | 56.0% |
| اردو | 2.38 | 6.11 | 16.5% | 41.0% | 46.0% |
| বাংলা | 2.48 | 3.95 | 14.5% | 40.5% | 57.0% |
| ಕನ್ನಡ | 2.63 | 7.97 | 17.0% | 40.0% | 53.0% |
| ਪੰਜਾਬੀ | 2.54 | 12.62 | 15.5% | 38.5% | 40.5% |
| ગુજરાતી | 2.52 | 3.92 | 12.5% | 35.5% | 34.0% |
| తెలుగు | 2.55 | 4.73 | 11.5% | 34.0% | 44.0% |
| ଓଡ଼ିଆ | 2.59 | 5.60 | 15.5% | 33.5% | 37.0% |
| தமிழ் | 2.54 | 4.19 | 14.5% | 30.5% | 39.5% |
| संस्कृतम् | 2.61 | 5.58 | 13.5% | 28.0% | 30.5% |
| অসমীয়া | 2.79 | 11.28 | 8.5% | 27.0% | 26.5% |

**Pooled over all 3,000: P50 2.52 ms · P70 2.81 ms · P100 12.62 ms · 0.00% over
budget.**

Two numbers per language, deliberately, because they fail for different reasons:

- **hit@5** is whether *retrieval* ranked the gold passage. Read from
  `RagAnswer.retrieved`, which is populated even on a refusal - otherwise
  "retrieval never found it" and "the gate rejected it" are indistinguishable.
- **answered** is whether the *guardrails* let it through. The confidence
  threshold was fitted on Hindi, so a language whose embeddings sit at
  systematically lower cosine is refused more often **while still ranking
  correctly**.

Cross-lingual retrieval works and degrades smoothly. English sits near the Hindi
ceiling, unsurprising since this corpus was translated *from* English MS MARCO.
Assamese and Sanskrit are the floor, both low-resource in e5's training mix.
Latency is flat across all fifteen.

BM25 helps the corpus language and English, where the shared numbers, Latin
acronyms and proper nouns are. A Tamil question and a Hindi passage share no
tokens at all, so the dense indices carry that result entirely. **The seventh
index is not a cross-lingual gain and we do not claim one.**

Voice input covers the eleven languages Sarvam's recogniser identifies, listed
in the picker. Retrieval covers the fifteen above.

### Three bugs this sweep found

All three are the same shape: a component that was correct for the language it
was written against.

**1. Nine of the fourteen languages were refused before they were read.** Gate 1
tested for letters with `[a-zA-Zऀ-ॿ]` - Latin and Devanagari. Bengali,
Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia and Urdu therefore
contained "no words" and were rejected as GIBBERISH in 0.1 ms, before a vector
existed. A fast, confident refusal is indistinguishable from a working
guardrail, which is exactly why it survived. Now `\p{L}`, plus a regression case
per language.

**2. The not-a-question rule was refusing ordinary questions.** It fired on
"more than eight words and no interrogative marker", which cannot tell a command
from an information-seeking imperative. Across 3,000 parallel queries it refused
98, and those 98 retrieved the gold passage at hit@5 35.7% - the average for
this corpus. The sample said it in one line:

> `Explain what a bone scan is and what it is used for.` → `NOT_A_QUESTION`

Replaced with a positive test for verbs that ask the system to *act* (delete,
run, deploy), exempted when a question marker is present so "how do I delete a
file" stays a question.

**3. The 200 ms guarantee had an unbounded-input hole.** Embedding is the only
stage whose cost is set by the input rather than the corpus, and nothing bounded
it. A 6,594-character query embedded in 219.7 ms and blew the budget outright.
The deadline planner could not help: it degrades *retrieval*, and by the time it
runs the embedding is already paid for. Queries are now clamped to 320
characters before embedding, chosen by measurement rather than by the model's
512-*token* limit, since characters and tokens are not the same across scripts.
The p99 real query is 71 characters.

There was a fourth, in the measuring apparatus itself: the in-app sweep painted
its own progress line inside the first `await` of the next query and charged it
to that query, reporting P100 122 ms against a true 13 ms. It now waits for the
frame to land before starting the clock.

---

## Bring your own sources

Three ways in - paste, file, link - and all of them land in the same place.

| Input | Handled by | Notes |
|---|---|---|
| Pasted text | client | title inferred from the first line |
| `.pdf` | client, `pdf.js` lazy-loaded | detects a missing text layer and says so rather than indexing an empty scan |
| `.docx` | client, `fflate` (~8 KB) | unzips `word/document.xml`; no OOXML library |
| `.txt .md .csv .json .jsonl .html .xml .yaml` + code | client | JSON is flattened to labelled leaves, so `battery_hours: 400` stays findable |
| A URL | Worker | a browser cannot read a cross-origin page body |

What happens to a source, all of it in the browser:

```
extract text → cut into passage-sized units → 6 chunking strategies + BM25
  → embed (same ONNX graph as the corpus, off-thread)
  → PCA-project → binarise + int8 quantise → searchable
```

**Your sources rank against the corpus, not beside it.** Both sides are
projected into the same PCA space and quantised identically, so the 98,867
shipped passages and your PDF compete in a single RRF fusion rather than in two
result lists that have to be reconciled.

**Nothing is uploaded.** Embedding happens in a Web Worker on your machine. The
Worker in the cloud only ever sees a URL you explicitly ask it to fetch.

**It is embedded once.** Quantised vectors are persisted to IndexedDB, so a
400-page PDF costs its indexing time exactly once, not once per visit. Stored
vectors are stamped with the model and PCA geometry and discarded if either
changes - vectors from a different model are not comparable, and silently mixing
them would corrupt ranking with no error anywhere.

**Ingestion cannot slow a query.** This is why the encoder lives in a worker at
all. Embedding a long document is minutes of work, and a query queued behind it
would blow the budget before its first stage started. The main thread holds the
ingestion queue, dispatches one small batch at a time, and stops dispatching
entirely while a query is in flight.

MS MARCO arrives pre-cut at passage size and a PDF does not, so user text is
first cut into passage-sized units on paragraph → sentence → whitespace
boundaries, and only then do the strategies run over those units - exactly as
they run over MS MARCO passages. Without that step, "the same strategies" would
be a claim about function names rather than about behaviour.

Their BM25 index is built in the browser at ingest, because a personal document
needs it more than the corpus does. The shipped corpus is encyclopedia prose,
where a paraphrase usually still finds the passage. A CV, an invoice or a bank
statement is mostly proper nouns, dates and amounts, and the question is usually
about exactly one of them. "How old am I" against `name: priya rao, age: 31` is
a lexical match or it is nothing.

---

## Generation

Retrieval finds the passage that contains the answer. Generation turns it into
one. Ask a CV "what is my name" and retrieval returns:

    name: srinidhi bhat, age: 45, sex: M

That is a search result. The answer is "Your name is Srinidhi Bhat", and getting
from one to the other is the *generation* half of retrieval-augmented
generation. So the retrieved passages go to a model with the question, and what
comes back is checked against them (gates 3a and 3b) before it is allowed to
stand. The passage stays one click under every answer, because a claim about a
document should be checkable against the document.

An answer written by a hosted model means the two or three passages that answer
your question are sent to that provider. Three ways to sit with that:

| Generator | Passages leave the machine? | Speed | Cost |
|---|---|---|---|
| **Groq** (default) | yes, 2-3 passages | ~1000 tok/s | free tier, no card |
| Anthropic | yes, 2-3 passages | ~1-2 s | paid |
| Local (Ollama / llama.cpp) | **no** | your hardware | free |
| none configured | no | - | free |

With nothing configured the app still retrieves and still answers under 200 ms.
It shows the matching passage, *labelled as a quotation* rather than dressed up
as an answer.

### Answers written ahead of time

The shipped corpus never changes, so a question it can answer has one answer
that could have been written at build time. `npm run precompute` walks the
answerable query set through the real pipeline, generates each answer once, and
stores it in `web/public/answers/`. A question that matches one skips the model
round trip; retrieval still runs in full and is still what the figure under the
answer reports, and the message says **written ahead** rather than showing a
generation time that never elapsed.

A stored answer is served only when the query vector is within a high cosine of
a stored one **and** the passage retrieval just ranked first is one that answer
was written from. The second condition is what makes it safe: cosine alone
cannot separate "what is a corporation" from "what is a corporation tax", and it
also covers the case that breaks every naive answer cache - once you add your
own sources, retrieval returns different passages and the stored answer is stale
by definition.

**99 answers are currently stored**, against 6,988 answerable queries in the
set. Filling the rest is one resumable command; measured throughput on the Groq
free tier is ~12.8 queries/minute, so the full set is about nine hours of
wall-clock. `npm run bench:precomputed` verifies the store before shipping it: 7
of 8 stored answers matched their own question exactly, and all 6 decoys -
questions that retrieve well from this corpus but are not the stored one - were
correctly rejected.

---

## Architecture

```
                    ┌───────── measured: 200ms budget ─────────┐
 mic ─► Sarvam STT ─┤  embed → 7 indices + your sources        │─► grounded answer
        (streaming) │  → RRF → rescore → gates → extract       │
        partials ───┘         ALL IN THE BROWSER               │
                    └──────────────────────────────────────────┘
                                     │
                                     └─► (off the clock) tool loop → gate 3 → the answer you read
                                                │        ▲
                                                └────────┘
                                          search_corpus runs HERE,
                                          against the same index
```

**Speculative retrieval:** partial transcripts trigger retrieval while the user
is still speaking, so the answer is warm before they finish. The final
transcript re-runs it for correctness, and the engine serialises the two.

User sources use a **flat** scan rather than IVF. With thousands of chunks
instead of 810k, building clusters would mean running k-means in the browser on
every add: seconds of work to save microseconds of query. The scan is hard-capped
at 150k chunks so a large personal corpus still cannot blow the budget, and the
UI says when that cap was hit rather than being silently partial.

**Three threads, on purpose.** The main thread owns the DOM and the query. The
encoder worker owns ONNX. The fire owns a third: it is a WebGL2 fragment shader
driven from its own worker over an `OffscreenCanvas`, so the thread that runs
queries never participates in a frame. That is not decoration with a nice story
attached - an idle animation on the query thread would land its frames inside
the very budget it exists to advertise.

Where `OffscreenCanvas` transfer or worker construction is blocked, the renderer
falls back in-thread and is explicitly *stopped* for the duration of every
measured query, so the fallback cannot flatter the numbers either. Under
`prefers-reduced-motion` there is no renderer at all, just a static ember.

---

## Running it

```bash
# 1. build the index (one-time, ~90 min of embedding on an M3 Pro)
cd pipeline
uv run python src/acquire.py --parquet data/raw/validation/hinval.parquet --queries 10000
uv run python src/chunking/build.py
uv run python src/embed_all.py
uv run python src/build_index.py
uv run python src/lexical.py          # BM25 index - seconds, no embedding
```

```bash
# 2. build the parallel query set for the multilingual sweep
npm run parallel-queries
```

```bash
# 3. calibrate guardrails, then benchmark
npm install
npm run calibrate              # fits the gate-2 threshold; writes public/thresholds.json
npm run bench                  # P50 / P70 / P100 + retrieval quality
npm run bench:deadline         # proves the cap holds under budget pressure
npm run bench:gates            # 90 guardrail cases, incl. one per language
npm run bench:lexparity        # the browser looks up the terms the builder indexed
npm run bench:ablation         # leave-one-out: what each of the 7 indices is worth
npm run bench:tools            # the tool-call loop, scripted and live
npm run bench:voice            # end to end: real audio in, grounded answer out
npm run bench:usersource       # gate 2 on user documents
npm run bench:multilingual     # 15 languages x 200 queries, cross-lingual hit@k
npm run bench:precomputed      # the stored-answer store matches only its own question
```

```bash
# 4. turn on generated answers - free, ~15 seconds
#    Key from https://console.groq.com/keys, no card required.
echo 'GROQ_API_KEY=gsk_your_key_here' > web/.env.local

# 5. run it
npm run dev                    # http://localhost:5173
```

**No Cloudflare account is needed for step 4.** The dev server serves
`/synthesize` and `/health` itself, from the *same source file* the Worker uses
(`worker/src/synthesize.ts`), so there is no second implementation to drift and
local development exercises the production prompt and wire protocol. The key is
read by `vite.config.ts` in Node and is deliberately not `VITE_`-prefixed, so it
never reaches the browser bundle.

Skip step 4 and everything still runs: retrieval answers in under 200 ms and the
app shows the matching passage, labelled as a quotation.

Voice in, voice out and link-adding are the parts that genuinely need the Worker
- a browser cannot set a custom header on a WebSocket handshake, cannot hold a
speech vendor's key, and cannot read a cross-origin page body. Set
`VITE_WORKER_BASE` in `web/.env.local` to point at a deployed one. Without it
the interface falls back to the browser's own speech APIs, labelled as a
fallback in the UI, so the app is demonstrable with no keys at all.

---

## Deploying

Static site on Cloudflare Pages, one Worker beside it, free tier throughout.
Step-by-step instructions, including how to verify the cap on the live origin,
are in [DEPLOY.md](DEPLOY.md).

```bash
cd worker && npx wrangler deploy          # Worker first - the web build bakes its URL in
npx wrangler secret put GROQ_API_KEY      # + SARVAM_API_KEY

cd .. && echo "VITE_WORKER_BASE=https://chehrag-worker.<subdomain>.workers.dev" >> web/.env.local
npm run deploy:web                        # build + `wrangler pages deploy web/dist`
```

Then set `ALLOWED_ORIGIN` in `worker/wrangler.toml` to the deployed Pages origin
and redeploy the Worker. This matters: `/fetch-url` makes outbound requests, and
leaving CORS open lets any site use your Worker as a fetch proxy on your quota.

Three hosting details that are load-bearing rather than hygiene:

**Cross-origin isolation.** `public/_headers` sets `Cross-Origin-Opener-Policy:
same-origin` and `Cross-Origin-Embedder-Policy: credentialless`. These are what
let onnxruntime use `SharedArrayBuffer` and run the embedding graph on four
threads - worth 35 ms per query, as the table above shows. Without them
transformers.js falls back to single-threaded WASM, which used to be entirely
silent, so boot now checks `crossOriginIsolated` and says so in the console when
it is false. `credentialless` rather than `require-corp` because the model
weights come from the Hugging Face CDN and the fonts from Google, and neither
sets CORP. `vite preview` is configured with the same headers so the built
output can be tested representatively.

**Cache busting.** Index blob paths are not content-hashed, so the loader
appends `?v=<manifest.builtAt>` to every one of them and the manifest itself is
fetched `no-cache`. That is what makes `Cache-Control: immutable` safe:
rebuilding the index changes `builtAt`, which changes every URL. Without it a
rebuild would serve year-old vectors against a new manifest, out of step and
silent.

**SSRF.** `/fetch-url` is the one endpoint that fetches an address a user
supplies. It allows only http/https, refuses private, loopback, link-local and
cloud-metadata addresses, follows redirects manually and re-checks every hop (a
public host is free to redirect to 169.254.169.254), and caps the response size
and time. The Worker holds API keys; it must never become a way to read what is
only reachable from where it runs.

### Cost

| Item | Cost |
|---|---|
| Hosting (Cloudflare Pages, static) | ₹0 |
| Worker (STT proxy, URL fetch, free tier 100k req/day) | ₹0 |
| Embeddings (local ONNX, build time and in-browser) | ₹0 |
| Vector search (in-browser) | ₹0 |
| Sarvam STT | ₹30/hr; ₹100 free credits ≈ 3.3 hrs |
| Answer generation (Groq free tier) | $0 |

Free and fast are the same decision here. Every millisecond spent on a paid API
is a millisecond of the budget gone; local is what makes 200 ms possible.

### What it costs to load

| Asset | Over the wire | Raw |
|---|---|---|
| App shell (HTML 5.1 + CSS 5.8 + JS 44.7, gzipped) | **56 KB** | 160 KB |
| onnxruntime WASM | 5.1 MB | 21.6 MB |
| Model weights (e5-small int8), from the HF CDN | ~135 MB | ~135 MB |
| Index vectors (quantised; compression buys ~1.0-1.2x) | ~66 MB | 75.9 MB |
| Index passage text (brotli 5.7x) | ~14 MB | 77.8 MB |
| `pdf.js` | 129 KB | 433 KB |

A first visit is therefore a couple of hundred megabytes, almost all of it the
model and the index, and it is cached afterwards. Measured on localhost:
**13.1 s cold, 2.6 s warm**. None of it is in the 200 ms path - index load is a
one-time session cost, reported separately and never folded in. It is also why
the first-run curtain names the three things it is waiting for rather than
showing an unlabelled bar.

---

## Notes on scope

- **Latency is measured warm, per query, with the answer cache bypassed.** P100
  is the true maximum over the sample: no trimming, no outlier removal.
- **The 200 ms figure is retrieval through to a grounded extractive answer.**
  Speech-to-text happens before the question exists and a generated answer after
  the extractive one is on screen; both are timed separately and the chip under
  each answer says which is which.
- **The fifteen-language claim is about retrieval, not about the corpus.** The
  corpus is Hindi and every other language is cross-lingual retrieval into it.
  Fifteen languages can be *asked*; they are not fifteen corpora. Voice input
  covers the eleven Sarvam identifies.
- **Script detection is not language detection.** Routing an answer to a voice
  uses the script it is written in, and Devanagari is shared by Hindi, Marathi,
  Nepali and Sanskrit. When Sarvam returns a `language_code` with the transcript
  that is used instead.
- **Speech out uses Sarvam `bulbul:v2`.** An ElevenLabs path is wired for the
  languages it covers and switches on when `ELEVENLABS_API_KEY` is set.
- **The browser speech fallback is not the requirement.** Where no Sarvam key is
  configured the UI uses the browser's own speech APIs so it is still
  demonstrable, and says so in plain text under the voice pickers.
- `sharp` (a transitive dependency of transformers.js, for image input) carries
  high-severity advisories with no upstream fix. We do text only and it never
  reaches the browser bundle.

See [APPROACH.md](APPROACH.md) for a jargon-free explanation and
[CONTEXT.md](CONTEXT.md) for the full decision log.
