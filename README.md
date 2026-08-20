# Chehrag — ask the fire

**चिराग़** *(chirāġ)*, a lamp — crossed with **RAG**. A flame you ask questions
of, that answers from its own corpus or from documents you hand it, grounded in
sources, in **under 200 milliseconds**, in **fifteen languages**.

The entire retrieval pipeline runs client-side. The measured path contains no
network at all.

**HH Goa 2026 — Task 2 submission.**

---

## The core idea

The brief asks for the full pipeline — chunking, vector retrieval, everything
through to final output — in under 200ms.

We measured the obvious architecture before writing code, and it cannot work:

| Component | Cost |
|---|---|
| One network round trip | 50–200 ms |
| Fastest streaming STT on the market (ElevenLabs Scribe v2) | 150 ms typical, ~400 ms p99 |
| LLM answer generation | 500 ms+ |

The fastest speech-to-text in existence spends **150 of the 200ms on its own**.

So we moved the work instead of trying to outrun physics. The index and the
retrieval engine are shipped to the browser once; every subsequent question is
answered on the user's own machine. **The network isn't reduced — it's absent.**

Consequences, all of which we wanted anyway:

- No round trip in the measured path.
- Static hosting: free, global CDN, **never sleeps, no cold starts**. A sleeping
  free tier would put a 30-second first request into P100 and destroy it.
- The system gets faster with more users, since each brings their own CPU.
- Documents a user adds never leave their machine.

---

## How the six requirements are met

| # | Requirement | Where |
|---|---|---|
| 1 | STT via Sarvam or ElevenLabs | **Both, and speech in both directions.** In: **Sarvam** `saaras:v3-realtime` streaming, proxied by a Cloudflare Worker (`web/src/stt/sarvam.ts`). Out: **ElevenLabs** `eleven_flash_v2_5` where it has the language, **Sarvam** `bulbul:v2` for the eight Indic languages it does not (`web/src/tts/speak.ts`) |
| 2 | Chunking must be "vast", not naive fixed-size | **6 strategies**, separately indexed, fused with RRF — applied to the corpus *and* to anything the user adds. `pipeline/src/chunking/strategies.py`, `web/src/sources/chunk.ts` |
| 3 | Full pipeline under 200ms | Runs in-browser. Measured below, and **enforced** by a deadline-aware planner rather than hoped for |
| 4 | P50 / P70 / P100 over many queries | `web/bench/run.ts`, `web/bench/deadline.ts`, `web/bench/multilingual.ts` (3,000 queries across 15 languages), and an in-page benchmark |
| 5 | A real harness | Typed stage pipeline: budgets, retries, degradation, circuit breaker. `web/src/harness/pipeline.ts` |
| 6 | Guardrails | Three gates, threshold **fitted on real labels**. `web/src/guardrails/gates.ts` |

---

## Bring your own sources

Chehrag ships with a corpus, but the point of a lamp is that you get to ask it
about *your* things. Three ways in — **paste**, **file**, **link** — and all of
them land in the same place.

| Input | Handled by | Notes |
|---|---|---|
| Pasted text | client | title inferred from the first line |
| `.pdf` | client, `pdf.js` lazy-loaded | detects a missing text layer and says so rather than indexing an empty scan |
| `.docx` | client, `fflate` (~8 KB) | unzips `word/document.xml`; no OOXML library |
| `.txt .md .csv .json .jsonl .html .xml .yaml` + code | client | JSON is flattened to labelled leaves, so `battery_hours: 400` stays findable |
| A URL | Worker | a browser cannot read a cross-origin page body; the Worker fetches and strips it |

What happens to a source, all of it in the browser:

```
extract text → cut into passage-sized units → 6 chunking strategies
  → embed (same ONNX graph as the corpus, off-thread)
  → PCA-project → binarise + int8 quantise → searchable
```

Three properties this design buys:

**Your sources rank against the corpus, not beside it.** Both sides are
projected into the same PCA space and quantised identically, so the 98,867
shipped passages and your PDF compete in a single RRF fusion under a single
calibrated confidence threshold. There is no "search my documents" toggle
because there does not need to be one.

**Nothing is uploaded.** Embedding happens in a Web Worker on your machine.
The Worker in the cloud only ever sees a URL you explicitly ask it to fetch.

**It is embedded once.** Quantised vectors are persisted to IndexedDB, so a
400-page PDF costs its indexing time exactly once, not once per visit. Stored
vectors are stamped with the model and PCA geometry and discarded if either
changes — vectors from a different model are not comparable, and silently
mixing them would corrupt ranking with no error anywhere.

**Ingestion cannot slow a query.** This is the reason the encoder lives in a
worker at all. Embedding a long document is minutes of work; a query that
queued behind it would blow the budget before its first stage started. The main
thread holds the ingestion queue, dispatches one small batch at a time, and
stops dispatching entirely while a query is in flight.

---

## Chunking: shaped by the data, not by folklore

We measured the corpus first. MS MARCO passages here are **mean 317 / median 295
characters** — they are *already chunk-sized*.

That kills the standard playbook. A 512-token splitter over a 295-char passage
emits one chunk identical to its input: an expensive no-op. On this corpus naive
fixed-size chunking isn't merely naive, it's **inert**.

So the axes that actually pay are different:

| Strategy | Direction | What it wins | What it costs |
|---|---|---|---|
| `whole` | baseline | self-contained answers; strongest single index | one topic-mixed passage averages into one vector |
| `sentence` | finer | precision — a one-sentence answer isn't averaged away | 2.6x vectors; bare sentences lose anaphora |
| `sliding` | finer | answers straddling boundaries; robust to bad punctuation (this corpus is machine-translated) | cuts mid-clause; weakest alone |
| `contextual` | same size, richer | numerics + Latin-script proper nouns, which dense vectors handle worst | header tokens dilute pure-prose queries |
| `semantic` | finer | keeps complete ideas together | **low yield here — median passage is ~3 sentences.** Included and measured rather than assumed |
| `document` | coarser | answers spread across sibling passages | dilute vectors; contributes recall, never precision |

**809,607 chunks from 98,867 passages (8.19x).** Fused with Reciprocal Rank
Fusion, which uses *rank* not score — necessary because a 111-char sentence and a
925-char document are not score-comparable, and averaging their scores would hand
the sentence index a permanent, invisible advantage.

User documents get the same six strategies, with one addition in front: MS MARCO
arrives pre-cut at passage size, and a PDF does not. So user text is first cut
into passage-sized units on paragraph → sentence → whitespace boundaries, and
only then do the six strategies run over those units — exactly as they run over
MS MARCO passages. Without that step, "the same strategies" would be a claim
about function names rather than about behaviour.

---

## Guardrails: three gates, and one of them is fitted

| Gate | Question | Method |
|---|---|---|
| 1 — input | Should we process this at all? | unsafe content, prompt injection (EN + HI), gibberish via character entropy, non-questions |
| 2 — retrieval | Did we actually find anything? | confidence threshold + cross-strategy agreement |
| 3 — grounding | Is every claim traceable to a source? | token-level support check against retrieved context |

Gate 2 is the important one, and it is **calibrated rather than guessed**. The
corpus ships **3,012 queries with no relevant passage** — genuine cases where
refusing is correct. The threshold is fitted on a calibration split and reported
on a holdout that never influenced selection (`web/bench/calibrate.ts`).

The objective is deliberately *not* accuracy. Answering confidently when you
shouldn't is worse than declining too often, so we target abstention precision
and report the recall it costs.

The extractive fast path is **grounded by construction** — every character is
copied from a retrieved passage. Gate 3 exists for the optional LLM-polished
answer, where a fluent model can quietly invent a fact.

---

## The 200ms cap is enforced, not hoped for

Per-stage timeouts tell you a deadline was missed. They do not stop you missing
it. The only way to *guarantee* a cap is for the expensive stage to look at the
clock before it starts and choose an amount of work it can finish.

The query path has one variable cost and several near-constant ones. Embedding
is the variable: ~6ms here, but the same WASM graph on a mid-range Android is
40–70ms and on a thermally-throttled laptop can spike past 100ms. So retrieval
is told what is *left* after embedding and sizes itself to it, giving up
capability in a fixed order (`web/src/harness/deadline.ts`):

| Knob | Given up | Why in this order |
|---|---|---|
| `nprobe` | first | linear in cost, sublinear in recall — the first clusters hold most of the answer |
| `perStrategyK` | second | narrows what fusion has to work with |
| `rescoreTopN` | last | gate 2's calibrated threshold reads the rescored score |

**How we tested it without a slow phone.** We shrank the budget instead.
Squeezing the pipeline into 20ms on this machine exercises the exact code path a
10x slower device hits at 200ms. `web/bench/deadline.ts` runs the full query set
at nine budgets and reports both whether the deadline held *and* what recall it
cost — a cap met by returning nothing useful is not a cap worth having.

```
 budget      p50     p100   over  degraded   kept@3     n
   200ms    3.99     7.60      0        0%   100.0%   122
   100ms    3.91     8.09      0        0%   100.0%   122
    50ms    3.88     7.44      0        0%   100.0%   122
    30ms    3.80    10.59      0        0%   100.0%   122
    20ms    3.82     9.88      0        0%   100.0%   122
    15ms    3.85     7.46      0        0%   100.0%   122
    12ms    3.03     8.24      0       95%    93.4%   122
    10ms    2.63    11.86      2       95%    69.7%   122   <- first breach
```

**Zero overruns from 200ms down to a 12ms budget** — the system tolerates a
machine roughly 17x slower than this one and still meets the requirement. Below
~10ms the budget is smaller than a single embedding forward pass, which no
retrieval plan can recover; that is the floor of the technique, and it is
reported rather than trimmed off the bottom of the table.

**This benchmark earned its keep.** It found two real bugs that the
fast-machine numbers could never have surfaced, because on a fast machine
neither code path ever executes:

1. `rescore` reserved 12ms for an operation costing **0.03ms**. Under a tight
   budget it was therefore skipped — and gate 2 then compared its calibrated
   cosine threshold (0.4788) against the raw Hamming proxy, a different scale
   entirely, refusing almost everything. Fixed on both sides: the reservation is
   now proportionate, and if rescoring is ever skipped the gate drops the score
   test rather than applying a threshold fitted on a different quantity.
2. `embed` carried `retries: 1`. On a timeout it retried with no budget left,
   doubling the spend at exactly the moment the budget was tightest. The
   pipeline now refuses to start an attempt it cannot finish.

---

## Architecture

```
                    ┌───────── measured: 200ms budget ─────────┐
 mic ─► Sarvam STT ─┤  embed → 6 corpus indices + your sources │─► grounded answer
        (streaming) │  → RRF → rescore → gate 2 → extract      │
        partials ───┘         ALL IN THE BROWSER               │
                    └──────────────────────────────────────────┘
                                     │
                                     └─► (off the clock) LLM polish → gate 3 → shown if it passes
```

**Speculative retrieval:** partial transcripts trigger retrieval *while the user
is still speaking*, so the answer is warm before they finish. The final
transcript re-runs it for correctness.

Two-stage retrieval keeps the worst case bounded:
1. **binary codes + Hamming** over IVF clusters — wide net, capped candidates
2. **int8 dot product** on fused passage candidates only — real similarity for
   ranking and for the confidence threshold

User sources use a **flat** scan rather than IVF — with thousands of chunks
instead of 810k, building clusters would mean running k-means in the browser on
every add, seconds of work to save microseconds of query. The scan is hard-capped
at 150k chunks so a large personal corpus still cannot blow the budget, and the
UI says when that cap was hit rather than being silently partial.

**Three threads, on purpose.** The main thread owns the DOM and the query. The
encoder worker owns ONNX. **The fire owns a third**: it is a WebGL2 fragment
shader driven from its own worker over an `OffscreenCanvas`, so the thread that
runs queries never participates in a frame.

That is not decoration-with-a-nice-story, it is the only arrangement that keeps
the number honest — an idle animation on the query thread would land its frames
inside the very budget it exists to advertise. Verified rather than asserted:
with the fire rendering continuously, **`requestAnimationFrame` fires zero times
on the main thread over a 2-second window**.

Where `OffscreenCanvas` transfer or worker construction is blocked, the renderer
falls back in-thread and is explicitly *stopped* for the duration of every
measured query, so the fallback cannot flatter the numbers either. Under
`prefers-reduced-motion` there is no renderer at all, just a static ember.

---

## Measured results

### Latency — requirements 3 & 4

**In-browser** (Chrome, M3 Pro, warm index, cache bypassed, distinct queries):

| | dev, n=120 | production build, n=80 |
|---|---|---|
| **P50** | **7.71** | **8.46** |
| **P100 (worst query)** | **9.47** | **23.05** |
| under 200ms | 100% | 100% |

**Node harness** (n=300, same modules, native onnxruntime). Reported across
**five consecutive runs**, because one run would misrepresent the tail:

| | ms |
|---|---|
| **P50** | **3.9 – 4.1** |
| **P70** | **4.1 – 4.4** |
| P90 / P95 / P99 | 4.5–5.1 / 4.7–5.7 / 5.3–8.1 |
| **P100 (worst of 300)** | **5.9 – 15.9** |
| mean | 3.8 – 4.1 |

**The tail is noisy and we are not going to pretend otherwise.** P50 is stable
to within 0.2ms across runs; P100 varies 2.7x. Every bit of that spread is a
single outlier inside one stage — `embed` p100 ranged 3.85 to 13.48 ms across
the same five runs while its p50 never moved from ~1.8. That is a GC or
scheduler event landing on one query out of three hundred, not a code path, and
the honest way to report it is a range rather than the prettiest run.

It also does not threaten anything: the worst P100 we recorded across five runs
is **8% of the budget**.

Per stage (Node, representative run, p50 / p100 ms): `embed 1.85 / 3.85` ·
`retrieve 1.98 / 4.32` · `guard:input 0.01 / 1.24` · `rescore 0.03 / 0.15` ·
`guard:retrieval 0.03 / 0.10` · `answer:extract 0.00 / 0.02`

Searching all six indices costs **~2ms**. Query embedding dominates the budget.
The browser and Node numbers differ because Node uses native onnxruntime while
the browser uses WASM; **the browser number is the one that counts**, since that
is where the system runs.

Refused queries short-circuit: an injection attempt is rejected at **0.1ms**
without spending an embedding.

Two changes since the previous submission halved the browser tail:
- **The encoder moved to a Web Worker.** Ingestion and UI work no longer share
  an allocation arena with ONNX, and P100 went from 20.18ms to 9.47ms.
- **Warm-up got serious.** The first query costs ~20ms of embed against ~6ms
  once warm, so an under-warmed engine sets P100 with its own first request.
  Warm-up now covers both scripts and a range of lengths, and costs ~150ms of a
  multi-second load — the cheapest tail reduction anywhere in this system.

### Multilingual — 3,000 queries, 15 languages

The corpus is Hindi. The embedder (`multilingual-e5-small`) is not. That is a
claim about cross-lingual retrieval, so we measured it instead of repeating it.

MSMARCO-XI is one set of MS MARCO queries professionally translated into
fourteen Indian languages, keyed by `query_id`. Joining on that key gives
**genuine parallel text** — the same question, 15 ways. Machine-translating our
own would have put translation error inside the measurement, where it is
indistinguishable from retrieval error. `pipeline/src/parallel_queries.py`
builds the join by reading only three columns over HTTP range requests, which is
why fourteen 460 MB files cost a few MB instead of 6 GB.

`npm run bench:multilingual` — 200 queries per language, Node harness:

| language | P50 ms | P100 ms | hit@1 | hit@5 | answered |
|---|---|---|---|---|---|
| **हिन्दी** (corpus language) | 3.85 | 7.82 | 27.5% | **58.5%** | 90.0% |
| English | 3.72 | 6.16 | 25.0% | 54.5% | 73.5% |
| नेपाली | 3.81 | 8.46 | 19.5% | 48.0% | 71.5% |
| മലയാളം | 5.57 | 9.70 | 16.0% | 41.5% | 63.5% |
| বাংলা | 3.99 | 22.38 | 15.5% | 40.5% | 66.5% |
| اردو | 4.00 | 21.48 | 16.5% | 40.0% | 56.0% |
| ಕನ್ನಡ | 5.53 | 10.51 | 17.0% | 39.0% | 64.0% |
| मराठी | 5.13 | 9.99 | 14.5% | 39.0% | 65.5% |
| ਪੰਜਾਬੀ | 5.98 | 24.72 | 15.0% | 37.5% | 50.0% |
| తెలుగు | 3.97 | 5.94 | 11.0% | 34.0% | 52.5% |
| ગુજરાતી | 4.00 | 5.85 | 13.0% | 33.0% | 41.5% |
| தமிழ் | 4.00 | 5.38 | 15.0% | 31.0% | 49.5% |
| ଓଡ଼ିଆ | 5.56 | 13.44 | 14.0% | 30.0% | 48.0% |
| संस्कृतम् | 4.49 | 9.45 | 12.0% | 26.0% | 33.0% |
| অসমীয়া | 4.20 | 12.46 | 8.0% | 25.5% | 32.5% |

**Pooled: P50 3.95 ms · P70 5.12 ms · P100 24.72 ms · 0.00% over budget.**

Two numbers per language, deliberately, because they fail for different reasons:

- **hit@5** is whether *retrieval* ranked the gold passage. It is read from
  `RagAnswer.retrieved`, which is populated even on a refusal — otherwise
  "retrieval never found it" and "the gate rejected it" are indistinguishable.
- **answered** is whether the *guardrails* let it through. The confidence
  threshold was fitted on Hindi, so a language whose embeddings sit at
  systematically lower cosine gets refused more often **while still ranking
  correctly**. Assamese ranks at 44% of Hindi's hit@5 but answers at 36% of its
  rate — the gate, not the retriever, is the larger part of that gap.

The honest summary: **cross-lingual retrieval works and degrades smoothly.**
English is nearly at the Hindi ceiling — unsurprising, since the corpus is
translated *from* English MS MARCO. Assamese and Sanskrit are the floor, both
low-resource in e5's training mix. Nothing falls over, and latency is flat
across all fifteen.

### Three bugs this sweep found

None of these were visible from the outside. All three are the same shape: a
component that was correct for the language it was written against.

**1. Nine of the fourteen languages were refused before they were read.**
Gate 1 tested for letters with `[a-zA-Z\u0900-\u097F]` — Latin and Devanagari.
Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia and Urdu
therefore contained "no words" and were rejected as **GIBBERISH in 0.1 ms**,
before a vector existed. A fast confident refusal is indistinguishable from a
working guardrail, which is exactly why it survived. Now `\p{L}`, plus a
regression case per language in `bench/gatetest.ts`.

**2. The not-a-question rule was refusing ordinary questions.** It fired on
"more than eight words and no interrogative marker", which cannot tell a command
from an information-seeking imperative. Measured across 3,000 parallel queries it
refused 98 — and those 98 retrieved the gold passage at hit@5 **35.7%**, the
average for this corpus. The sample said it in one line:

> `Explain what a bone scan is and what it is used for.` → `NOT_A_QUESTION`

Replaced with a positive test for verbs that ask the system to *act* (delete,
run, deploy…), exempted when a question marker is present so "how do I delete a
file" stays a question. Hindi's answered rate went 79% → 90%.

**3. The 200 ms guarantee had an unbounded-input hole.** Embedding is the only
stage whose cost is set by the *input* rather than the corpus, and nothing
bounded it. A 6,594-character query embedded in **219.7 ms** and blew the budget
outright — P100 216 ms. The deadline planner could not help: it degrades
*retrieval*, which costs about a millisecond, and by the time it runs the
embedding is already paid for.

Two fixes, because there were two faults:
- **Queries are clamped to 320 characters** before embedding, chosen by
  measurement rather than by the model's 512-*token* limit — characters and
  tokens are not the same across scripts, and 512 characters of Assamese sat
  right on the embed stage's budget. p99 of a real query is 71 characters.
- **A timeout is no longer retried.** Retries exist for transient faults; a
  stage that timed out was handed deterministic work that did not fit, and
  running it again produces the identical overrun at double the price. That is
  how a merely-slow query became a 122.9 ms *failure*. `bench/deadline.ts` now
  holds the cap down to an **8 ms** budget, where it previously breached at 10.

There was a fourth, in the measuring apparatus itself: the in-app sweep painted
its own progress line inside the first `await` of the next query and charged it
to that query, reporting P100 122 ms against a true 13 ms. **The reporting UI was
nine tenths of the number it was reporting.** It now waits for the frame to land
before starting the clock.

### Retrieval quality

Brute-force ceiling on 500 gold-labelled queries: fp32-384 R@10 = 0.758.

Quantisation ladder (measured, not assumed):

| config | R@10 | codes |
|---|---|---|
| fp32 384 (ceiling) | 0.758 | — |
| PCA-256 fp32 | 0.744 | 25.9 MB |
| → binary-256 | 0.604 | 25.9 MB |
| **binary-384, no PCA** | **0.228** | 38.9 MB |

**PCA is not compression here — it is what makes binary quantisation work at
all.** Sign bits over raw embedding dimensions are near-worthless because those
dimensions are correlated and off-centre. Dropping PCA to "keep more information"
would have destroyed retrieval.

End to end on the held-out query set (n=300): **hit@1 0.32, hit@3 0.54** over
187 graded answerable queries.

### Guardrails — requirement 6

Gate 1 and gate 3: **25/25 unit tests pass**, including the false-positive guards
— "How do vaccines work?", "what caused the second world war", and "how to treat a
snake bite" all correctly pass. The filter blocks operational harm requests, not
subject matter. Self-harm routes to a support line rather than a bare refusal.

Gate 2, on the MS MARCO answerable/unanswerable split (n=300):
coverage of answerable **87.0%**, abstention precision 0.53, recall 0.36, F1 0.43.

**Why those abstention numbers are modest, stated plainly.** MS MARCO's
"unanswerable" queries still carry ten topically-relevant passages retrieved by
the original search engine — they are on-topic but non-answering. Measured
separability is genuinely weak: **AUC 0.6497**, and no combination of margin,
strategy-agreement or lexical overlap beat it.

Gate 2's real production job is easier and it does that well. On genuinely
out-of-corpus questions ("who won the 2029 world cup on mars", "what is my bank
balance"), the fitted threshold **blocks 7 of 8 while keeping 84% of in-corpus
queries answerable**.

The threshold is selected by an explicit **coverage policy** (answer ≥85% of
answerable queries), not by maximising F1 — F1 chose 0.68 and produced a system
that refused 71% of real questions. Technically optimal, practically useless.

---

## Running it

```bash
# 1. build the index (one-time, ~90 min of embedding on an M3 Pro)
cd pipeline
uv run python src/acquire.py --parquet data/raw/validation/hinval.parquet --queries 10000
uv run python src/chunking/build.py
uv run python src/embed_all.py
uv run python src/build_index.py
```

```bash
# 2. build the parallel query set for the multilingual sweep
#    (reads 3 columns of 14 files over range requests — a few MB, not 6 GB)
npm run parallel-queries
```

```bash
# 3. calibrate guardrails, then benchmark
npm install
npm run calibrate              # fits the gate-2 threshold; writes public/thresholds.json
npm run bench                  # P50 / P70 / P100 + retrieval quality
npm run bench:deadline         # proves the cap holds under budget pressure
npm run bench:gates            # 42 guardrail cases, incl. one per language
npm run bench:multilingual     # 15 languages x 200 queries, cross-lingual hit@k
```

```bash
# 4. run it
npm run dev                    # http://localhost:5173
```

Voice in, voice out, link-adding and the LLM rewrite need the Worker; typed
questions do not. Copy `web/.env.example` to `web/.env` and set
`VITE_WORKER_BASE` to enable them. Without it the interface falls back to the
browser's own speech APIs, **labelled as a fallback in the UI** — that exists so
the app is demonstrable with no keys, not to satisfy the requirement.

---

## Deploying

Static site on Cloudflare Pages, one Worker beside it. Free tier throughout.
**Step-by-step instructions, including how to verify the 200 ms cap on the live
origin, are in [DEPLOY.md](DEPLOY.md).** The short version:

```bash
cd worker && npx wrangler deploy          # Worker first — the web build bakes its URL in
npx wrangler secret put SARVAM_API_KEY    # + ELEVENLABS_API_KEY, + ANTHROPIC_API_KEY (optional)

cd .. && echo "VITE_WORKER_BASE=https://chehrag-worker.<subdomain>.workers.dev" > web/.env
npm run deploy:web                        # build + `wrangler pages deploy web/dist`
```

Then set `ALLOWED_ORIGIN` in `worker/wrangler.toml` to the deployed Pages origin
and redeploy the Worker. This matters: `/fetch-url` makes outbound requests, and
leaving CORS open lets any site use your Worker as a fetch proxy on your quota.

Per-query latency does not depend on any of this. Nothing leaves the browser
during a measured query, so hosting cannot make a query faster or slower — with
exactly one exception, which is the first of the three notes below.

Three hosting details that are load-bearing rather than hygiene:

**Cross-origin isolation.** `public/_headers` sets `Cross-Origin-Opener-Policy:
same-origin` and `Cross-Origin-Embedder-Policy: credentialless`. These are what
let onnxruntime use `SharedArrayBuffer` and run the embedding graph on multiple
threads. Without them transformers.js falls back to single-threaded WASM and
the dominant cost in the query budget roughly triples. That used to be entirely
silent, which made it the worst kind of regression — the app works, it is just
three times slower for no visible reason — so boot now checks
`crossOriginIsolated` and says so in the console when it is false.
`credentialless` rather than `require-corp` because
the model weights come from the Hugging Face CDN and the fonts from Google, and
neither sets CORP. `vite preview` is configured with the same headers so the
built output can be tested representatively.

**Cache busting.** Index blob paths are not content-hashed, so the loader appends
`?v=<manifest.builtAt>` to every one of them and the manifest itself is fetched
`no-cache`. That is what makes `Cache-Control: immutable` safe: rebuilding the
index changes `builtAt`, which changes every URL. Without it a rebuild would
serve year-old vectors against a new manifest — out of step, and silent.

**SSRF.** `/fetch-url` is the one endpoint that fetches an address a user
supplies. It allows only http/https, refuses private, loopback, link-local and
cloud-metadata addresses, follows redirects manually and **re-checks every hop**
(a public host is free to redirect to 169.254.169.254), and caps the response
size and time. The Worker holds two API keys; it must never become a way to read
what is only reachable from where it runs.

### Cost

| Item | Cost |
|---|---|
| Hosting (Cloudflare Pages, static) | ₹0 |
| Worker (STT proxy, URL fetch, free tier 100k req/day) | ₹0 |
| Embeddings (local ONNX, build time and in-browser) | ₹0 |
| Vector search (in-browser) | ₹0 |
| Sarvam STT | ₹30/hr; ₹100 free credits ≈ 3.3 hrs |
| LLM polish (optional, off the fast path) | a few $ |

**Free and fast are the same decision here.** Every millisecond spent on a paid
API is a millisecond of the budget gone; local is what makes 200ms possible.

---

## What it costs to load

The honest counterweight to a 9ms query. Nothing here is in the measured path,
but it is real and a judge on hotel wifi will feel it.

| Asset | Transferred | Note |
|---|---|---|
| App shell (JS + CSS + HTML) | **~31 KB gz** | was 249 KB before the encoder moved off-thread |
| onnxruntime WASM | 5.1 MB gz | 21.6 MB raw; fetched by the encoder worker |
| Model weights (e5-small int8) | ~35 MB | Hugging Face CDN, browser-cached |
| Index vectors | ~53 MB | quantised data; brotli buys ~1.0–1.2x, as expected |
| Index passage text | **~13 MB** | 74 MB raw, brotli 5.7x — Cloudflare compresses JSON automatically |
| `pdf.js` | 129 KB gz | **only** if you add a PDF |

Boot is **~1.0s warm** (blobs from IndexedDB) and ~9s cold on localhost with no
network cost at all. The main bundle shrank 94% when the transformers.js import
in the in-thread fallback was made dynamic — it was pulling the whole library
into the main chunk purely to have a fallback ready, on top of the copy already
in the worker.

---

## Honest notes

- **Latency is measured warm, per query, with the cache bypassed.** Index load is
  a one-time session cost and is reported separately, never folded in.
- **Speech is an external service, in both directions**, with its own latency.
  STT happens before the question exists and TTS after the answer is already on
  screen; neither is inside the measured span, and the chip under each answer
  says so. They are reported against their own published figures, never hidden
  inside ours.
- **The 15-language claim is about retrieval, not about the corpus.** The corpus
  is Hindi. Every other language is cross-lingual retrieval *into* it, and the
  table above is what that actually costs — from 93% of the Hindi ceiling for
  English down to 44% for Assamese. Fifteen languages can be *asked*; they are
  not fifteen corpora.
- **Script detection is not language detection.** Routing an answer to a voice
  uses the script it is written in, and Devanagari is shared by Hindi, Marathi,
  Nepali and Sanskrit — so it cannot separate them and does not claim to. When
  Sarvam returns a `language_code` with the transcript, that is used instead: a
  model that heard the speaker knows more than any heuristic.
- **P100 is the true maximum** over the sample. No trimming, no outlier removal.
- The `semantic` strategy has low yield on this corpus. We report that rather
  than quietly dropping it.
- **Abstention on the MS MARCO unanswerable split is weak (AUC 0.65).** We report
  the number and the reason instead of quietly switching to an easier test set.
- **Gate 2's threshold was fitted on MS MARCO prose.** A user source that is
  structured rather than prose — a config dump, a table — sits closer to that
  threshold than an article does. We did not re-tune the threshold per source,
  because there are no labels to tune it against, and inventing one would be
  worse than the honest behaviour of declining a weak match.
- First visit downloads ~100 MB across model, WASM and index. Cached afterwards,
  and **not** in the 200ms path — but it is a real first-load cost.
- **The fire is decoration with a constraint.** It carries seven states and a
  microphone level, and it renders in its own worker on an `OffscreenCanvas`,
  because in this project an animation on the query thread would be a
  correctness bug. Measured: zero main-thread animation frames while it burns.
- **The browser speech fallback is not the requirement.** Where no Sarvam or
  ElevenLabs key is configured, the UI uses the browser's own speech APIs so it
  is still demonstrable — and says so in plain text under the voice pickers, so
  you always know which of the two you are hearing.
- `sharp` (a transitive dep of transformers.js, for image input) carries
  high-severity CVEs with no upstream fix. We do text only and it never reaches
  the browser bundle.

See `APPROACH.md` for a jargon-free explanation, and `CONTEXT.md` for the full
decision log including two silent cross-language bugs we caught with parity tests.
