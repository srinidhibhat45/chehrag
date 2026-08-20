# CONTEXT — Chehrag (HH Goa 2026, Task 2)

Working memory for this build. Update as decisions land. Not for the team — see `APPROACH.md` for that.

---

## The ask (6 requirements)

| # | Requirement | Where it's satisfied |
|---|---|---|
| 1 | Speech-to-text via Sarvam **or** ElevenLabs | Sarvam STT (streaming) + ElevenLabs/Sarvam TTS, all proxied through CF Worker |
| 2 | Chunking strategy must be "vast" — not naive fixed-size | 5 strategies + RRF fusion across 5 indices |
| 3 | Full pipeline < 200ms | Runs in-browser; zero network in measured path |
| 4 | Report P50 / P70 / P100 over many queries | `bench/` harness, per-stage breakdown |
| 5 | Proper harness — orchestration, retries, structured I/O, error recovery | `web/src/harness/` typed stage pipeline |
| 6 | Guardrails — off-topic, unsafe, hallucination, ungrounded | `web/src/guardrails/`, 3 gates |

## Locked architecture

**Browser-first. The 200ms path never touches the network.**

```
mic → Sarvam STT (streaming, partials)
        ↓ transcript
    [BROWSER — the measured 200ms path]
      embed query (ONNX, local)
      → search 5 indices (int8, in-memory)
      → RRF fusion
      → guardrail gates
      → extractive answer
        ↓ (async, OFF the fast path)
    LLM synthesis via CF Worker → streamed in after
```

**Why browser-first:** eliminates the network round trip entirely — not measured around it, genuinely absent. Also means static hosting: free, global CDN, never sleeps, no cold starts. A judge's laptop is faster than any free cloud box.

### Hosting (chosen for £0 + never-sleeps)
- **Cloudflare Pages** — static app + index shards. Free, unlimited bandwidth, global CDN.
- **Cloudflare Worker** — only for (a) hiding the Sarvam API key, (b) optional LLM synthesis. Free tier 100k req/day.
- No always-on server. Nothing to keep alive. Nothing to cold-start.

### Rejected alternatives (don't revisit)
- **HF Spaces** — Docker/Gradio Spaces now require paid PRO (checked 2026-08). Only static is free.
- **Oracle Always Free** — halved to 2 OCPU/12GB in June 2026, capacity lottery.
- **Render/Railway/Fly free** — 512MB–1GB RAM (index won't fit) and they sleep. Sleep = 30s first request = P100 destroyed.
- **Sharding one query across clouds** — tail amplification: waiting on the slowest of N makes P100 strictly worse. Never do this.

## Machine (build box)

M3 Pro · 11 cores (5P/6E) · **18 GB RAM** · 188 GB free · Node v26.2.0 · Python via `uv` (system 3.9 too old)

18GB is the binding constraint on the offline embed job. Batch accordingly.

## Dataset

`ai4bharat/MSMARCO-XI` — **11.45M rows, 55.6 GB, 14 Indic languages.**
HF dataset viewer is crashing (`JobManagerCrashedError`) → stream parquet, infer schema ourselves.

Fields: `query`, `Answer`, `passages{is_selected, English_passages, Translated_passages}`, `Eng_Query`, `Eng_Answer`, `query_id`, `query_type`, `source_lang`, `target_lang`.

**Scope decision:** do NOT index all 14 languages. `passages` is nested per query → heavy duplication; dedupe hard.

## Budget

| Item | Cost |
|---|---|
| Sarvam STT | ₹30/hr, ₹100 free credits ≈ 3.3 hrs. Test suite ≈ 40 min. |
| Embeddings | local ONNX, ₹0 |
| Vector search | in-browser, ₹0 |
| Hosting | Cloudflare Pages + Workers free tier, ₹0 |
| LLM synthesis (off fast path) | Haiku 4.5 @ $1/$5 per Mtok, few $ total |

Target total: **under $5.**

## Key constraint to never forget

**P100, not P50.** The *worst* query must be under 200ms. That kills:
- anything on the network in the fast path
- cold starts / lazy init
- allocation during a query
- unbounded candidate sets

Everything in the fast path must be warmed at load and allocation-free per query.

---

## Status

- [x] Machine + tooling surveyed
- [x] Architecture locked
- [x] uv / Python 3.12 env
- [x] Dataset acquisition + subset — 10k queries, 98,867 passages
- [x] Chunking strategies — 6 strategies, 809,607 chunks
- [x] Embedder verified (drift + throughput measured)
- [x] Embedding run — 809,607 vectors in 58.2 min
- [x] PCA 384->**256** (192 rejected: only 0.80 variance), quantization, IVF build
- [x] Browser index format + loader (sharded passages)
- [x] Harness — stage budgets, retries, degradation, circuit breaker
- [x] Guardrails — 25/25 gate tests pass; gate 2 calibrated on holdout
- [x] Bench — **P50 7.71 / P100 9.47 ms in-browser, 100% under 200ms**
- [x] STT client (Sarvam streaming + batch fallback) — written, needs live key to verify
- [x] Running locally, verified in browser (dev and production build)
- [x] User sources — paste / file / URL, chunked+embedded client-side, fused with the corpus
- [x] **App starts empty** — the shipped corpus is opt-in, so an answer's origin is
      unambiguous. Gate 1 refuses "nothing loaded" in 0.1ms as its own reason
- [x] **English on user sources** — the MS MARCO-fitted absolute threshold refused
      25% of answerable English questions; a measured mid-band lexical rescue
      (user sources only) takes coverage 75% -> 96.4% with no new false answers,
      and leaves the corpus's calibrated numbers identical to the query
- [x] Deadline planner + `bench/deadline.ts` — cap holds to an **8ms** budget, 0 overruns
- [x] Hosting config — `_headers`, cache busting, SSRF guards, deploy scripts
- [x] Fireball — WebGL2 shader in a worker; 0 main-thread frames, measured
- [x] Layout — three panes (sources / conversation / studio); latency is a chip per answer
- [x] TTS — ElevenLabs `eleven_flash_v2_5` + Sarvam `bulbul:v2`, routed by language
- [x] Multilingual stress test — 15 languages x 200 queries; found 5 bugs
- [x] **Deployed — https://chehrag.pages.dev** (Worker: `chehrag-worker.chehrag-worker.workers.dev`)

## Measured facts (don't re-derive)

**Corpus (Hindi validation split)**
- Source file 97,941 rows -> sampled 10,000 queries -> 98,867 unique passages
- 6,988 answerable / 3,012 unanswerable. The unanswerable slice IS the abstention test set.
- Passage length: mean 317 / median 295 chars. **Already chunk-sized** — this is why
  naive fixed-size chunking is inert here, and it drove the whole strategy design.
- query_type mix: DESCRIPTION 54%, NUMERIC 25%, ENTITY 9%, PERSON 6%, LOCATION 6%
- Telugu has NO train file (13 train vs 14 validation files). Not a bug in our code.

**Chunking output**
| strategy | chunks | x/passage |
|---|---|---|
| whole | 98,867 | 1.00 |
| sentence | 252,597 | 2.55 |
| sliding | 250,240 | 2.53 |
| contextual | 98,867 | 1.00 |
| semantic | 75,788 | 0.77 |
| document | 33,248 | 0.34 |
| **TOTAL** | **809,607** | **8.19** |

**Embedder — `Xenova/multilingual-e5-small`, 384-dim**
- Same ONNX weights in Python build and browser query. Non-negotiable: mismatched
  weights degrade retrieval silently.
- Quantized-vs-fp32 drift: cosine **0.996** mean, **100% top-10 rank agreement**.
  => build with int8 weights; fp32 build time buys nothing.
- Throughput (M3 Pro): int8+length-sorted **157/s** | int8 unsorted 97/s | fp32 69/s
  | CoreML EP slower (74 graph partitions).
- **Single query encode: 2.2 ms** (int8). This is the number that matters for the budget.
- e5 REQUIRES `query: ` / `passage: ` prefixes.

**Storage budget (809,607 chunks)**
- binary @384bit = 38.9 MB | int8 @384d = 310.9 MB | fp32 = 1.24 GB
- Plan: PCA 384->192. binary@192bit = 19.4 MB, int8 parents only = 19 MB, text gz = 23 MB
- **Target first load ~62 MB**, cached in IndexedDB. Not in the 200ms path (one-time).

## Open questions

- Corpus size target — tuning against browser download budget. Starting hypothesis: ~250k passages ≈ 100–150 MB int8.
- Cross-encoder rerank in-browser: may be too slow. Fallback = drop it, land ~15ms.

---

## Investigation log

### Quantized-model batch dependence (resolved — no action needed)

Symptom: transformers.js (browser) and Python onnxruntime produced vectors at
cos 0.9966, despite loading a byte-identical `model_quantized.onnx`
(sha256 `f80102d3…` verified on both sides).

Chased it down:
- python **batched** vs python **single** : cos 0.9966  <- the real source
- python **single**  vs js **single**     : cos 0.9993  <- essentially identical
- Not padding: an exact-length (zero-padding) batch still drifted (0.9977).

Cause: **dynamic quantization computes activation scales across the whole
tensor**, so batch composition changes rounding. Inherent to the export; not
fixable by batching strategy.

Decision — measured rather than argued. A/B on 5,950 passages / 600 gold-labelled
queries, query side always q8-single (what the browser does):

| index build | build time | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| q8 batched   |  62.3s | 0.3467 | 0.7733 | **0.8867** |
| fp32 batched | 141.7s | 0.3550 | 0.7733 | 0.8850 |

Within noise at n=600; fp32 costs 2.3x build time for no measurable gain.
**Keep q8 batched.** Document the discrepancy rather than pay for it.

**Baseline to beat: R@10 = 0.887** (brute-force, whole-passage only, closed pool).
Multi-strategy fusion has to improve on this or it isn't earning its complexity.

### Tooling notes
- npm workspaces would not hoist to root; bench lives in `web/bench/` so module
  resolution works. Don't move it back.
- `sharp` has high-severity CVEs — transitive dep of transformers.js for IMAGE
  input only. We do text only and it never reaches the browser bundle. No fix
  available upstream; accepted.

### Cross-language parity bugs (found by targeted tests, both fixed)

The Python builder writes binary data that JS reads. Every such boundary is a
silent-failure surface: wrong layout still produces numbers, just wrong ones.
Two tests now guard them (`web/bench/bitparity.ts`, `web/bench/pcaparity.ts`).

**BUG 1 — `popcnt` integer overflow in `ivf.ts` (severity: total).**
Original used `(x * 0x01010101) >> 24`. In JS `*` yields a double; for large
inputs the product exceeds 2^31 and the following `>>` does a lossy ToInt32,
returning a wrong (often negative) popcount. Hamming distances came out with
deltas up to 2.07e9. Retrieval would have run without error and ranked by noise.
Fix: `Math.imul(x, 0x01010101) >>> 24`, and `>>>` for the intermediate shifts.
**Verified: 0/64 Hamming mismatches vs numpy after the fix.**

**BUG 2 — Node Buffer pooling in the test harness (severity: test-only).**
`readFileSync(p).buffer` returns the whole POOLED ArrayBuffer, not the file's
region. Must use `new T(b.buffer, b.byteOffset, b.byteLength / BPE)`.
This bites the bench loaders too — they already slice correctly.

PCA projection parity: worst cosine 0.9999999, max abs 1.94e-7. Clean.

### Deployment constraint discovered
Cloudflare Pages rejects any file over **25 MiB**. Hindi passage text is ~91 MB
raw (Devanagari is 3 bytes/char in UTF-8). Passage text is therefore SHARDED into
~8 MB files, listed in `manifest.passageShards`, fetched in parallel at boot and
concatenated **in order** (ordinals are positional — never sort or race them).
The builder asserts each shard is under the limit.


---

## Second pass — Chehrag (2026-08-20)

Renamed from "Voice RAG". चिराग़ = lamp; Cheh + RAG. The lamp is the interface.

### Encoder moved to a Web Worker

**Why:** not raw speed — contention. Ingesting a user document is minutes of
embedding. A query issued during that would queue behind hundreds of batches and
blow the budget *before its first stage started*, which no per-stage budget can
rescue because the time is spent before the stage exists.

Scheduling lives on the main thread deliberately: the worker handles messages
serially, so priority has to be enforced by the sender. It holds the ingestion
queue, dispatches one small batch (8 passages, ~25ms) at a time, and stops
dispatching while a query is in flight. Worst case a query inherits ~25ms of a
200ms budget, and only if it arrives mid-batch. `ask()` also pauses bulk
dispatch outright for its duration.

**Measured side effect, unplanned:** browser P100 fell 20.18 -> 9.47 ms. ONNX
allocations no longer share a GC arena with the UI.

**Fallback kept.** Worker construction fails under some CSPs and privacy
extensions. `createEncoder()` falls back in-thread. That fallback's
transformers.js import is **dynamic** — a static one pulled the whole 870 KB
library into the main chunk purely to have a fallback ready, on top of the copy
already in the worker chunk. Main bundle: 924 KB -> 57.7 KB.

### Warm-up is load-bearing, and 2 samples was not enough

First query ~20ms embed vs ~6ms warm. Warm-up now runs 8 queries spanning both
scripts and a range of lengths — tokenisation and sequence length are what the
graph specialises on, so warming only on short ASCII leaves the Devanagari path
cold. Costs ~150ms of a multi-second load. Node harness P100: 14.7 -> 7.1 ms.

### Deadline planner (`src/harness/deadline.ts`)

Per-stage timeouts detect a miss; they don't prevent one. `budgetPlan()` reads
`ctx.remaining()` after embedding and picks a retrieval plan that fits, giving
up `nprobe` first (linear cost, sublinear recall), then `perStrategyK`, then
`rescoreTopN` last.

`rescoreTopN` is last for a specific reason: gate 2's threshold was fitted on
the **rescored cosine**. Drop rescoring and the gate compares 0.4788 against a
Hamming proxy on a different scale.

### `bench/deadline.ts` — and the two bugs it found

Can't borrow a slow phone, so shrink the budget instead: 20ms here exercises the
path a 10x slower device hits at 200ms. Nine budgets, reporting both whether the
deadline held and what recall it cost.

**Result: 0 overruns from 200ms down to 12ms.** First breach at 10ms (2/150),
below one embedding forward pass. ~17x slower machine still meets the brief.

**BUG 3 — `rescore` reserved 12ms for a 0.03ms operation.** Under a tight budget
`minRemainingMs: 12` skipped it, gate 2 then thresholded the Hamming proxy with
a cosine threshold, and the system refused nearly everything (kept@3 -> 0%).
Fixed both halves: reservation is now 2ms, and if rescoring is ever skipped the
gate drops the score test rather than applying a threshold fitted on a different
quantity — and marks the answer degraded so it is never passed off as calibrated.

**BUG 4 — `embed` retried with no budget left.** `retries: 1` plus a timeout
means the second attempt spends time that was already gone, overshooting 2x at
exactly the tightest moment. `Pipeline` now refuses to start an attempt when
under `RETRY_MIN_REMAINING_MS`. Removed the last overruns at 12–15ms.

**Bench methodology bug (mine, caught before reporting):** first version used
`DEFAULT_THRESHOLDS` (0.80 / 2 agreement — the conservative placeholder for
"calibration never run") instead of the fitted `thresholds.json`. The corpus
refused nearly everything, so kept@3 was computed over **n=4** and looked like a
measurement. Always load what ships.

### User sources

`src/sources/` — ingest -> passages -> 6 strategies -> embed -> quantise -> flat index.

- **Flat, not IVF.** Thousands of chunks, not 810k. Clustering would mean k-means
  in the browser on every add: seconds of work to save microseconds of query,
  plus recall loss on the corpus the user cares most about. Scan hard-capped at
  150k chunks (~2ms) and the UI reports when the cap truncated a search.
- **Passage-forming step in front of the six strategies.** MS MARCO arrives
  pre-cut at ~317 chars; a PDF does not. Without this, "the same strategies"
  would be a claim about function names, not behaviour.
- **`USER_BASE = 10_000_000`** offsets user passages in the shared id space.
  Fixed offset, not a tagged pair: one integer compare in the hot loop.
- **Quantisation must be byte-identical to `build_index.py`** (`sources/quantise.ts`).
  Both corpora fuse in one RRF pass under one calibrated threshold; different
  quantisation on either side means the guardrail reads two incompatible scales.
- **Persistence stores the quantised form**, not floats — 32x smaller for codes,
  4x for passage vectors, and exactly what search needs, so reload replays
  straight in. Stamped with model + dim + codeWords; mismatches are discarded
  rather than replayed, because vectors from a different model are not comparable.
- **BUG 5 — `extractJson` dropped keys for non-string scalars.** `battery_hours: 400`
  was indexed as a bare `400`. Bare numbers are unfindable — no question embeds
  near them — so exactly the facts people ask about were the ones that couldn't be
  retrieved. Confirmed by probe: the query refused at conf 0.412, and 0.667 after
  the fix. Keys now carried for every scalar, nested paths dotted.

### Hosting

- **COOP/COEP are load-bearing, not hygiene.** Cross-origin isolation is what
  lets ORT use SharedArrayBuffer; without it transformers.js silently drops to
  single-threaded and the dominant budget cost triples with nothing in the
  console. `credentialless`, not `require-corp` — HF CDN and Google Fonts set no
  CORP. Mirrored into `vite preview` so the built output tests representatively.
- **Cache busting.** Index paths aren't content-hashed, so the loader appends
  `?v=<manifest.builtAt>` and fetches the manifest `no-cache`. That is what makes
  `immutable` safe; without it a rebuild serves old vectors against a new
  manifest, silently out of step.
- **SSRF on `/fetch-url`.** http/https only; private, loopback, link-local and
  metadata addresses refused; redirects followed manually with **every hop
  re-checked** (a public host may redirect to 169.254.169.254); size and time
  capped. The Worker holds two API keys.
- **Compression measured, not assumed:** passages JSON brotli **5.74x** (74 -> 13 MB);
  binary blobs 1.00–1.18x. Quantised data is near-random and does not compress —
  so precompressing the blobs would have been wasted build time.

### Loader memory

Switched passage shards back to `r.json()` from `r.text()` + `JSON.parse`. The
text form holds a 74 MB string alongside the parsed array at peak; on a 4 GB
phone that headroom isn't free. Cost: exact transferred size is no longer
observable, so progress is charged per shard against the estimate — which is all
a progress bar needs.

### UI constraint worth keeping

The lamp animates **only** `transform`, `opacity` and registered custom
properties, all compositor-side. In this project a main-thread idle animation
would be a correctness bug: its frames would land inside the budget it exists to
advertise. Mic amplitude is the one JS-driven part, and it runs only while
recording — never during a measured query.


---

## Third pass — multilingual + the fire (2026-08-20)

### The orb is now a real shader, in its own worker

The old orb was a radial-gradient with a CSS blur behind it and read as exactly
that: a flat disc in fog. Fire is legible because of *structure* — turbulent
filaments that curl and burn out — and a gradient has no structure at any scale,
so no amount of tuning was going to get there.

`src/ui/fire.ts` is a WebGL2 fragment shader. Three things carry it:
turbulence (`sum |noise|`, whose creases are the filaments) rather than smooth
fbm; domain warping, which is what makes filaments shear instead of scroll; and
a blackbody-ish ramp built from overlapping smoothsteps so brightness and hue
move together the way they do in a flame.

**It renders in a worker over an `OffscreenCanvas`.** This is the same
constraint as before, satisfied properly rather than by restricting what CSS is
allowed to animate. `requestAnimationFrame` *is* available in a dedicated worker
wherever `OffscreenCanvas` is, so the loop stays vsync-locked — this is not a
`setTimeout` approximation. Verified: **0 main-thread rAF callbacks over 2s**
while the fire renders continuously.

Fallbacks, in order: worker+OffscreenCanvas -> in-thread WebGL (explicitly
*stopped* during every measured query, so it cannot flatter the numbers) ->
static CSS ember. `prefers-reduced-motion` goes straight to the ember.

**Shader bugs worth remembering:**
- The turbulence term has a positive mean, so ungated it lifted every pixel on
  the canvas slightly above zero. Invisible on its own — but the alpha feather
  cut it off at a fixed radius, and the cut read as a **hard circular outline**
  drawn around the fireball. Fixed by gating turbulence with a radial `env`, so
  the fire is genuinely absent where there is no fire rather than
  present-but-clipped.
- Every falloff was written `smoothstep(hi, lo, x)`. That works on this GPU but
  GLSL leaves `edge0 >= edge1` **undefined**. Rewritten as `1.0 - smoothstep(lo,
  hi, x)`. A shader that only renders correctly on the machine it was written on
  is a bug waiting for someone else's laptop.
- Backticks inside a GLSL comment terminate the JS template literal holding the
  shader. Obvious in hindsight; the error points at the shader, not the string.

### Layout: three panes, and the stopwatch moved

Sources | conversation | studio, after Gemini/NotebookLM. Sources are a
permanent rail rather than a modal drawer, because in a notebook the sources are
the subject, not a setting — you want to switch one off and re-ask without
losing your place.

Answers are a **thread**, not a result panel that overwrites itself. The
interesting comparisons here are *between* answers (this one refused and that
one didn't; this took 6ms and that 40), and a self-replacing panel destroys the
only evidence for judging that.

The latency reading is now **a chip in each answer's footer**, expanding to the
stage breakdown in place. It was a whole tab, which has the pathology of any
metrics tab: the number lives somewhere you have to go and look, at which point
it is no longer attached to the thing it is a fact about.

### Speech out — and why BOTH providers

The brief says Sarvam *or* ElevenLabs. Using one would mean losing something:

- ElevenLabs `eleven_flash_v2_5`: ~75ms model latency, 32 languages — but of the
  fourteen in MSMARCO-XI it has only Hindi and Tamil.
- Sarvam `bulbul:v2`: native Indic, covers Bengali, Gujarati, Kannada,
  Malayalam, Marathi, Odia, Punjabi, Telugu — exactly the set ElevenLabs lacks.

So `src/tts/speak.ts` routes by language. Not hedging: it is the only
arrangement under which every language the corpus contains can be spoken.

The browser's `speechSynthesis` sits under both as a labelled fallback, so the
app is demonstrable with no keys. It is never counted as satisfying requirement
1, and `/health` — not the presence of a Worker URL — decides what the note
under the voice pickers claims.

### Multilingual stress test — `bench/multilingual.ts`

MSMARCO-XI is the same MS MARCO queries in 14 languages keyed by `query_id`, so
joining on that key gives **real parallel text**. Machine-translating our own
would have put translation error inside the measurement, where it cannot be told
apart from retrieval error.

`pipeline/src/parallel_queries.py` reads only `query_id`, `query`, `Eng_Query`.
Parquet is columnar, so those three columns over HTTP range requests skip the
`passages` column that is ~95% of each 460MB file — 14 languages for a few MB.
All 6,988 answerable query ids joined in all 14 languages.

**Scored twice, deliberately.** hit@k answers "did retrieval rank the gold
passage" (read from `RagAnswer.retrieved`, which is now populated even on a
refusal); answered@ answers "did the guardrails allow it". A language can score
well on the first and badly on the second — the threshold was fitted on Hindi —
and one number would hide that in the flattering direction.

Results: pooled P50 3.95 / P100 24.72ms, 0% over budget across 3,000 queries.
hit@5 from 58.5% (Hindi, the ceiling) through English 54.5% down to Assamese
25.5%. Degrades smoothly; nothing falls over.

### BUG 6 — nine languages refused before they were read (severity: total)

Gate 1 tested for letters with `[a-zA-Z ऀ-ॿ]`. In a system advertising fourteen
Indian languages, **nine of them contained "no words"** and were rejected as
GIBBERISH in 0.1ms, before any vector existed.

It survived because a fast confident refusal is indistinguishable from a working
guardrail. Nothing errored; the numbers looked *good*, because refusals are
cheap. Now `\p{L}` — the general fix, so the next script added does not need
this line edited — plus one real query per language in `bench/gatetest.ts`.

### BUG 7 — the not-a-question rule refused ordinary questions

The rule was "more than 8 words and no interrogative marker". That cannot
distinguish a command from an **information-seeking imperative**, which is one
of the commonest query shapes:

    "Explain what a bone scan is and what it is used for."  ->  NOT_A_QUESTION

Measured: 98 refusals across 3,000 parallel queries, and those 98 retrieved gold
at hit@5 **35.7%** — the average for this corpus. It was refusing perfectly good
questions and the interrogative lists it consulted only covered two scripts.

Replaced with a **positive** test for verbs that ask the system to act
(delete/run/deploy/...), exempted when a question marker is present so "how do I
delete a file" stays a question. Interrogative markers now exist for all ten
scripts, used to *exempt* rather than to condemn. Hindi answered 79% -> 90%.

### BUG 8 — the 200ms guarantee had an unbounded-input hole (severity: high)

Embedding is the only stage whose cost is set by the **input** rather than by the
corpus, and nothing bounded it:

      30 chars ->   2.2 ms   <- p50 query
    2834 chars ->  78.4 ms
    6594 chars -> 219.7 ms   <- P100 216ms. Budget blown.

The deadline planner cannot save this. It degrades *retrieval*, worth about a
millisecond, and by the time it runs the embedding is already paid for. A bound
has to be applied **before** the cost is incurred, not after.

`MAX_QUERY_CHARS = 320`, set by measurement and not by e5's 512-*token* limit —
characters and tokens are not the same across scripts, and 512 characters of
Assamese sat right on the embed stage's budget. p99 of a real query is 71
characters. Truncates rather than refuses (this is a voice interface) and
reports it in the plan, because a silently shortened question is a wrong answer
waiting to happen.

### BUG 9 — a timeout was retried, turning slow into failed

`retries: 1` on embed applied to **timeouts** as well as faults. A stage that
timed out was handed deterministic work that did not fit; running it again
produces the identical overrun at double the price, and then `onError: "FAIL"`
aborts the pipeline. A long Assamese query embedded in ~61ms against a 60ms
budget -> timeout -> retry -> another ~61ms -> **122.9ms and an error**. Worst of
both outcomes, and the single worst number in a 3,000-query sweep.

`withTimeout` now throws a tagged `StageTimeout`, and the pipeline never retries
one. Retries are for transient faults; a deadline miss is not a fault.

**Side effect:** `bench/deadline.ts` now holds the cap down to an **8ms** budget.
It previously breached at 10ms (2/150). The floor moved because the retries were
what was breaching it.

### BUG 10 — the benchmark UI was most of what it measured

The in-app multilingual sweep painted a progress line and then immediately
started a query. `totalMs` is wall-clock on the main thread across an `await` on
the encoder worker, and it cannot tell a repaint from retrieval — so the paint
landed inside the first query and was charged to it. **Reported P100 122ms
against a true 13ms.**

Fixed by waiting for the frame to land before starting the clock. And the naive
fix — `requestAnimationFrame(() => setTimeout(r, 0))` — **deadlocked the sweep in
a background tab**, because browsers stop servicing rAF entirely while hidden.
Now raced against a 250ms timeout: when nothing is being painted there is no
paint to wait for, which makes falling through correct rather than a compromise.

### Query-side facts worth not re-deriving

- Real query length: **p50 30 / p95 55 / p99 71 chars**, max 6,594 (degenerate
  repeated translations, 0.1% of rows).
- Embed cost by clamped length, worst script, Node / in-browser (~4x, WASM):
  256 -> 7.6 / ~30ms · 320 -> 9.3 / ~37ms · 512 -> 15.3 / ~60ms.
- Cross-origin isolation confirmed live in `vite preview`
  (`crossOriginIsolated === true`, SAB present) — so the browser/Node gap is
  WASM overhead, not a silently single-threaded ORT.

---

## Fourth pass — the first-run curtain (2026-08-20)

### The first visit was 35 seconds of a dead interface

Measured on a genuinely cold cache (IndexedDB and cache storage cleared,
production build): **34.9 s to ready.** For all of it the visitor saw a lamp
with `disabled` on it, a "Type instead" button that opened a text field which
refused keystrokes, and an "Add" button wired to nothing — `SourcesPanel` is
not constructed until the encoder and the store exist. The only account of any
of it was eleven pixels of grey byte counter under the lamp.

A disabled control with no stated reason is indistinguishable from a broken
one, and there were three at once.

`src/ui/curtain.ts` + markup in `index.html`. Modal because the block is real:
it names the three things `boot()` actually awaits, in the shape it awaits them
(index and model in parallel, then warm-up), and lifts *after* the inputs are
enabled rather than before.

`.app` carries `inert` from the initial HTML rather than from JS, so there is
no frame in which the dead controls are focusable. Where `inert` is
unsupported the curtain still covers and pointer-blocks; only tabbing behind it
survives, which is the right thing to lose first.

### The model was a 135 MB download that reported nothing

Found by watching the first version: the bar froze at 88% for the entire
encoder load, because only the index had progress and the model — the other
half of the parallel pair — had none. That reproduced the exact confusion the
curtain exists to remove, one step further in.

transformers.js takes a `progress_callback`; it was simply never passed. Now
plumbed out of `encoder.worker.ts` as an unsolicited `op: "progress"` message
(`id: 0`, routed on `op` before the pending-request lookup, or it is dropped
silently), summed across the model's files, throttled to 10/s.

Consequences worth keeping:

- **The bar is split 0.44 / 0.44 / 0.12** across index, model and warm-up, not
  weighted by bytes. Byte weighting makes the bar's rate a property of the
  cache state rather than of the boot, and a bar only one of two parallel
  downloads can move freezes whenever the other is the slow half.
- **A phase that completes without ever reporting a byte jumps to 1.** A warm
  cache fetches nothing and so reports nothing; absence of progress is not
  zero progress.
- **Bytes finishing is not the phase finishing.** ONNX still builds the session
  and runs a warm-up inference after the last byte, so the row switches from
  `135 of ~135 MB` to `preparing` rather than sitting on a completed-looking
  count that has not ticked.

### Two failure paths, because they fail differently

- `boot()` rejecting is reported *on the curtain*, not only on the orb — the
  orb is behind it, and an error the user cannot see is the same as no error.
- The module never running at all (chunk 404, CSP, parse error) cannot be
  reported by `curtain.ts`, which is the thing that did not load. A classic
  inline `<script>` in `index.html` handles that one, keyed on a `data-wired`
  flag the Curtain constructor sets — not on a timer against the download,
  because a cold first visit is legitimately long and must not trip it.

Plus a stall detector: 30 s with no byte and no phase change says the wait has
become abnormal and offers Reload. It does not claim failure — the fetch may
still be alive.

Cost: +3.0 kB JS, +3.1 kB CSS, +4.2 kB HTML.

### Live on Cloudflare (2026-08-20)

`./scripts/deploy.sh` → verified by `scripts/verify-deploy.sh`: COOP + COEP
present, index and largest blob served, passage text brotli'd, Worker healthy,
and a real cross-lingual generation over the wire.

**Measured on the live origin, not locally:**

| | p50 | p70 | p95 | P100 | over 200 ms |
|---|---|---|---|---|---|
| localhost | 9.42 | — | — | 21.58 | 0 / 500 |
| chehrag.pages.dev | 9.39 | 10.05 | 12.38 | 25.77 | 0 / 500 |

`crossOriginIsolated === true` live, so the threaded WASM path is real and the
numbers above are not single-threaded ones.

Cold first visit on the live origin: **32.4 s** — index and the 135 MB model in
parallel, both now reported by the curtain.

**Benchmark methodology note, same family as BUG 10.** The first two live runs
read p50 50 / P100 137 and looked like the deployment was 5x slower than local.
It was not: both were taken immediately after the tab was activated, and the
activation's repaint and worker resume landed inside the measured span —
`totalMs` is wall-clock across an `await` on the encoder worker and cannot tell
them apart. A control run on localhost under identical conditions is what
separated "the deployment is slow" from "the measurement is". **Never read a
percentile from a tab that just changed visibility state.**

ElevenLabs remains unconfigured (`/health` → `elevenlabs: false`); Sarvam covers
TTS for every language in the corpus, so this is a missing alternative rather
than a missing capability.
