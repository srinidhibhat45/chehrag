# Getting Chehrag online

Target: a public URL where **P100 stays under 200 ms**, the number that is
actually being graded.

---

## The one thing that can break the budget in production

Per-query latency does **not** depend on hosting. Retrieval, ranking and
grounding all run in the visitor's browser; no request leaves the machine during
a measured query, so a server in Mumbai and a server in Iowa produce identical
numbers. Latency here is a property of the build, not of the network.

There is exactly one exception, and it is a header:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

These are what make `SharedArrayBuffer` available, which is what lets
onnxruntime run the embedding graph on more than one thread. `embed` is the
dominant stage in the query (p50 1.9 ms of a 4.0 ms query, p100 13.5 ms of a
15.6 ms one). Without the headers it silently falls back to a single thread and
roughly triples — **nothing errors, nothing logs, the app just gets slower than
it was on localhost.**

`web/public/_headers` sets them and Cloudflare Pages reads that file
automatically. That is the entire reason the host below is Cloudflare Pages
rather than anything else. If you move to another host, porting this file is the
migration — everything else is static assets.

The app now warns in the console when it boots without isolation, so this can
never fail silently again. **Step 5 verifies it on the live origin.**

---

## Prerequisites

- Node 20+
- A Cloudflare account (free tier is enough for all of this)
- `npx wrangler login` — opens a browser for OAuth, one time

One of these turns on written answers; the rest are for voice. Retrieval works
fully without any of them — it just quotes the matching passage instead of
answering from it, and says so.

| Key | Buys |
|---|---|
| `GROQ_API_KEY` | **written answers.** Free tier, no card, ~1000 tok/s — https://console.groq.com/keys |
| `SARVAM_API_KEY` | real speech input (streaming), and spoken answers in the 8 Indic languages ElevenLabs lacks |
| `ELEVENLABS_API_KEY` | spoken answers in English, Hindi, Tamil |

Instead of `GROQ_API_KEY` you may set `ANTHROPIC_API_KEY` (paid, better at
combining several passages), or `LLM_BASE_URL` + `LLM_MODEL` for any other
OpenAI-compatible endpoint including a local one. `worker/src/synthesize.ts`
resolves them in that order.

---

## Step 1 — push the code

```bash
git push -u origin main
```

The repo includes `web/public/index/` (~131 MB of index blobs). That is
deliberate: it is the only thing standing between a fresh clone and a working
deploy, and rebuilding it needs the Python pipeline, a 111 MB corpus subset and
a 578 MB model, none of which are in the repo. The largest single file is
24.1 MiB — under GitHub's 100 MB limit and under Cloudflare's 25 MiB one, with
about 1.9 MiB of headroom.

The first push moves ~131 MB and will take a few minutes.

---

## Step 2 — deploy the Worker

Do this **before** the site: the web build bakes the Worker's URL in at build
time, so the site needs the URL to exist first.

```bash
cd worker
npm install          # worker/ has its own lockfile — it is not an npm workspace
npx wrangler deploy
```

That prints the URL — `https://chehrag-worker.<your-subdomain>.workers.dev`.
Keep it.

Now attach the keys. Secrets take effect immediately; no redeploy needed.

```bash
npx wrangler secret put GROQ_API_KEY           # written answers — free tier
npx wrangler secret put SARVAM_API_KEY         # voice in
npx wrangler secret put ELEVENLABS_API_KEY     # voice out
```

Check what it thinks it has:

```bash
curl https://chehrag-worker.<your-subdomain>.workers.dev/health
```

The app calls this same endpoint rather than assuming that a configured Worker
URL means voice works. "A Worker is deployed" and "Sarvam is reachable with a
valid key" are different claims and it only makes the second one when `/health`
says so.

Nothing the Worker does is on the measured path — it exists for four things a
browser is not allowed to do: set a custom header on a WebSocket handshake
(Sarvam STT), hold the TTS keys off the client, read a cross-origin page body,
and hold the Anthropic key.

---

## Step 3 — deploy the site

```bash
cd ..
echo "VITE_WORKER_BASE=https://chehrag-worker.<your-subdomain>.workers.dev" >> web/.env.local
npm install
npm run build
npx wrangler pages deploy web/dist --project-name chehrag --branch main
```

No trailing slash on the URL. The first run offers to create the Pages project —
accept it.

You get `https://chehrag.pages.dev`. Uploading ~131 MB of index blobs takes a
few minutes the first time; subsequent deploys only send changed files.

`npm run deploy:web` is the same thing as a one-liner once the project exists.

---

## Step 4 — close the CORS hole

`worker/wrangler.toml` ships with `ALLOWED_ORIGIN = "*"`, which is right for
local development and wrong now. `/fetch-url` makes outbound requests on your
behalf, so an open policy lets any site on the internet use your Worker as a
fetch proxy on your quota.

```toml
[vars]
ALLOWED_ORIGIN = "https://chehrag.pages.dev"
```

Exact origin — scheme included, no path, no trailing slash. Then:

```bash
cd worker && npx wrangler deploy
```

---

## Step 5 — verify the budget on the live origin

This is the step that matters. Do not skip it: everything above can succeed
while the thing being claimed is false.

Open the deployed site, wait for the lamp to finish lighting, open the browser
console and paste:

```js
const { engine } = window.chehrag;
const shard = await (await fetch("/index/passages.000.json")).json();
const qs = shard.slice(0, 600).map(p => p.split(/\s+/).slice(0, 8).join(" "));
for (let i = 0; i < 30; i++) await engine.ask(qs[i], { skipCache: true });
const t = [];
for (let i = 0; i < 500; i++) {
  t.push((await engine.ask(qs[(i + 30) % qs.length], { skipCache: true })).totalMs);
}
t.sort((a, b) => a - b);
const pct = p => t[Math.min(t.length - 1, Math.floor(p / 100 * t.length))];
console.table({
  crossOriginIsolated,
  p50: +pct(50).toFixed(2),
  p95: +pct(95).toFixed(2),
  P100: +t.at(-1).toFixed(2),
  over_200ms: t.filter(x => x >= 200).length,
});
```

Two things must be true:

- **`crossOriginIsolated: true`.** If it is `false`, the `_headers` file is not
  reaching the browser and every number below it is a single-threaded number.
  The console will already be carrying the warning.
- **`over_200ms: 0`**, and `P100` in the same range as the local benchmark
  (15.6 ms on an M3 Pro). Judge hardware will be slower; the headroom is ~13x,
  so it has a long way to fall before it matters.

The snippet writes nothing to the DOM on purpose. `totalMs` is wall-clock across
an `await` on the encoder worker and cannot tell a repaint from retrieval — the
in-app sweep this replaces once reported P100 122 ms against a true 13 ms
because it painted a progress line inside the first `await` of the next query.

To confirm the headers independently of the app:

```bash
curl -sI https://chehrag.pages.dev | grep -i cross-origin
```

---

## Optional — deploy on every push

Once Step 3 has created the project, Cloudflare Pages can build from GitHub
instead of from your laptop:

Cloudflare dashboard → Workers & Pages → `chehrag` → Settings → Builds →
Connect to Git → pick `srinidhibhat45/chehrag`.

| Setting | Value |
|---|---|
| Production branch | `main` |
| Build command | `npm install && npm run build` |
| Build output directory | `web/dist` |
| Environment variable | `VITE_WORKER_BASE` = your Worker URL |

`VITE_WORKER_BASE` has to be set in the dashboard, not just in `web/.env.local` —
`.env` is gitignored, and without it a Git-driven build ships a site with no
voice and no link-fetching.

---

## Optional — custom domain

Pages → `chehrag` → Custom domains → add. Cloudflare issues the certificate.
Then update `ALLOWED_ORIGIN` in `worker/wrangler.toml` to the new origin and
redeploy the Worker, or the link-fetching breaks on the new domain.

---

## What a first visit actually costs

None of this is in the 200 ms path — it happens once, before any query — but it
is real, and someone on conference wifi will feel it.

| Asset | Transferred | From |
|---|---|---|
| App shell (JS + CSS + HTML) | ~31 KB gz | your origin |
| onnxruntime WASM | 5.1 MB gz | your origin |
| Model weights (e5-small int8) | ~35 MB | Hugging Face CDN |
| Index vectors | ~53 MB | your origin, ~1.1x compressible |
| Index passage text | ~13 MB | your origin, 74 MB raw, brotli 5.7x |

Roughly **~100 MB cold, ~1 s warm.** Index blobs go into IndexedDB keyed by the
manifest's build timestamp; model weights go into the browser's Cache API. The
second visit fetches neither.

The model is the one asset served by someone else. It could be self-hosted to
remove the third-party dependency, but not from Pages — `model_quantized.onnx`
is 118 MB and the per-file limit is 25 MiB, so it would need an R2 bucket with
CORS and CORP headers. Left as-is deliberately: it is a cold-start dependency,
not a query-path one.

---

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Queries ~3x slower than local, no errors | not cross-origin isolated | `curl -sI <url> \| grep -i cross-origin`. `_headers` must land at the root of `web/dist` — confirm it survived the build |
| `File too large` during `pages deploy` | a blob went over 25 MiB | reshard: raise the shard count in `pipeline/src/build_index.py` and rebuild |
| Mic button stays disabled | no `SARVAM_API_KEY` and a non-Chromium browser | set the key, or use Chrome/Edge for the browser fallback |
| Adding a link fails, console shows CORS | `ALLOWED_ORIGIN` doesn't match | exact origin, no trailing slash, then redeploy the Worker |
| Answers never spoken | no TTS key | check `/health`; the note under the voice pickers says which voice will speak |
| Boot hangs at "Lighting the lamp…" | an index blob 404s | check `/index/manifest.json` loads, and that `web/public/index/` was actually committed |
| Stale answers after rebuilding the index | never happens by design | every blob URL carries `?v=<manifest.builtAt>`; the manifest itself is fetched `no-cache` |
