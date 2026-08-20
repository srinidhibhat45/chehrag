/**
 * Answer the shipped corpus offline, so the generation hop is paid once.
 *
 * The corpus never changes, so every question it can answer has one answer that
 * could have been written at build time. This runs the real pipeline over the
 * shipped query set — the same `RagEngine`, the same retrieval, the same prompt
 * out of `worker/src/synthesize.ts` — and writes the results into
 * `public/answers/` for `src/answer/precomputed.ts` to read back.
 *
 * ── what parallelism actually buys, measured ─────────────────────────────
 *
 * Splitting the work across N workers is the obvious move and on a free Groq
 * key it buys nothing at all. One request takes ~540 ms, so serial execution
 * already offers ~111 requests/minute against a 30 rpm cap: the limiter binds
 * long before latency does, and the eleventh worker waits exactly as long as
 * the second. 6,988 answerable queries take ~3.9 hours either way.
 *
 * Where it does pay is a paid tier or a self-hosted endpoint (`LLM_BASE_URL`),
 * where the ceiling is high enough that round-trip latency is the constraint
 * again. So the scheduler below is written for that case and simply idles
 * against the limiter on the free one — `--workers` is the knob, and leaving it
 * at 1 costs nothing on a free key.
 *
 * ── the budget is tokens, not requests ───────────────────────────────────
 *
 * Both limits bind, and the token one binds first on this workload: a prompt is
 * the system rules (~500 tok) plus three passages (~400) plus the question,
 * against ~80 tokens out, so 30 requests/minute is already ~30,000 tokens per
 * minute. Packing by request count alone would overshoot the token ceiling on
 * long passages and undershoot it on short ones, so `TokenBucket` below meters
 * both and every job is charged its own estimate.
 *
 *   npx tsx bench/precompute.ts --limit 200        try it on a slice first
 *   npx tsx bench/precompute.ts                    the whole answerable set
 *
 * Resumable: results are checkpointed, and a re-run skips what is already done.
 */

import { readFileSync, writeFileSync, appendFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG, USER_BASE } from "../src/harness/rag";
import { gateGrounding } from "../src/guardrails/gates";
import { synthesizeStream, resolveProvider } from "../../worker/src/synthesize";

const INDEX_DIR = process.env.INDEX_DIR ?? "public/index";
const QUERIES = process.env.QUERIES ?? "../pipeline/data/subset/queries.hin.jsonl";
const OUT_DIR = process.env.OUT_DIR ?? "public/answers";
const CHECKPOINT = join(OUT_DIR, ".progress.jsonl");

const argv = process.argv.slice(2);
const argOf = (name: string, fallback: number): number => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? Number(argv[i + 1]) : fallback;
};
const LIMIT = argOf("--limit", Infinity);
const WORKERS = argOf("--workers", 1);
const RPM = argOf("--rpm", 30);
/**
 * Measured against a free key, not taken from the docs: the model reports
 * `service tier on_demand ... tokens per minute (TPM): Limit 8000`. Requests
 * per minute are nowhere near binding at that ceiling — ~1,050 tokens a request
 * puts the token budget at about 7 requests a minute, a quarter of the 30 rpm
 * the request limit would allow.
 */
const TPM = argOf("--tpm", 8_000);

/**
 * A refilling budget, used once for requests and once for tokens.
 *
 * Continuous refill rather than a fixed window: a window lets a burst spend the
 * whole allowance in its first second and then stall, which reads to the
 * provider as exactly the spike the limit exists to stop. Capacity is one
 * minute's worth, so a genuinely idle start can still burst once.
 *
 * The estimate this is charged against is only ever approximate, so `penalise`
 * exists for when the provider disagrees: a 429 spends the remaining budget
 * outright, which stalls every worker at once instead of letting each discover
 * the limit separately.
 */
class TokenBucket {
  private available: number;
  private last = Date.now();

  constructor(private readonly perMinute: number) {
    this.available = perMinute;
  }

  private refill(): void {
    const now = Date.now();
    this.available = Math.min(
      this.perMinute,
      this.available + ((now - this.last) / 60_000) * this.perMinute,
    );
    this.last = now;
  }

  /** Wait until `cost` is available, then spend it. */
  async take(cost: number): Promise<void> {
    const want = Math.min(cost, this.perMinute);
    for (;;) {
      this.refill();
      if (this.available >= want) { this.available -= want; return; }
      const deficit = want - this.available;
      await sleep(Math.max(50, (deficit / this.perMinute) * 60_000));
    }
  }

  /** The provider says we are over. Drop what is left so everyone waits. */
  penalise(): void {
    this.refill();
    this.available = 0;
  }
}

/**
 * A 429 is not a failure, it is the limiter working.
 *
 * The token estimate is approximate by construction, so overshooting is
 * expected and the response is to wait and repeat rather than to discard the
 * query. Backoff is exponential and the token budget is zeroed alongside it, so
 * a burst of 429s across several workers collapses into one shared pause
 * instead of each worker retrying into the same wall.
 */
const MAX_RETRIES = 5;

/**
 * `synthesizeStream` never throws — it converts a provider failure into an
 * `{error}` event, because the browser it was written for always has an
 * extractive answer to fall back on and would rather degrade than crash. So the
 * rate-limit case arrives here as the string `describe()` produced for it, and
 * matching on that is working with the protocol rather than around it.
 */
const RATE_LIMITED = "rate limited";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Tokens a prompt will cost, erring high.
 *
 * Not a real tokenizer: the generator's is not the retriever's, and being
 * roughly right is enough to pace a rate limiter. Devanagari is charged at two
 * characters per token because byte-pair vocabularies trained mostly on Latin
 * text split it far harder than they split English, and undercounting here
 * would mean overshooting the provider's ceiling rather than merely running
 * slower than necessary.
 */
function estimateTokens(text: string): number {
  let latin = 0, other = 0;
  for (const ch of text) (/[\x00-\x7F]/.test(ch) ? latin++ : other++);
  return Math.ceil(latin / 4 + other / 2);
}

interface QRec {
  query_id: string;
  query: string;
  answerable: boolean;
}

function loadIndexFromDisk() {
  const manifest: Manifest = JSON.parse(readFileSync(join(INDEX_DIR, "manifest.json"), "utf8"));
  const buf: Record<string, ArrayBuffer> = {};
  for (const n of indexBlobNames(manifest)) {
    const b = readFileSync(join(INDEX_DIR, n));
    buf[n] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  }
  const passages: string[] = [];
  for (const sh of manifest.passageShards ?? ["passages.json"]) {
    for (const s of JSON.parse(readFileSync(join(INDEX_DIR, sh), "utf8")) as string[]) {
      passages.push(s);
    }
  }
  return { index: assembleIndex(manifest, buf, passages), manifest };
}

/** Drain the SSE stream `synthesizeStream` produces into one answer string. */
async function generateOnce(
  provider: NonNullable<ReturnType<typeof resolveProvider>>,
  query: string,
  sources: Array<{ title: string; text: string }>,
): Promise<{ text: string } | { insufficient: true } | { error: string }> {
  const stream = synthesizeStream(provider, query, sources);
  const reader = stream.getReader();
  const dec = new TextDecoder();
  let buffer = "";
  let full = "";
  let insufficient = false;
  let error: string | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += dec.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      try {
        const ev = JSON.parse(payload) as
          { t?: string; insufficient?: boolean; error?: string };
        if (ev.error) error = ev.error;
        else if (ev.insufficient) insufficient = true;
        else if (ev.t) full += ev.t;
      } catch { /* keep-alive or partial frame */ }
    }
  }
  if (error) return { error };
  if (insufficient || !full.trim()) return { insufficient: true };
  return { text: full.trim() };
}

interface Done {
  query_id: string;
  query: string;
  answer: string;
  passages: number[];
}

async function main(): Promise<void> {
  const provider = resolveProvider(process.env as Record<string, string | undefined>)
    ?? resolveProvider(dotenv("../web/.env.local") ?? dotenv(".env.local") ?? {});
  if (!provider) {
    console.error(
      "no generator configured. Put GROQ_API_KEY in web/.env.local, or set\n" +
      "ANTHROPIC_API_KEY, or LLM_BASE_URL + LLM_MODEL for anything else.");
    process.exit(1);
  }
  console.log(`generator: ${provider.label}`);

  const { index, manifest } = loadIndexFromDisk();
  const enc = new E5Encoder();
  await enc.init();

  const thresholds = existsSync("public/thresholds.json")
    ? JSON.parse(readFileSync("public/thresholds.json", "utf8"))
    : {};
  const engine = new RagEngine(index, enc, {
    ...DEFAULT_CONFIG,
    thresholds: {
      ...DEFAULT_CONFIG.thresholds,
      minTopScore: thresholds.minTopScore ?? DEFAULT_CONFIG.thresholds.minTopScore,
      minAgreement: thresholds.minAgreement ?? DEFAULT_CONFIG.thresholds.minAgreement,
      minLexicalOverlap: thresholds.minLexicalOverlap ?? 0,
    },
  });
  engine.setCorpusEnabled(true);
  await engine.warmup();

  const all: QRec[] = readFileSync(QUERIES, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  // Unanswerable queries are in the set on purpose — they are what gate 2 is
  // calibrated on — but there is by definition no answer to precompute.
  const wanted = all.filter((q) => q.answerable).slice(0, LIMIT);

  mkdirSync(OUT_DIR, { recursive: true });
  const done = new Map<string, Done>();
  if (existsSync(CHECKPOINT)) {
    let torn = 0;
    for (const line of readFileSync(CHECKPOINT, "utf8").trim().split("\n")) {
      if (!line) continue;
      try {
        const d = JSON.parse(line) as Done;
        done.set(d.query_id, d);
      } catch {
        // A run interrupted mid-append leaves a partial last line. Skipping it
        // costs one query on the next pass; refusing to parse the file would
        // cost every query already paid for.
        torn++;
      }
    }
    console.log(`resuming: ${done.size.toLocaleString()} already answered` +
                (torn ? ` (${torn} torn line${torn > 1 ? "s" : ""} skipped)` : ""));
  }
  const todo = wanted.filter((q) => !done.has(q.query_id));

  console.log(
    `${wanted.length.toLocaleString()} answerable queries, ` +
    `${todo.length.toLocaleString()} to do\n` +
    `workers ${WORKERS} · ${RPM} rpm · ${TPM.toLocaleString()} tpm\n`);
  if (todo.length && WORKERS > 1 && RPM <= 60) {
    console.log(
      `note: at ${RPM} rpm the rate limiter binds before latency does, so\n` +
      `      ${WORKERS} workers will not finish sooner than 1. Raise --rpm only\n` +
      `      if your key's tier actually allows it.\n`);
  }

  const reqBucket = new TokenBucket(RPM);
  const tokBucket = new TokenBucket(TPM);
  const checkpoint = openAppend(CHECKPOINT);

  let answered = 0, insufficient = 0, ungrounded = 0, failed = 0, refused = 0;
  let rateLimited = 0;
  const t0 = Date.now();
  let next = 0;

  const worker = async (): Promise<void> => {
    for (;;) {
      const i = next++;
      if (i >= todo.length) return;
      const q = todo[i];

      // The real pipeline, so the passages are exactly the ones the browser
      // would retrieve for this question.
      const r = await engine.ask(q.query, { skipCache: true });
      if (r.status !== "answered" || !r.citations.length) { refused++; continue; }
      // A user source cannot appear here — the corpus is the only thing
      // attached — but the id space is shared, so this is asserted rather than
      // assumed.
      if (r.citations.some((c) => c.passageId >= USER_BASE)) { refused++; continue; }

      const sources = r.citations.map((c) => ({ title: "MS MARCO-XI", text: c.text }));
      const cost = estimateTokens(sources.map((s) => s.text).join(" ") + q.query) + 600;
      await Promise.all([reqBucket.take(1), tokBucket.take(cost)]);

      let out: Awaited<ReturnType<typeof generateOnce>> | null = null;
      for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        try {
          out = await generateOnce(provider, q.query, sources);
        } catch (err) {
          failed++;
          console.error(`\n  ! ${q.query_id}: ${err instanceof Error ? err.message : String(err)}`);
          break;
        }
        if (!("error" in out) || out.error !== RATE_LIMITED) break;
        if (attempt === MAX_RETRIES) break;
        // Over budget: stall every worker, back off, and buy the tokens again.
        rateLimited++;
        tokBucket.penalise();
        await sleep(Math.min(60_000, 2_000 * 2 ** attempt));
        await tokBucket.take(cost);
        out = null;
      }
      if (!out) { failed++; continue; }

      if ("error" in out) { failed++; continue; }
      if ("insufficient" in out) { insufficient++; continue; }

      // Gate 3 at build time rather than at read time. A stored answer is a
      // fact about the corpus, so it is verified once here and never again.
      if (!gateGrounding(out.text, r.citations.map((c) => c.text)).pass) {
        ungrounded++;
        continue;
      }

      const rec: Done = {
        query_id: q.query_id,
        query: q.query,
        answer: out.text,
        passages: r.citations.map((c) => c.passageId),
      };
      done.set(q.query_id, rec);
      checkpoint(JSON.stringify(rec));
      answered++;

      const n = answered + insufficient + ungrounded + failed + refused;
      if (n % 25 === 0) {
        const rate = n / ((Date.now() - t0) / 60_000);
        const left = (todo.length - n) / Math.max(rate, 0.01);
        process.stdout.write(
          `\r  ${n}/${todo.length}  ${rate.toFixed(1)}/min  ` +
          `~${left < 60 ? `${left.toFixed(0)}m` : `${(left / 60).toFixed(1)}h`} left  ` +
          `(${answered} kept, ${insufficient} insufficient, ${ungrounded} ungrounded, ${failed} failed)   `);
      }
    }
  };

  await Promise.all(Array.from({ length: Math.max(1, WORKERS) }, worker));
  process.stdout.write("\n\n");

  console.log(`answered      ${answered.toLocaleString()}`);
  console.log(`insufficient  ${insufficient.toLocaleString()}   model said the passages don't answer it`);
  console.log(`ungrounded    ${ungrounded.toLocaleString()}   failed gate 3, discarded`);
  console.log(`refused       ${refused.toLocaleString()}   retrieval gate declined before generation`);
  console.log(`failed        ${failed.toLocaleString()}   provider error`);
  console.log(`rate limited  ${rateLimited.toLocaleString()}   429s absorbed and retried`);

  if (!done.size) {
    console.log("\nnothing to write.");
    return;
  }

  // ---- emit -----------------------------------------------------------------
  // Query vectors are stored in the index's own reduced space and quantised the
  // same way passages are, so the browser matches with `PassageRescorer` and no
  // second code path exists to drift.
  console.log(`\nembedding ${done.size.toLocaleString()} queries for lookup…`);
  const records = [...done.values()];
  const dim = manifest.dim;
  const int8 = new Int8Array(records.length * dim);
  const scales = new Float32Array(records.length);
  const raw = new Float32Array(dim);

  for (let i = 0; i < records.length; i++) {
    const vec = await enc.encodeQuery(records[i].query);
    index.projectQuery(vec, raw);
    let max = 0;
    for (let j = 0; j < dim; j++) { const a = Math.abs(raw[j]); if (a > max) max = a; }
    const s = max > 0 ? max / 127 : 1;
    for (let j = 0; j < dim; j++) {
      const v = Math.round(raw[j] / s);
      int8[i * dim + j] = v > 127 ? 127 : v < -128 ? -128 : v;
    }
    scales[i] = s * 127;
    if (i % 500 === 0) process.stdout.write(`\r  ${i}/${records.length}   `);
  }
  process.stdout.write(`\r  ${records.length}/${records.length}   \n`);

  writeFileSync(join(OUT_DIR, "answers.q8.bin"), Buffer.from(int8.buffer));
  writeFileSync(join(OUT_DIR, "answers.scale.f32.bin"), Buffer.from(scales.buffer));
  const entries = records.map((r) => ({ q: r.query, a: r.answer, p: r.passages }));
  writeFileSync(join(OUT_DIR, "answers.json"), JSON.stringify(entries));
  writeFileSync(join(OUT_DIR, "manifest.json"), JSON.stringify({
    indexBuiltAt: manifest.builtAt,
    dim,
    count: records.length,
    model: provider.label,
    builtAt: new Date().toISOString(),
  }, null, 2));

  const bytes = int8.byteLength + scales.byteLength + JSON.stringify(entries).length;
  console.log(`\nwrote ${OUT_DIR}/ — ${records.length.toLocaleString()} answers, ` +
              `${(bytes / 1e6).toFixed(1)} MB`);
}

/** Minimal .env reader — this runs in Node, so the key never leaves the machine. */
function dotenv(path: string): Record<string, string> | null {
  if (!existsSync(path)) return null;
  const out: Record<string, string> = {};
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const m = /^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/.exec(line);
    if (m) out[m[1]] = m[2].trim().replace(/^["']|["']$/g, "");
  }
  return out;
}

function openAppend(path: string): (line: string) => void {
  return (line: string) => appendFileSync(path, line + "\n");
}

await main();
