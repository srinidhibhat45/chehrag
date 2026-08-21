/**
 * End-to-end voice latency: spoken question in, grounded answer out.
 *
 * Everywhere else in this project the 200 ms figure covers retrieval, and the
 * README argues at length why speech-to-text cannot be inside it - the fastest
 * streaming STT on the market spends 150 ms of a 200 ms budget on its own. That
 * argument is sound and it is still an argument. This is the measurement.
 *
 * Method, and its one real weakness stated first:
 *
 *   The audio is synthesised. Real questions are spoken by people in rooms with
 *   traffic outside, and TTS audio is cleaner than that. So the accuracy
 *   numbers below are optimistic and are reported as a sanity check rather than
 *   as a claim about transcription quality in the wild. The latency numbers
 *   are not flattered in the same way: Sarvam is transcribing a real audio file
 *   of a real length over a real network, and that is what the clock measures.
 *
 *   1. Sarvam `bulbul:v2` speaks each Hindi query          (setup, not measured)
 *   2. that audio goes to Sarvam `saaras:v3`               (measured - network)
 *   3. the transcript runs through the shipped pipeline    (measured - local)
 *
 * Reported as three distributions rather than one, because they are three
 * different kinds of cost and only one of them is ours to control. Adding them
 * into a single number would hide exactly the thing this measurement exists to
 * expose.
 *
 * Batch STT is the honest choice for a measurement, and it is also the slower
 * one. In the app the socket is already open and partial transcripts arrive
 * during speech, so the wait a user experiences after they stop talking is
 * shorter than this. A number measured on the streaming path would depend on
 * how long the speaker talked, which is not a property of this system.
 *
 *   npx tsx bench/voice.ts               30 queries
 *   npx tsx bench/voice.ts --limit 60
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG } from "../src/harness/rag";
import { percentiles } from "../src/harness/pipeline";
import { contentTokens } from "../src/retrieval/tokens";

const SARVAM_TTS = "https://api.sarvam.ai/text-to-speech";
const SARVAM_STT = "https://api.sarvam.ai/speech-to-text";
const INDEX_DIR = process.env.INDEX_DIR ?? "public/index";
const QUERIES = process.env.QUERIES ?? "../pipeline/data/subset/queries.hin.jsonl";

const argv = process.argv.slice(2);
const argOf = (n: string, d: number) => {
  const i = argv.indexOf(n);
  return i >= 0 && argv[i + 1] ? Number(argv[i + 1]) : d;
};
const LIMIT = argOf("--limit", 30);
const GAP_MS = argOf("--gap", 250);

function dotenv(path: string): Record<string, string> | null {
  if (!existsSync(path)) return null;
  const out: Record<string, string> = {};
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const m = /^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/.exec(line);
    if (m) out[m[1]] = m[2].trim().replace(/^["']|["']$/g, "");
  }
  return out;
}

function loadFromDisk() {
  const manifest: Manifest = JSON.parse(readFileSync(join(INDEX_DIR, "manifest.json"), "utf8"));
  const buf: Record<string, ArrayBuffer> = {};
  for (const n of indexBlobNames(manifest)) {
    const b = readFileSync(join(INDEX_DIR, n));
    buf[n] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  }
  const passages: string[] = [];
  for (const sh of manifest.passageShards ?? ["passages.json"]) {
    for (const s of JSON.parse(readFileSync(join(INDEX_DIR, sh), "utf8")) as string[]) passages.push(s);
  }
  return assembleIndex(manifest, buf, passages);
}

/** Word-level agreement between what was asked and what was heard. */
function transcriptOverlap(said: string, heard: string): number {
  const a = contentTokens(said);
  const b = contentTokens(heard);
  if (!a.size) return 0;
  let hit = 0;
  for (const w of a) if (b.has(w)) hit++;
  return hit / a.size;
}

async function speak(key: string, text: string): Promise<Uint8Array | null> {
  const res = await fetch(SARVAM_TTS, {
    method: "POST",
    headers: { "content-type": "application/json", "api-subscription-key": key },
    body: JSON.stringify({
      text: text.slice(0, 480),
      target_language_code: "hi-IN",
      model: "bulbul:v2",
      speaker: "anushka",
      enable_preprocessing: true,
    }),
  });
  if (!res.ok) return null;
  const data = await res.json() as { audios?: string[] };
  const b64 = data.audios?.[0];
  return b64 ? Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)) : null;
}

async function transcribe(key: string, wav: Uint8Array): Promise<{ text: string; ms: number } | null> {
  const form = new FormData();
  form.append("file", new Blob([wav as unknown as BlobPart], { type: "audio/wav" }), "q.wav");
  form.append("model", "saaras:v3");
  form.append("language_code", "hi-IN");
  const t0 = performance.now();
  const res = await fetch(SARVAM_STT, {
    method: "POST",
    headers: { "api-subscription-key": key },
    body: form,
  });
  const ms = performance.now() - t0;
  if (!res.ok) return null;
  const data = await res.json() as { transcript?: string };
  return { text: (data.transcript ?? "").trim(), ms };
}

async function main() {
  const env = { ...process.env, ...(dotenv("../worker/.dev.vars") ?? {}), ...(dotenv(".env.local") ?? {}) };
  const key = (env.SARVAM_API_KEY ?? "").trim();
  if (!key) {
    console.error("no SARVAM_API_KEY - put one in worker/.dev.vars");
    process.exit(1);
  }

  const index = loadFromDisk();
  const enc = new E5Encoder();
  await enc.init();
  let cfg = { ...DEFAULT_CONFIG };
  try {
    const th = JSON.parse(readFileSync("public/thresholds.json", "utf8"));
    cfg = { ...cfg, thresholds: { ...cfg.thresholds,
      minTopScore: th.minTopScore, minAgreement: th.minAgreement,
      minLexicalOverlap: th.minLexicalOverlap ?? 0 } };
  } catch { /* placeholders */ }
  const engine = new RagEngine(index, enc, cfg);
  engine.setCorpusEnabled(true);
  await engine.warmup();

  const all = readFileSync(QUERIES, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const sample = all.filter((q: { answerable: boolean }) => q.answerable).slice(50, 50 + LIMIT);

  console.log("=".repeat(76));
  console.log("END-TO-END VOICE - spoken question to grounded answer");
  console.log("=".repeat(76));
  console.log(`stt        : Sarvam saaras:v3 (batch), over the network`);
  console.log(`audio      : synthesised by Sarvam bulbul:v2 - see the header of this file`);
  console.log(`queries    : ${sample.length}\n`);

  const sttMs: number[] = [];
  const ragMs: number[] = [];
  const totalMs: number[] = [];
  const overlaps: number[] = [];
  let ttsFailed = 0, sttFailed = 0, answered = 0, audioBytes = 0;

  for (const q of sample) {
    const wav = await speak(key, q.query);
    if (!wav) { ttsFailed++; process.stdout.write("t"); continue; }
    audioBytes += wav.byteLength;

    const heard = await transcribe(key, wav);
    if (!heard || !heard.text) { sttFailed++; process.stdout.write("s"); continue; }

    const r = await engine.ask(heard.text, { skipCache: true });

    sttMs.push(heard.ms);
    ragMs.push(r.totalMs);
    totalMs.push(heard.ms + r.totalMs);
    overlaps.push(transcriptOverlap(q.query, heard.text));
    if (r.status === "answered") answered++;
    process.stdout.write(r.status === "answered" ? "." : "·");
    await new Promise((res) => setTimeout(res, GAP_MS));
  }

  if (!totalMs.length) {
    console.log("\n\nno successful runs - check the Sarvam key and quota");
    process.exit(1);
  }

  const row = (label: string, v: number[]) => {
    const p = percentiles(v);
    console.log(`  ${label.padEnd(22)} ${p.p50.toFixed(1).padStart(8)} ${p.p70.toFixed(1).padStart(8)} ` +
                `${p.p100.toFixed(1).padStart(8)} ${p.mean.toFixed(1).padStart(8)}`);
  };

  console.log("\n\n" + "-".repeat(76));
  console.log(`LATENCY  (n=${totalMs.length}, ms)`);
  console.log("-".repeat(76));
  console.log(`  ${"".padEnd(22)} ${"P50".padStart(8)} ${"P70".padStart(8)} ${"P100".padStart(8)} ${"mean".padStart(8)}`);
  row("speech-to-text", sttMs);
  row("retrieval pipeline", ragMs);
  row("TOTAL voice -> answer", totalMs);

  const rp = percentiles(ragMs);
  const tp = percentiles(totalMs);
  const sp = percentiles(sttMs);
  console.log("-".repeat(76));
  console.log(`  retrieval under 200 ms : ${(100 * ragMs.filter((t) => t < 200).length / ragMs.length).toFixed(1)}%` +
              `   (P100 ${rp.p100.toFixed(1)} ms - the figure the app reports)`);
  console.log(`  STT share of the total : ${(100 * sp.p50 / tp.p50).toFixed(1)}% at P50`);
  console.log(`  audio                  : ${(audioBytes / Math.max(1, totalMs.length) / 1024).toFixed(0)} KB per question`);

  console.log("\n" + "-".repeat(76));
  console.log("TRANSCRIPTION  (sanity check - synthesised audio, see file header)");
  console.log("-".repeat(76));
  const op = percentiles(overlaps.map((o) => o * 100));
  console.log(`  content-word overlap   : median ${op.p50.toFixed(0)}%  mean ${op.mean.toFixed(0)}%`);
  console.log(`  answered after STT     : ${answered}/${totalMs.length} ` +
              `(${(100 * answered / totalMs.length).toFixed(0)}%)`);
  console.log(`  tts failed ${ttsFailed}  ·  stt failed ${sttFailed}`);
  console.log("=".repeat(76));
  console.log("The retrieval column is the one under a hard cap. The STT column is a");
  console.log("network round trip to a vendor and is reported, not claimed.");
  console.log("=".repeat(76));
}

main().catch((e) => { console.error(e); process.exit(1); });
