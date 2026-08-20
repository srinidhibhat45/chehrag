/**
 * Benchmark harness — requirement 4 (P50 / P70 / P100) plus quality and
 * guardrail measurement.
 *
 * Runs the SAME modules the browser runs (`assembleIndex` -> `RagEngine`), just
 * with blobs read from disk instead of fetched. Nothing here re-implements the
 * pipeline, so the numbers describe the shipped code.
 *
 * Methodology, stated plainly because latency numbers are easy to flatter:
 *   - warmup queries run first and are DISCARDED. Un-warmed JS runs interpreted
 *     before the JIT promotes it; including that would measure the JIT, not us.
 *   - the answer cache is cleared before the measured run, and every measured
 *     query is distinct. A cached hit is ~0ms and would be dishonest.
 *   - index load is excluded and reported separately. It happens once per
 *     session; the requirement is about per-query cost.
 *   - P100 is the max over the whole sample. No trimming, no outlier removal.
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG, type RagConfig } from "../src/harness/rag";
import { percentiles } from "../src/harness/pipeline";

const INDEX_DIR = process.env.INDEX_DIR ?? "public/index";
const QUERIES = process.env.QUERIES ?? "../pipeline/data/subset/queries.hin.jsonl";
const N = Number(process.env.N ?? 500);
const WARMUP = Number(process.env.WARMUP ?? 30);

interface QRec {
  query_id: number; query: string; answer: string; query_type: string;
  gold_passage_ids: string[]; answerable: boolean;
}

function loadFromDisk() {
  const manifest: Manifest = JSON.parse(readFileSync(join(INDEX_DIR, "manifest.json"), "utf8"));
  const buf: Record<string, ArrayBuffer> = {};
  let bytes = 0;
  for (const n of indexBlobNames(manifest)) {
    const b = readFileSync(join(INDEX_DIR, n));
    bytes += b.byteLength;
    buf[n] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  }
  const shards: string[] = manifest.passageShards ?? ["passages.json"];
  const passages: string[] = [];
  for (const sh of shards) {
    for (const s of JSON.parse(readFileSync(join(INDEX_DIR, sh), "utf8")) as string[]) passages.push(s);
  }
  return { index: assembleIndex(manifest, buf, passages), bytes, manifest };
}

function fmt(p: Record<string, number>): string {
  return `p50 ${p.p50.toFixed(1)}  p70 ${p.p70.toFixed(1)}  p90 ${p.p90.toFixed(1)}  ` +
         `p95 ${p.p95.toFixed(1)}  p99 ${p.p99.toFixed(1)}  P100 ${p.p100.toFixed(1)}  ` +
         `mean ${p.mean.toFixed(1)}`;
}

async function main() {
  if (!existsSync(join(INDEX_DIR, "manifest.json"))) {
    console.error(`no index at ${INDEX_DIR} — run pipeline/src/build_index.py first`);
    process.exit(1);
  }

  const t0 = performance.now();
  const { index, bytes, manifest } = loadFromDisk();
  const loadMs = performance.now() - t0;

  console.log("=".repeat(74));
  console.log("VOICE RAG — BENCHMARK");
  console.log("=".repeat(74));
  console.log(`index      : ${manifest.numPassages.toLocaleString()} passages, ` +
              `${Object.keys(manifest.strategies).length} strategies, dim ${manifest.dim}`);
  for (const [k, v] of Object.entries(manifest.strategies)) {
    console.log(`             ${k.padEnd(12)} ${v.count.toLocaleString().padStart(9)} chunks  ` +
                `nlist ${String(v.nlist).padStart(5)}  avg list ${v.avgListSize.toFixed(1)}`);
  }
  console.log(`blobs      : ${(bytes / 1e6).toFixed(1)} MB, assembled in ${loadMs.toFixed(0)} ms ` +
              `(one-time, EXCLUDED from per-query numbers)`);

  const enc = new E5Encoder();
  const encT = performance.now();
  await enc.init();
  console.log(`encoder    : ready in ${(performance.now() - encT).toFixed(0)} ms`);

  const all: QRec[] = readFileSync(QUERIES, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const pidOrd = new Map<string, number>();
  // passage ordinals come from corpus order; rebuild the id->ordinal map
  const corpusPath = "../pipeline/data/subset/corpus.hin.jsonl";
  readFileSync(corpusPath, "utf8").trim().split("\n").forEach((l, i) => {
    pidOrd.set(JSON.parse(l).id, i);
  });

  // Use the fitted thresholds if calibration has been run (bench/calibrate.ts).
  let cfg: RagConfig = { ...DEFAULT_CONFIG };
  try {
    const th = JSON.parse(readFileSync("public/thresholds.json", "utf8"));
    cfg = { ...cfg, thresholds: { ...cfg.thresholds,
                                  minTopScore: th.minTopScore, minAgreement: th.minAgreement,
                                  minLexicalOverlap: th.minLexicalOverlap ?? 0 } };
    console.log(`thresholds : fitted — minTopScore ${th.minTopScore} minAgreement ${th.minAgreement}`);
  } catch { console.log("thresholds : defaults (calibration not run)"); }
  const engine = new RagEngine(index, enc, cfg);
  // The engine ships with the corpus off — the app starts empty so that an
  // answer can only have come from what the user added. Every benchmark here
  // exists to measure that corpus, so it opts in explicitly.
  engine.setCorpusEnabled(true);

  // ---- warmup (discarded) -------------------------------------------------
  const warm = all.slice(0, WARMUP);
  for (const q of warm) await engine.ask(q.query, { skipCache: true });
  console.log(`warmup     : ${WARMUP} queries, discarded\n`);

  // ---- measured run -------------------------------------------------------
  const sample = all.slice(WARMUP, WARMUP + N);
  const totals: number[] = [];
  const stageMs: Record<string, number[]> = {};
  let answered = 0, refused = 0;
  let hit1 = 0, hit5 = 0, hitAny = 0, gradedAnswerable = 0;
  const refusalByType: Record<string, number> = {};
  // abstention confusion matrix
  // abstention is the positive class; names spelled out to prevent the mix-up
  let correctAbstain = 0, overRefusal = 0, missedAbstain = 0, correctAnswer = 0;

  for (const q of sample) {
    const r = await engine.ask(q.query, { skipCache: true });
    totals.push(r.totalMs);
    for (const s of r.telemetry) {
      (stageMs[s.name] ??= []).push(s.ms);
    }
    if (r.status === "answered") {
      answered++;
      if (q.answerable) {
        gradedAnswerable++;
        const gold = new Set(q.gold_passage_ids.map((g) => pidOrd.get(g)).filter((x) => x != null));
        const ids = r.citations.map((c) => c.passageId);
        if (gold.has(ids[0])) hit1++;
        if (ids.some((i) => gold.has(i))) hit5++;
        if (ids.some((i) => gold.has(i))) hitAny++;
        correctAnswer++;
      } else {
        missedAbstain++;    // unanswerable but answered -> should have abstained
      }
    } else {
      refused++;
      refusalByType[r.refusal ?? "?"] = (refusalByType[r.refusal ?? "?"] ?? 0) + 1;
      if (q.answerable) overRefusal++;   // refused a real answer
      else correctAbstain++;             // correct abstention
    }
  }

  const p = percentiles(totals);
  console.log("-".repeat(74));
  console.log(`LATENCY  (n=${totals.length}, end-to-end pipeline, warm index, no cache)`);
  console.log("-".repeat(74));
  console.log("  " + fmt(p));
  console.log(`  under 200ms: ${(100 * totals.filter((t) => t < 200).length / totals.length).toFixed(2)}%`);
  console.log(`  VERDICT    : P100 = ${p.p100.toFixed(1)} ms ` +
              `${p.p100 < 200 ? "PASS (<200ms)" : "FAIL (>=200ms)"}`);

  console.log("\n  per-stage:");
  for (const [name, arr] of Object.entries(stageMs)) {
    const sp = percentiles(arr);
    console.log(`    ${name.padEnd(20)} p50 ${sp.p50.toFixed(2).padStart(7)}  ` +
                `p100 ${sp.p100.toFixed(2).padStart(7)}  n=${arr.length}`);
  }

  console.log("\n" + "-".repeat(74));
  console.log("RETRIEVAL QUALITY  (answerable queries that were answered)");
  console.log("-".repeat(74));
  if (gradedAnswerable) {
    console.log(`  graded     : ${gradedAnswerable}`);
    console.log(`  hit@1      : ${(hit1 / gradedAnswerable).toFixed(4)}`);
    console.log(`  hit@3      : ${(hit5 / gradedAnswerable).toFixed(4)}   (3 citations returned)`);
  } else {
    console.log("  no answerable queries were answered — thresholds may be too strict");
  }

  console.log("\n" + "-".repeat(74));
  console.log("GUARDRAILS  (abstention treated as the positive class)");
  console.log("-".repeat(74));
  const prec = correctAbstain + overRefusal ? correctAbstain / (correctAbstain + overRefusal) : 0;
  const rec = correctAbstain + missedAbstain ? correctAbstain / (correctAbstain + missedAbstain) : 0;
  const cov = correctAnswer + overRefusal ? correctAnswer / (correctAnswer + overRefusal) : 0;
  console.log(`  answered ${answered}  refused ${refused}`);
  console.log(`  correct abstentions   (unanswerable, refused) : ${correctAbstain}`);
  console.log(`  missed abstentions    (unanswerable, answered): ${missedAbstain}`);
  console.log(`  over-refusals         (answerable,   refused) : ${overRefusal}`);
  console.log(`  correct answers       (answerable,   answered): ${correctAnswer}`);
  console.log(`  coverage of answerable: ${(cov * 100).toFixed(1)}%`);
  console.log(`  abstention precision  : ${prec.toFixed(4)}`);
  console.log(`  abstention recall     : ${rec.toFixed(4)}`);
  console.log(`  F1                    : ${(prec + rec ? 2 * prec * rec / (prec + rec) : 0).toFixed(4)}`);
  console.log("  refusal reasons:", JSON.stringify(refusalByType));
  console.log("=".repeat(74));
}

main().catch((e) => { console.error(e); process.exit(1); });
