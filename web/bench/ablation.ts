/**
 * Leave-one-out ablation over the indices.
 *
 * `pipeline/src/chunking/build.py` claims that keeping the strategies in
 * separate indices "lets us measure each strategy's contribution in isolation".
 * This is that measurement. Without it, seven indices is a story rather than a
 * result, and the honest answer to "is the sliding window worth 250k vectors"
 * is unknown.
 *
 * Measured on retrieval, not on answers. `RagAnswer.retrieved` is populated
 * whether or not the guardrails let the answer through, so removing an index
 * cannot flatter itself by shifting queries across the gate-2 threshold — which
 * is exactly what would happen if this graded citations. Answer rate is
 * reported alongside, as a separate column, for the same reason.
 *
 * Reading it: a strategy earns its place if removing it *costs* recall. A row
 * whose delta is zero or positive is a strategy paying for itself in download
 * size and query time and returning nothing.
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG, type RagConfig } from "../src/harness/rag";
import { percentiles } from "../src/harness/pipeline";

const INDEX_DIR = process.env.INDEX_DIR ?? "public/index";
const QUERIES = process.env.QUERIES ?? "../pipeline/data/subset/queries.hin.jsonl";
const N = Number(process.env.N ?? 300);
const WARMUP = Number(process.env.WARMUP ?? 20);

interface QRec {
  query_id: number; query: string; answerable: boolean; gold_passage_ids: string[];
}

function loadFromDisk() {
  const manifest: Manifest = JSON.parse(readFileSync(join(INDEX_DIR, "manifest.json"), "utf8"));
  const buf: Record<string, ArrayBuffer> = {};
  for (const n of indexBlobNames(manifest)) {
    const b = readFileSync(join(INDEX_DIR, n));
    buf[n] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  }
  const shards: string[] = manifest.passageShards ?? ["passages.json"];
  const passages: string[] = [];
  for (const sh of shards) {
    for (const s of JSON.parse(readFileSync(join(INDEX_DIR, sh), "utf8")) as string[]) passages.push(s);
  }
  return { index: assembleIndex(manifest, buf, passages), manifest };
}

interface Score {
  hit1: number; hit3: number; hit5: number; hit10: number; mrr: number;
  graded: number; answered: number; total: number; p50: number; p100: number;
}

async function evaluate(
  engine: RagEngine, sample: QRec[], gold: Map<number, Set<number>>,
): Promise<Score> {
  let hit1 = 0, hit3 = 0, hit5 = 0, hit10 = 0, mrr = 0, graded = 0, answered = 0;
  const times: number[] = [];
  for (const q of sample) {
    const r = await engine.ask(q.query, { skipCache: true });
    times.push(r.totalMs);
    if (r.status === "answered") answered++;
    const g = gold.get(q.query_id);
    if (!g || !g.size) continue;
    graded++;
    const ids = r.retrieved ?? [];
    let rank = 0;
    for (let i = 0; i < ids.length; i++) {
      if (g.has(ids[i])) { rank = i + 1; break; }
    }
    if (rank) {
      mrr += 1 / rank;
      if (rank <= 1) hit1++;
      if (rank <= 3) hit3++;
      if (rank <= 5) hit5++;
      if (rank <= 10) hit10++;
    }
  }
  const p = percentiles(times);
  return { hit1, hit3, hit5, hit10, mrr, graded, answered,
           total: sample.length, p50: p.p50, p100: p.p100 };
}

async function main() {
  if (!existsSync(join(INDEX_DIR, "manifest.json"))) {
    console.error(`no index at ${INDEX_DIR}`);
    process.exit(1);
  }
  const { index, manifest } = loadFromDisk();
  const enc = new E5Encoder();
  await enc.init();

  const strategies = [...Object.keys(manifest.strategies)];
  if (index.lexical) strategies.push("lexical");

  const all: QRec[] = readFileSync(QUERIES, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const pidOrd = new Map<string, number>();
  readFileSync("../pipeline/data/subset/corpus.hin.jsonl", "utf8")
    .trim().split("\n").forEach((l, i) => { pidOrd.set(JSON.parse(l).id, i); });

  const sample = all.slice(WARMUP, WARMUP + N).filter((q) => q.answerable);
  const gold = new Map<number, Set<number>>();
  for (const q of sample) {
    gold.set(q.query_id, new Set(
      q.gold_passage_ids.map((g) => pidOrd.get(g)).filter((x): x is number => x != null)));
  }

  let base: RagConfig = { ...DEFAULT_CONFIG };
  try {
    const th = JSON.parse(readFileSync("public/thresholds.json", "utf8"));
    base = { ...base, thresholds: { ...base.thresholds,
      minTopScore: th.minTopScore, minAgreement: th.minAgreement,
      minLexicalOverlap: th.minLexicalOverlap ?? 0 } };
  } catch { /* defaults */ }

  console.log("=".repeat(78));
  console.log("INDEX ABLATION — leave one out");
  console.log("=".repeat(78));
  console.log(`indices    : ${strategies.join(", ")}`);
  console.log(`queries    : ${sample.length} answerable, graded on retrieved rank (pre-guardrail)\n`);

  const runs: Array<{ label: string; on: string[] | null }> = [
    { label: "ALL", on: null },
    ...strategies.map((s) => ({
      label: `  – ${s}`, on: strategies.filter((x) => x !== s),
    })),
    { label: "dense only", on: strategies.filter((s) => s !== "lexical") },
    ...(index.lexical ? [{ label: "lexical only", on: ["lexical"] }] : []),
  ];

  const results: Array<{ label: string; s: Score }> = [];
  for (const run of runs) {
    const engine = new RagEngine(index, enc, { ...base, enabledStrategies: run.on });
    engine.setCorpusEnabled(true);
    // Per config, because the JIT specialises on the code path each one takes.
    for (const q of all.slice(0, WARMUP)) await engine.ask(q.query, { skipCache: true });
    results.push({ label: run.label, s: await evaluate(engine, sample, gold) });
    process.stdout.write(".");
  }
  process.stdout.write("\n\n");

  const ref = results[0].s;
  const pct = (n: number, d: number) => (d ? (100 * n) / d : 0);
  const col = (v: string, w: number) => v.padStart(w);
  console.log("config".padEnd(16) + ["hit@1", "hit@3", "hit@5", "hit@10", "MRR@10", "Δhit@5", "p50 ms"]
    .map((h) => col(h, 9)).join(""));
  console.log("-".repeat(78));
  for (const { label, s } of results) {
    const d = pct(s.hit5, s.graded) - pct(ref.hit5, ref.graded);
    console.log(label.padEnd(16) +
      col(pct(s.hit1, s.graded).toFixed(1) + "%", 9) +
      col(pct(s.hit3, s.graded).toFixed(1) + "%", 9) +
      col(pct(s.hit5, s.graded).toFixed(1) + "%", 9) +
      col(pct(s.hit10, s.graded).toFixed(1) + "%", 9) +
      col((s.mrr / Math.max(1, s.graded)).toFixed(4), 9) +
      col((d >= 0 ? "+" : "") + d.toFixed(1), 9) +
      col(s.p50.toFixed(2), 9));
  }
  console.log("-".repeat(78));
  console.log("Δhit@5 is against ALL. A NEGATIVE delta on a '– x' row means removing x");
  console.log("costs recall, i.e. x is earning its place.");
  console.log("=".repeat(78));
}

main().catch((e) => { console.error(e); process.exit(1); });
