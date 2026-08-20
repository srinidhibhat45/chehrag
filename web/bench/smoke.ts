import { readFileSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG } from "../src/harness/rag";

const D = "public/index";
const manifest: Manifest = JSON.parse(readFileSync(join(D, "manifest.json"), "utf8"));
const buf: Record<string, ArrayBuffer> = {};
for (const n of indexBlobNames(manifest)) {
  const b = readFileSync(join(D, n));
  buf[n] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
}
const passages: string[] = [];
for (const sh of manifest.passageShards) {
  for (const s of JSON.parse(readFileSync(join(D, sh), "utf8")) as string[]) passages.push(s);
}
console.log(`index: ${manifest.numPassages} passages, dim ${manifest.dim}, ${passages.length} texts loaded`);

const index = assembleIndex(manifest, buf, passages);
const enc = new E5Encoder();
await enc.init();
// thresholds off, so we can see raw behaviour before calibration
const engine = new RagEngine(index, enc, {
  ...DEFAULT_CONFIG,
  thresholds: { minTopScore: -1, minAgreement: 0, minLexicalOverlap: 0,
                rescueMinScore: Infinity, rescueMinOverlap: 1 },
});
// The engine ships with the corpus off — the app starts empty so that an
// answer can only have come from what the user added. Every benchmark here
// exists to measure that corpus, so it opts in explicitly.
engine.setCorpusEnabled(true);
await engine.warmup();

const qs = readFileSync("../pipeline/data/subset/queries.hin.jsonl", "utf8")
  .trim().split("\n").slice(0, 6).map((l) => JSON.parse(l));

for (const q of qs) {
  const r = await engine.ask(q.query, { skipCache: true });
  const gold = new Set(q.gold_passage_ids);
  console.log("\n" + "─".repeat(70));
  console.log(`Q  ${q.query}`);
  console.log(`   type=${q.query_type} answerable=${q.answerable} ${r.totalMs.toFixed(1)}ms conf=${r.confidence.toFixed(4)}`);
  console.log(`A  ${r.answer.slice(0, 150)}`);
  console.log(`   strategies: ${r.citations[0]?.strategies.join(",") ?? "-"}`);
  console.log(`   gold=${gold.size} | reference: ${String(q.answer).slice(0, 90)}`);
}
