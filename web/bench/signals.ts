/**
 * How separable are answerable and unanswerable queries, really?
 *
 * Collects every confidence signal per query and reports AUC for each, plus a
 * simple combination. AUC is threshold-free, so it answers "is the signal there
 * at all?" before we argue about where to cut.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG } from "../src/harness/rag";
import { lexicalOverlap } from "../src/retrieval/fusion";

const D = "public/index";
const N = Number(process.env.N ?? 2000);

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
const index = assembleIndex(manifest, buf, passages);
const enc = new E5Encoder();
await enc.init();
const engine = new RagEngine(index, enc, {
  ...DEFAULT_CONFIG,
  thresholds: { minTopScore: -1, minAgreement: 0, minLexicalOverlap: 0,
                rescueMinScore: Infinity, rescueMinOverlap: 1 },
});
// The engine ships with the corpus off - the app starts empty so that an
// answer can only have come from what the user added. Every benchmark here
// exists to measure that corpus, so it opts in explicitly.
engine.setCorpusEnabled(true);
await engine.warmup();

const qs = readFileSync("../pipeline/data/subset/queries.hin.jsonl", "utf8")
  .trim().split("\n").map((l) => JSON.parse(l)).slice(0, N);

interface Row { top: number; margin: number; agree: number; lex: number; ans: boolean; }
const rows: Row[] = [];
for (const q of qs) {
  const r = await engine.ask(q.query, { skipCache: true });
  const c = r.citations;
  rows.push({
    top: c[0]?.score ?? 0,
    margin: c.length > 1 ? (c[0].score - c[1].score) / Math.max(1e-6, c[0].score) : 1,
    agree: c[0]?.strategies.length ?? 0,
    lex: lexicalOverlap(q.query, c[0]?.text ?? ""),
    ans: q.answerable,
  });
}

/** AUC via rank-sum. 0.5 = no signal, 1.0 = perfect. */
function auc(vals: number[], pos: boolean[]): number {
  const idx = vals.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const rank = new Array(vals.length).fill(0);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) rank[idx[k][1]] = avg;
    i = j + 1;
  }
  const nPos = pos.filter(Boolean).length, nNeg = pos.length - nPos;
  if (!nPos || !nNeg) return 0.5;
  let sum = 0;
  for (let k = 0; k < vals.length; k++) if (pos[k]) sum += rank[k];
  return (sum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

const isAns = rows.map((r) => r.ans);
console.log("=".repeat(70));
console.log(`SIGNAL SEPARABILITY  (n=${rows.length}, ` +
            `${isAns.filter(Boolean).length} answerable / ${isAns.filter((x) => !x).length} unanswerable)`);
console.log("=".repeat(70));
console.log("AUC - probability a random answerable query scores above a random unanswerable one");
console.log("  1.00 = perfect separation, 0.50 = no signal at all\n");
for (const [name, get] of [
  ["topScore", (r: Row) => r.top],
  ["margin", (r: Row) => r.margin],
  ["agreement", (r: Row) => r.agree],
  ["lexicalOverlap", (r: Row) => r.lex],
  ["top + 0.3*lex", (r: Row) => r.top + 0.3 * r.lex],
  ["top + 0.3*lex + 0.02*agree", (r: Row) => r.top + 0.3 * r.lex + 0.02 * r.agree],
] as const) {
  const a = auc(rows.map(get as (r: Row) => number), isAns);
  const bar = "█".repeat(Math.round((a - 0.5) * 80));
  console.log(`  ${name.padEnd(28)} ${a.toFixed(4)}  ${bar}`);
}

const ansScores = rows.filter((r) => r.ans).map((r) => r.top).sort((a, b) => a - b);
const unaScores = rows.filter((r) => !r.ans).map((r) => r.top).sort((a, b) => a - b);
const q = (a: number[], p: number) => a[Math.floor(p * (a.length - 1))];
console.log("\ntopScore distribution:");
console.log(`  answerable    p10 ${q(ansScores,.1).toFixed(3)}  p50 ${q(ansScores,.5).toFixed(3)}  p90 ${q(ansScores,.9).toFixed(3)}`);
console.log(`  unanswerable  p10 ${q(unaScores,.1).toFixed(3)}  p50 ${q(unaScores,.5).toFixed(3)}  p90 ${q(unaScores,.9).toFixed(3)}`);

writeFileSync("/tmp/signals.json", JSON.stringify(rows));
console.log("\nwrote /tmp/signals.json");
