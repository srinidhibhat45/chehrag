/**
 * Does the precomputed store answer the right question, and only the right one?
 *
 * Two properties, and the second is the one that matters. Returning a stored
 * answer for a question it was not written for is worse than any latency it
 * saves, so the decoys below are questions that retrieve *well* from this corpus
 * and still must not match — a near neighbour is the failure mode, not a random
 * string.
 *
 * Run after `bench/precompute.ts`:
 *   npx tsx bench/precomputed.ts
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG } from "../src/harness/rag";
import { PrecomputedAnswers, type PrecomputedManifest } from "../src/answer/precomputed";

const D = "public/index";
const A = "public/answers";
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
const th = JSON.parse(readFileSync("public/thresholds.json", "utf8"));
const engine = new RagEngine(index, enc, {
  ...DEFAULT_CONFIG,
  thresholds: { ...DEFAULT_CONFIG.thresholds, minTopScore: th.minTopScore,
                minAgreement: th.minAgreement, minLexicalOverlap: 0 },
});
engine.setCorpusEnabled(true);
await engine.warmup();

const am: PrecomputedManifest = JSON.parse(readFileSync(join(A, "manifest.json"), "utf8"));
const rd = (n: string) => {
  const b = readFileSync(join(A, n));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};
const store = new PrecomputedAnswers(
  am,
  new Int8Array(rd("answers.q8.bin")),
  new Float32Array(rd("answers.scale.f32.bin")),
  JSON.parse(readFileSync(join(A, "answers.json"), "utf8")),
);
console.log(`store: ${store.size} answers, ${store.model}`);
console.log(`index parity: ${am.indexBuiltAt === manifest.builtAt ? "ok" : "MISMATCH"}\n`);

const done = store.pairs.map((p) => ({ query: p.q, answer: p.a }));

let hit = 0, miss = 0;
console.log("─".repeat(74));
console.log("EXACT — every precomputed question must find its own answer");
console.log("─".repeat(74));
for (const d of done.slice(0, 8)) {
  const r = await engine.ask(d.query, { skipCache: true });
  const top = r.citations[0]?.passageId;
  const got = top !== undefined ? store.lookup(engine.lastQueryVector, top) : null;
  const ok = got?.answer === d.answer;
  ok ? hit++ : miss++;
  console.log(`${ok ? "  ok  " : "  MISS"} sim=${got ? got.similarity.toFixed(4) : "  --  "}  ${d.query.slice(0, 46)}`);
}

console.log("\n" + "─".repeat(74));
console.log("WRONG QUESTION — must NOT return a stored answer");
console.log("─".repeat(74));
const DECOYS = [
  "what is the capital of Peru",
  "how do I make sourdough bread",
  "who won the 2029 world cup",
  "निगम क्या है",
  "what is a corporation",
  "photosynthesis kaise hota hai",
];
let clean = 0, leaked = 0;
for (const q of DECOYS) {
  const r = await engine.ask(q, { skipCache: true });
  const top = r.citations[0]?.passageId;
  const got = top !== undefined ? store.lookup(engine.lastQueryVector, top) : null;
  got ? leaked++ : clean++;
  console.log(`${got ? "  LEAK" : "  ok  "} ${got ? `sim=${got.similarity.toFixed(4)} -> "${got.answer.slice(0, 40)}"` : "no match"}   ${q.slice(0, 40)}`);
}

console.log("\n" + "─".repeat(74));
console.log(`exact ${hit}/${hit + miss} matched · decoys ${clean}/${clean + leaked} correctly rejected`);
console.log("─".repeat(74));
if (miss || leaked) process.exit(1);
