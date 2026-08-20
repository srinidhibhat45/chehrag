/**
 * What does a genuinely out-of-corpus query score?
 *
 * MS MARCO's "unanswerable" split is adversarial: those queries still carry ten
 * topically-relevant passages, so retrieval confidence barely separates them
 * (AUC 0.65). But gate 2's real production job is different — catching questions
 * the corpus has nothing to say about at all. This measures that case, which the
 * MS MARCO label cannot.
 */
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
const engine = new RagEngine(assembleIndex(manifest, buf, passages), await (async () => {
  const e = new E5Encoder(); await e.init(); return e;
})(), { ...DEFAULT_CONFIG, thresholds: { minTopScore: -1, minAgreement: 0, minLexicalOverlap: 0,
                rescueMinScore: Infinity, rescueMinOverlap: 1 } });
// The engine ships with the corpus off — the app starts empty so that an
// answer can only have come from what the user added. Every benchmark here
// exists to measure that corpus, so it opts in explicitly.
engine.setCorpusEnabled(true);
await engine.warmup();

const OUT_OF_CORPUS = [
  "who won the 2029 football world cup on mars",
  "what is my bank account balance right now",
  "मेरे फोन का पासवर्ड क्या है",
  "summarise the internal quarterly revenue of this company",
  "what did I have for breakfast yesterday",
  "translate this document into Klingon for me",
  "क्या कल बारिश होगी मेरे शहर में",
  "what is the airspeed velocity of an unladen swallow named Geoffrey",
];

const IN_CORPUS = readFileSync("../pipeline/data/subset/queries.hin.jsonl", "utf8")
  .trim().split("\n").map((l) => JSON.parse(l)).filter((q) => q.answerable).slice(0, 200);

console.log("=".repeat(72));
console.log("OUT-OF-CORPUS QUERIES (nothing in the index can answer these)");
console.log("=".repeat(72));
const outScores: number[] = [];
for (const q of OUT_OF_CORPUS) {
  const r = await engine.ask(q, { skipCache: true });
  outScores.push(r.confidence);
  console.log(`  ${r.confidence.toFixed(4)}  ${q.slice(0, 58)}`);
}

const inScores: number[] = [];
for (const q of IN_CORPUS) {
  const r = await engine.ask(q.query, { skipCache: true });
  inScores.push(r.confidence);
}

const s = (a: number[]) => [...a].sort((x, y) => x - y);
const qq = (a: number[], p: number) => s(a)[Math.floor(p * (a.length - 1))];
console.log("\n" + "-".repeat(72));
console.log(`out-of-corpus (n=${outScores.length})  min ${Math.min(...outScores).toFixed(3)}  ` +
            `p50 ${qq(outScores, .5).toFixed(3)}  max ${Math.max(...outScores).toFixed(3)}`);
console.log(`in-corpus     (n=${inScores.length})  p10 ${qq(inScores, .1).toFixed(3)}  ` +
            `p50 ${qq(inScores, .5).toFixed(3)}  p90 ${qq(inScores, .9).toFixed(3)}`);

console.log("\nseparation at candidate thresholds:");
for (const t of [0.40, 0.45, 0.50, 0.55, 0.60]) {
  const blocked = outScores.filter((x) => x < t).length;
  const kept = inScores.filter((x) => x >= t).length;
  console.log(`  thr ${t.toFixed(2)}  blocks ${blocked}/${outScores.length} out-of-corpus  ` +
              `keeps ${kept}/${inScores.length} (${(100 * kept / inScores.length).toFixed(0)}%) in-corpus`);
}
