/**
 * Multilingual stress test — the offline half.
 *
 * The corpus is Hindi and the embedder is multilingual, which is a claim about
 * cross-lingual retrieval. This measures it: the same 500 MS MARCO queries in
 * fourteen Indian languages plus English, joined on `query_id`, asked against a
 * Hindi index.
 *
 * Runs the SAME modules the browser runs, so the numbers describe shipped code.
 * The scoring logic itself lives in `src/eval/multilingual.ts` and is shared
 * with the in-app runner — a second implementation here would be a second thing
 * to keep correct, and the two would drift.
 *
 * The refusal histogram is the part worth reading. Latency barely moves across
 * languages; what moves is why a language gets refused, and a per-reason
 * breakdown is the only way to tell "retrieval could not find it" apart from
 * "a gate turned it away before retrieval ran".
 *
 *   npx tsx web/bench/multilingual.ts
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG, type RagConfig, type RagAnswer } from "../src/harness/rag";
import {
  runMultilingual, formatReport, ENGLISH, type MultilingualData,
} from "../src/eval/multilingual";

const INDEX_DIR = process.env.INDEX_DIR ?? "public/index";
const DATA = process.env.ML_DATA ?? "bench/data/multilingual.json";
const PER_LANG = Number(process.env.PER_LANG ?? 120);

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

async function main() {
  if (!existsSync(join(INDEX_DIR, "manifest.json"))) {
    console.error(`no index at ${INDEX_DIR} — run pipeline/src/build_index.py first`);
    process.exit(1);
  }
  if (!existsSync(DATA)) {
    console.error(`no parallel queries at ${DATA} — run pipeline/src/parallel_queries.py first`);
    process.exit(1);
  }

  const { index, manifest } = loadFromDisk();
  const data: MultilingualData = JSON.parse(readFileSync(DATA, "utf8"));

  const encoder = new E5Encoder();
  await encoder.init();

  let cfg: RagConfig = { ...DEFAULT_CONFIG };
  const thPath = "public/thresholds.json";
  if (existsSync(thPath)) {
    const t = JSON.parse(readFileSync(thPath, "utf8"));
    cfg = { ...cfg, thresholds: {
      ...cfg.thresholds,
      minTopScore: t.minTopScore, minAgreement: t.minAgreement,
      minLexicalOverlap: t.minLexicalOverlap ?? 0,
    } };
  } else {
    console.warn("WARNING no thresholds.json — using conservative defaults, " +
                 "which refuse far more than the calibrated values");
  }

  const engine = new RagEngine(index, encoder, cfg);
  // The engine ships with the corpus off — the app starts empty so that an
  // answer can only have come from what the user added. Every benchmark here
  // exists to measure that corpus, so it opts in explicitly.
  engine.setCorpusEnabled(true);
  await engine.warmup();

  console.log("=".repeat(74));
  console.log(`corpus ${manifest.numPassages.toLocaleString()} Hindi passages · ` +
              `${data.languages.length + 1} languages · ${PER_LANG} queries each`);

  // Refusal reasons, per language. Collected by wrapping `ask` rather than
  // by extending the shared runner: this is a diagnostic for tuning gates, not
  // part of what the test reports, and the browser has no use for it.
  const refusals = new Map<string, Map<string, number>>();
  let currentLang = "";
  const ask = async (q: string): Promise<RagAnswer> => {
    const r = await engine.ask(q, { skipCache: true });
    if (r.status === "refused" && r.refusal) {
      const m = refusals.get(currentLang) ?? new Map<string, number>();
      m.set(r.refusal, (m.get(r.refusal) ?? 0) + 1);
      refusals.set(currentLang, m);
    }
    return r;
  };

  const langs = [...data.languages, ENGLISH];
  const rep = await runMultilingual(data, ask, {
    perLang: PER_LANG,
    onProgress: (done, _total, label) => {
      currentLang = langs[done]?.code ?? label;
      process.stderr.write(`  ${label}…\r`);
    },
  });
  process.stderr.write(" ".repeat(40) + "\r");

  console.log(formatReport(rep));

  console.log("REFUSALS BY REASON");
  console.log("-".repeat(74));
  const reasons = [...new Set([...refusals.values()].flatMap((m) => [...m.keys()]))].sort();
  console.log(`${"language".padEnd(12)}${reasons.map((r) => r.slice(0, 13).padStart(15)).join("")}`);
  for (const l of langs) {
    const m = refusals.get(l.code);
    if (!m) { console.log(`${l.name.padEnd(12)}${reasons.map(() => "-".padStart(15)).join("")}`); continue; }
    console.log(`${l.name.padEnd(12)}${reasons.map((r) => String(m.get(r) ?? 0).padStart(15)).join("")}`);
  }
  console.log("-".repeat(74));
  console.log(
    "Read this alongside hit@5. A language with high hit@5 and a large\n" +
    "LOW_CONFIDENCE count is retrieving correctly and being refused by a\n" +
    "threshold fitted on Hindi. A large GIBBERISH or NOT_A_QUESTION count is a\n" +
    "gate-1 bug: those fire before any embedding exists, so they cannot be a\n" +
    "statement about relevance.",
  );
}

main().catch((e) => { console.error(e); process.exit(1); });
