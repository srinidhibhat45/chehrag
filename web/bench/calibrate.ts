/**
 * Fit the guardrail-gate-2 confidence threshold on real data.
 *
 * The corpus ships 3,012 queries with NO relevant passage — genuine cases where
 * refusing is the correct behaviour. That makes "when should we decline?" an
 * empirical question rather than a vibe, so we answer it empirically.
 *
 * Protocol:
 *   - split queries into calibration / holdout (no threshold is ever chosen on
 *     the holdout; that is what makes the reported number meaningful)
 *   - sweep the score threshold over the calibration split
 *   - pick the operating point, then report it on the holdout
 *
 * Objective, which measurement changed.
 *
 * Maximising F1 on abstention selects threshold 0.68 and refuses 392 of 549
 * answerable queries: 71% over-refusal, technically optimal and useless.
 *
 * The reason is that MS MARCO's "unanswerable" split is adversarial. Those
 * queries still carry ten topically-relevant passages retrieved by the original
 * search engine, so they are on-topic but non-answering. Measured separability
 * is weak: AUC 0.6497 on topScore (bench/signals.ts), and no combination of
 * margin / agreement / lexical overlap improved on it.
 *
 * Gate 2's real production job is different and much easier: rejecting questions
 * the corpus has nothing to say about. Measured on genuinely out-of-corpus
 * queries (bench/offtopic.ts), threshold 0.50 blocks 7/8 while keeping 84% of
 * in-corpus queries answerable.
 *
 * So selection is by an explicit coverage policy rather than F1: maximise
 * abstention recall subject to still answering at least MIN_COVERAGE of
 * answerable queries. That target is a product decision rather than an artefact
 * of an objective function.
 */

/** Answer at least this share of answerable queries. Product decision. */
const MIN_COVERAGE = 0.85;

import { readFileSync, existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG } from "../src/harness/rag";

const INDEX_DIR = process.env.INDEX_DIR ?? "public/index";
const QUERIES = process.env.QUERIES ?? "../pipeline/data/subset/queries.hin.jsonl";
const N = Number(process.env.N ?? 1200);

interface QRec { query: string; answerable: boolean; query_type: string; }

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
  return assembleIndex(manifest, buf, passages);
}

async function main() {
  if (!existsSync(join(INDEX_DIR, "manifest.json"))) {
    console.error(`no index at ${INDEX_DIR} — build it first`);
    process.exit(1);
  }
  const index = loadFromDisk();
  const enc = new E5Encoder();
  await enc.init();

  // Thresholds are disabled during collection so every query yields a score.
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

  const all: QRec[] = readFileSync(QUERIES, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const sample = all.slice(0, N);

  type Row = { score: number; agreement: number; answerable: boolean };
  const rows: Row[] = [];
  for (const q of sample) {
    const r = await engine.ask(q.query, { skipCache: true });
    rows.push({
      score: r.citations[0]?.score ?? -1,
      agreement: r.citations[0]?.strategies.length ?? 0,
      answerable: q.answerable,
    });
  }

  // deterministic split
  const calib = rows.filter((_, i) => i % 2 === 0);
  const hold  = rows.filter((_, i) => i % 2 === 1);

  /**
   * Abstention is the positive class. Names are spelled out because the short
   * forms invite exactly the mix-up that produced nonsense numbers here once.
   *
   *   correctAbstain  refused & unanswerable   (TP)
   *   overRefusal     refused & answerable     (FP — false alarm)
   *   missedAbstain   answered & unanswerable  (FN — the dangerous one)
   *   correctAnswer   answered & answerable    (TN)
   */
  function evaluate(rs: Row[], thr: number, minAgree: number) {
    let correctAbstain = 0, overRefusal = 0, missedAbstain = 0, correctAnswer = 0;
    for (const r of rs) {
      const refuse = r.score < thr || r.agreement < minAgree;
      if (refuse) { r.answerable ? overRefusal++ : correctAbstain++; }
      else { r.answerable ? correctAnswer++ : missedAbstain++; }
    }
    const precision = correctAbstain + overRefusal
      ? correctAbstain / (correctAbstain + overRefusal) : 0;
    const recall = correctAbstain + missedAbstain
      ? correctAbstain / (correctAbstain + missedAbstain) : 0;
    const coverage = correctAnswer + overRefusal
      ? correctAnswer / (correctAnswer + overRefusal) : 0;
    return {
      correctAbstain, overRefusal, missedAbstain, correctAnswer,
      precision, recall, coverage,
      f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0,
      acc: (correctAbstain + correctAnswer) / rs.length,
    };
  }

  const scores = rows.map((r) => r.score).filter((s) => s > -1).sort((a, b) => a - b);
  const lo = scores[0] ?? 0, hi = scores[scores.length - 1] ?? 1;
  console.log("=".repeat(74));
  console.log("GUARDRAIL CALIBRATION — gate 2 (retrieval confidence)");
  console.log("=".repeat(74));
  console.log(`queries ${rows.length}  (calib ${calib.length} / holdout ${hold.length})`);
  console.log(`answerable ${rows.filter((r) => r.answerable).length}  ` +
              `unanswerable ${rows.filter((r) => !r.answerable).length}`);
  console.log(`score range ${lo.toFixed(4)} .. ${hi.toFixed(4)}\n`);

  console.log("  thr    agree |  absPrec  absRec     F1   cover | answered");
  console.log("  " + "-".repeat(64));
  // Policy selection: strongest abstention subject to the coverage floor.
  let best = { thr: lo, agree: 1, f1: 0, prec: 0, rec: -1 };
  for (let t = lo; t <= hi; t += (hi - lo) / 200) {
    for (const ag of [1, 2]) {
      const e = evaluate(calib, t, ag);
      if (e.coverage < MIN_COVERAGE) continue;
      if (e.recall > best.rec) best = { thr: t, agree: ag, f1: e.f1, prec: e.precision, rec: e.recall };
    }
  }
  console.log(`policy: keep >= ${(MIN_COVERAGE * 100).toFixed(0)}% of answerable queries answerable\n`);
  // print a readable sweep at the chosen agreement level
  for (let i = 0; i <= 12; i++) {
    const t = lo + (hi - lo) * (i / 12);
    const e = evaluate(calib, t, best.agree);
    const mark = Math.abs(t - best.thr) < (hi - lo) / 80 ? "  <-- selected" : "";
    console.log(`  ${t.toFixed(3)}   ${best.agree}    |  ${e.precision.toFixed(3)}   ${e.recall.toFixed(3)}  ` +
                `${e.f1.toFixed(3)}  ${(e.coverage * 100).toFixed(0).padStart(3)}% |  ${e.correctAnswer + e.missedAbstain}${mark}`);
  }

  const h = evaluate(hold, best.thr, best.agree);
  console.log("\n" + "-".repeat(74));
  console.log(`SELECTED  minTopScore=${best.thr.toFixed(4)}  minAgreement=${best.agree}`);
  console.log("-".repeat(74));
  console.log("HOLDOUT (never used for selection):");
  console.log(`  coverage of answerable: ${(h.coverage * 100).toFixed(1)}%  (policy floor ${(MIN_COVERAGE * 100).toFixed(0)}%)`);
  console.log(`  correct abstentions   : ${h.correctAbstain}`);
  console.log(`  missed abstentions    : ${h.missedAbstain}   <- answered when it should not have`);
  console.log(`  over-refusals         : ${h.overRefusal}   <- refused a real answer`);
  console.log(`  correct answers       : ${h.correctAnswer}`);
  console.log(`  abstention precision  : ${h.precision.toFixed(4)}`);
  console.log(`  abstention recall     : ${h.recall.toFixed(4)}`);
  console.log(`  F1                    : ${h.f1.toFixed(4)}`);

  const out = {
    minTopScore: +best.thr.toFixed(4),
    minAgreement: best.agree,
    minLexicalOverlap: 0,
    fittedOn: calib.length,
    holdout: { n: hold.length, precision: +h.precision.toFixed(4),
               recall: +h.recall.toFixed(4), f1: +h.f1.toFixed(4),
               coverage: +h.coverage.toFixed(4) },
    fittedAt: new Date().toISOString(),
  };
  writeFileSync("public/thresholds.json", JSON.stringify(out, null, 2));
  console.log("\nwrote public/thresholds.json");
  console.log("=".repeat(74));
}

main().catch((e) => { console.error(e); process.exit(1); });
