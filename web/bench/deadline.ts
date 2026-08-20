/**
 * Does the 200 ms cap hold when the machine is not fast?
 *
 * Every other benchmark here measures speed on the machine this was built on,
 * where the answer is ~7 ms against a 200 ms budget. Shrinking the budget
 * instead exercises the same code path a 10x slower device hits at 200 ms:
 * embedding eats most of it, `remaining()` comes back small, and `budgetPlan`
 * has to choose a retrieval plan that fits.
 *
 * Two things are reported, because the first alone would not mean much:
 *
 *   1. whether the deadline was respected  (the claim)
 *   2. what recall it cost to respect it   (the price)
 *
 * A cap met by returning nothing useful is not worth having, so hit@3 is
 * measured at every budget against the full-budget result.
 *
 *   npx tsx bench/deadline.ts
 */

import { readFileSync } from "node:fs";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG } from "../src/harness/rag";
import { percentiles } from "../src/harness/pipeline";

const INDEX = process.env.INDEX ?? "public/index";
const QUERIES = process.env.QUERIES ?? "../pipeline/data/subset/queries.hin.jsonl";
const N = Number(process.env.N ?? 200);

/** The full budget first, then progressively less room than embedding needs. */
const BUDGETS = [200, 100, 50, 30, 20, 15, 12, 10, 8];

function loadIndexFromDisk() {
  const manifest: Manifest = JSON.parse(readFileSync(`${INDEX}/manifest.json`, "utf8"));
  const buf: Record<string, ArrayBuffer> = {};
  for (const n of indexBlobNames(manifest)) {
    const b = readFileSync(`${INDEX}/${n}`);
    buf[n] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  }
  const passages: string[] = [];
  for (const s of manifest.passageShards) {
    for (const p of JSON.parse(readFileSync(`${INDEX}/${s}`, "utf8")) as string[]) passages.push(p);
  }
  return assembleIndex(manifest, buf, passages);
}

function loadQueries(): string[] {
  const out: string[] = [];
  for (const line of readFileSync(QUERIES, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line) as { query?: string };
      if (row.query) out.push(row.query);
    } catch { /* skip malformed */ }
  }
  return out;
}

/**
 * The thresholds the deployed app uses, not `DEFAULT_THRESHOLDS`.
 *
 * The defaults are a conservative placeholder for the case where calibration has
 * never run, and against them the corpus refuses nearly every query — which
 * would leave this measuring kept@3 over a sample of about four.
 */
function fittedConfig() {
  try {
    const th = JSON.parse(readFileSync("public/thresholds.json", "utf8"));
    return {
      ...DEFAULT_CONFIG,
      thresholds: {
        ...DEFAULT_CONFIG.thresholds,
        minTopScore: th.minTopScore,
        minAgreement: th.minAgreement,
        minLexicalOverlap: th.minLexicalOverlap ?? 0,
      },
    };
  } catch {
    console.warn("! thresholds.json missing — run bench/calibrate.ts first.");
    console.warn("! Falling back to conservative defaults; kept@3 will be near-meaningless.\n");
    return DEFAULT_CONFIG;
  }
}

async function main(): Promise<void> {
  const index = loadIndexFromDisk();
  const enc = new E5Encoder();
  await enc.init();
  const queries = loadQueries().slice(0, N);
  const base = fittedConfig();
  console.log(`\nthresholds : minTopScore ${base.thresholds.minTopScore} ` +
              `minAgreement ${base.thresholds.minAgreement}`);

  console.log(`\nDEADLINE STRESS  —  n=${queries.length} queries per budget\n`);
  console.log("Shrinking the budget on a fast machine reproduces what a slow one");
  console.log("hits at 200ms. The plan should shrink to fit; latency should follow.\n");

  // Reference run at the full budget: what the system finds when nothing is
  // constraining it. Every reduced budget is scored against this, so "recall"
  // here means "kept what it would have found", not corpus ground truth.
  const reference = new Map<string, number[]>();
  {
    const engine = new RagEngine(index, enc, { ...base, globalBudgetMs: 200 });
    engine.setCorpusEnabled(true);   // the corpus is what this measures
    await engine.warmup();
    for (const q of queries) {
      const r = await engine.ask(q, { skipCache: true });
      reference.set(q, r.citations.map((c) => c.passageId));
    }
  }

  console.log(
    "budget".padStart(7) + "p50".padStart(9) + "p100".padStart(9) +
    "over".padStart(7) + "degraded".padStart(10) + "kept@3".padStart(9) +
    "n".padStart(6) + "  worst plan");
  console.log("-".repeat(78));

  let lowestHeld = Infinity;
  const breached: number[] = [];

  for (const budget of BUDGETS) {
    const engine = new RagEngine(index, enc, { ...base, globalBudgetMs: budget });
    engine.setCorpusEnabled(true);   // the corpus is what this measures
    await engine.warmup();
    // Extra warming per engine. Each budget gets a fresh RagEngine, and a fresh
    // engine's first few queries run interpreted — without this the P100 column
    // measures this harness warming up rather than the deadline logic working.
    for (let i = 0; i < 25; i++) await engine.ask(queries[i % queries.length], { skipCache: true });

    const times: number[] = [];
    let over = 0, degraded = 0, kept = 0, scored = 0;
    let worstPlan = "";
    let worstNprobe = Infinity;

    for (const q of queries) {
      const r = await engine.ask(q, { skipCache: true });
      times.push(r.totalMs);
      if (r.totalMs > budget) over++;
      if (r.plan.degraded) degraded++;
      if (r.plan.nprobe < worstNprobe) {
        worstNprobe = r.plan.nprobe;
        worstPlan = `nprobe ${r.plan.nprobe}, k ${r.plan.perStrategyK}, rescore ${r.plan.rescoreTopN}`;
      }
      const ref = reference.get(q) ?? [];
      if (ref.length) {
        scored++;
        const got = new Set(r.citations.map((c) => c.passageId));
        if (ref.some((id) => got.has(id))) kept++;
      }
    }

    const p = percentiles(times);
    const keptPct = scored ? (100 * kept / scored) : 100;
    if (over > 0) breached.push(budget); else lowestHeld = Math.min(lowestHeld, budget);

    console.log(
      `${budget}ms`.padStart(7) +
      p.p50.toFixed(2).padStart(9) +
      p.p100.toFixed(2).padStart(9) +
      String(over).padStart(7) +
      `${(100 * degraded / queries.length).toFixed(0)}%`.padStart(10) +
      `${keptPct.toFixed(1)}%`.padStart(9) +
      String(scored).padStart(6) +
      `  ${worstPlan}`);
  }

  console.log("-".repeat(78));
  console.log("\nover     : queries whose total exceeded that run's own budget");
  console.log("degraded : share where budgetPlan reduced the retrieval plan");
  console.log("kept@3   : share still returning a citation the 200ms run also returned");
  console.log("n        : denominator for kept@3 — queries the 200ms run answered");

  const claim = BUDGETS[0];
  const claimHeld = !breached.includes(claim);
  console.log(`\nVERDICT`);
  console.log(`  ${claim}ms requirement : ${claimHeld ? "MET" : "NOT MET"}`);
  console.log(`  cap held down to : ${lowestHeld}ms (${(claim / lowestHeld).toFixed(0)}x tighter than required)`);
  if (breached.length) {
    console.log(`  breached at      : ${breached.map((b) => `${b}ms`).join(", ")}`);
  }
  console.log("\nBelow ~10ms the budget is smaller than one embedding forward pass on");
  console.log("this machine, so no retrieval plan can recover it — that is the floor of");
  console.log("the technique, not a bug in the deadline logic. It is reported rather");
  console.log("than trimmed off the bottom of the table.\n");
}

main().catch((e) => { console.error(e); process.exit(1); });
