/**
 * Gate 2, measured on user-added documents.
 *
 * `bench/calibrate.ts` fits `minTopScore` on the corpus's own 3,012 labelled
 * unanswerable queries. That split cannot measure the case this bench exists
 * for, because gate 2 treats the two corpora differently on purpose: the
 * mid-band rescue is applied only to passages that came from a document the
 * user added, precisely because absolute cosine does not transfer between
 * corpora. So the rescue rule has never been measured on the thing it governs.
 *
 * That gap had a cost. Asked "what is the name" of a document whose first line
 * is `name: srinidhi bhat, age: 45`, the system refused. Retrieval was not the
 * problem — the correct passage came back at rank 1, found by four independent
 * chunking strategies, sharing every content word with the question. It was
 * thrown away because its cosine was 0.263 and the rescue floor was 0.40.
 *
 * WHY THE COSINE IS LOW ON A PERFECT MATCH. e5 embeds a four-word question and
 * a line of form fields into vectors whose absolute similarity says more about
 * length and register than about relevance. The corpus this threshold was fitted
 * on is MS MARCO — passages written as prose, selected because they answered a
 * web query. A CV line, a table row, a heading with a value after it: none of
 * them are that, and all of them are what people actually upload.
 *
 * WHAT THIS MEASURES. Two documents, one field-structured and one prose, with
 * labelled answerable and unanswerable questions over each. It reports, for a
 * set of candidate rules:
 *
 *   coverage   share of answerable questions actually answered
 *   abstention share of unanswerable questions actually refused
 *
 * Both matter. A rule that answers everything scores 1.0 coverage and is
 * useless; the one that shipped scored well on abstention by refusing correct
 * answers, which is the failure that reached a user.
 *
 *   npx tsx bench/usersource.ts
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG } from "../src/harness/rag";
import { SourceStore } from "../src/sources/store";
import { gateRetrieval, type ConfidenceThresholds, type RetrievalSignals } from "../src/guardrails/gates";

const D = process.env.INDEX_DIR ?? "public/index";

// --- documents --------------------------------------------------------------

/**
 * Field-structured. This is the shape the shipped threshold handles worst and
 * that people upload most: a CV, a form, an invoice, a record export.
 */
const BIODATA = `name: srinidhi bhat, age: 45, sex: M, faults: crying

Srinidhi Bhat is a senior software engineer based in Bengaluru, India. He has
eleven years of experience building distributed systems, and previously led the
payments platform team at Vayu Labs from 2019 to 2024.

Education: B.E. in Computer Science, NITK Surathkal, 2013.
Languages spoken: Kannada, Hindi, English, Tulu.

Contact: reachable by email only. Notice period is 60 days.
Current salary: 48 lakh per annum. Expected: 62 lakh.`;

/** Ordinary prose, closer to what `minTopScore` was fitted on. */
const POLICY = `Refund and Returns Policy

Any item purchased from us may be returned within 30 days of delivery, provided
it is unused and in its original packaging. Refunds are issued to the original
payment method and take between five and seven working days to appear.

Items marked as final sale cannot be returned or exchanged under any
circumstances. This includes clearance stock and personalised goods.

Shipping charges are not refundable. If a return is the result of our error — a
damaged item, or the wrong item shipped — we cover return postage and refund the
original shipping cost in full.

To start a return, email support with your order number. Our support desk is
staffed Monday to Friday, nine to five.`;

interface Q { q: string; answerable: boolean; }

const QUESTIONS: Q[] = [
  // --- biodata, answerable -------------------------------------------------
  { q: "what is the name", answerable: true },
  { q: "what is my name", answerable: true },
  { q: "how old is srinidhi", answerable: true },
  { q: "what is the age", answerable: true },
  { q: "age?", answerable: true },
  { q: "WHAT IS SRINIDHI", answerable: true },
  { q: "where is he based", answerable: true },
  { q: "how many years of experience does he have", answerable: true },
  { q: "which company did he work at", answerable: true },
  { q: "what did he study", answerable: true },
  { q: "where did he go to college", answerable: true },
  { q: "what languages does he speak", answerable: true },
  { q: "what is the notice period", answerable: true },
  { q: "what is his expected salary", answerable: true },
  { q: "when did he leave vayu labs", answerable: true },
  { q: "what team did he lead", answerable: true },

  // --- biodata, unanswerable ----------------------------------------------
  { q: "what is his blood group", answerable: false },
  { q: "who is his manager", answerable: false },
  { q: "what is his phone number", answerable: false },
  { q: "is he married", answerable: false },
  { q: "what car does he drive", answerable: false },
  { q: "what is his github username", answerable: false },

  // --- policy, answerable --------------------------------------------------
  { q: "how long do I have to return something", answerable: true },
  { q: "what is the return window", answerable: true },
  { q: "how long do refunds take", answerable: true },
  { q: "can I return a clearance item", answerable: true },
  { q: "are shipping charges refundable", answerable: true },
  { q: "who pays return postage if you sent the wrong item", answerable: true },
  { q: "how do I start a return", answerable: true },
  { q: "when is support open", answerable: true },
  { q: "what condition does a returned item need to be in", answerable: true },

  // --- policy, unanswerable ------------------------------------------------
  { q: "do you ship internationally", answerable: false },
  { q: "what is your VAT number", answerable: false },
  { q: "can I pay in instalments", answerable: false },
  { q: "how much does express delivery cost", answerable: false },
  { q: "what is your warranty period", answerable: false },

  // --- nothing to do with either document ----------------------------------
  { q: "who won the 2029 world cup", answerable: false },
  { q: "what is the capital of Peru", answerable: false },
  { q: "how do I make sourdough bread", answerable: false },
];

// --- candidate rules --------------------------------------------------------

interface Rule {
  name: string;
  note: string;
  thresholds: ConfidenceThresholds;
}

const BASE = { minTopScore: 0.4788, minAgreement: 1, minLexicalOverlap: 0 };

const RULES: Rule[] = [
  {
    name: "shipped",
    note: "rescue floor 0.40, overlap 0.5 — what shipped before",
    thresholds: { ...BASE, rescueMinScore: 0.40, rescueMinOverlap: 0.5 },
  },
  {
    name: "no rescue",
    note: "absolute cosine only — what the corpus path does",
    thresholds: { ...BASE, rescueMinScore: Infinity, rescueMinOverlap: 1 },
  },
  {
    name: "floor 0.30",
    note: "same rule, lower floor",
    thresholds: { ...BASE, rescueMinScore: 0.30, rescueMinOverlap: 0.5 },
  },
  {
    name: "floor 0.20",
    note: "same rule, lower floor — SHIPPING",
    thresholds: { ...BASE, rescueMinScore: 0.20, rescueMinOverlap: 0.5 },
  },
  {
    name: "floor 0.15",
    note: "same rule, lower floor still",
    thresholds: { ...BASE, rescueMinScore: 0.15, rescueMinOverlap: 0.5 },
  },
  {
    name: "floor 0.15 / ov .34",
    note: "lower floor, and one content word in three is enough",
    thresholds: { ...BASE, rescueMinScore: 0.15, rescueMinOverlap: 0.34 },
  },
  {
    name: "floor 0.00",
    note: "overlap alone decides — no cosine floor at all",
    thresholds: { ...BASE, rescueMinScore: 0, rescueMinOverlap: 0.5 },
  },
];

// --- harness ----------------------------------------------------------------

const manifest: Manifest = JSON.parse(readFileSync(join(D, "manifest.json"), "utf8"));
const buf: Record<string, ArrayBuffer> = {};
for (const n of indexBlobNames(manifest)) {
  const b = readFileSync(join(D, n));
  buf[n] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
}
// Passage text is not needed: every question here is answered from the user
// documents, and the corpus stays off exactly as it does in the app.
const index = assembleIndex(manifest, buf, []);

const enc = new E5Encoder();
await enc.init();

const store = new SourceStore({
  encoder: enc,
  pcaMean: index.pcaMean,
  pcaComp: index.pcaComp,
  dim: manifest.dim,
  codeWords: manifest.codeWords,
  model: manifest.model,
  workerBase: "",
});
await store.addPaste(BIODATA, "biodata.pdf");
await store.addPaste(POLICY, "refund-policy.txt");

/**
 * The gate is evaluated offline rather than by re-running the engine per rule.
 * Retrieval is deterministic given the query, so the signals are collected once
 * and every candidate rule is scored against the same numbers — which is both
 * ~7x faster and, more importantly, means the rules are compared on identical
 * retrieval rather than on seven separate runs of it.
 */
const engine = new RagEngine(index, enc, {
  ...DEFAULT_CONFIG,
  // Wide open: we want the signals for every question, including the ones the
  // shipped gate would have refused.
  thresholds: { minTopScore: -1, minAgreement: 0, minLexicalOverlap: 0,
                rescueMinScore: Infinity, rescueMinOverlap: 1 },
});
// The engine now defaults the shipped corpus ON, because the app must answer
// from the provided dataset the moment it opens. This bench measures the *user
// source* gate specifically, so it opts out explicitly — otherwise corpus
// passages join the ranking, the rescue is disabled for the ones that win, and
// the numbers below quietly stop being about the thing being fitted. (They did:
// coverage read 64% instead of 88% until this line existed.)
engine.setCorpusEnabled(false);
engine.attachSources(store);
await engine.warmup();

interface Row { q: Q; signals: RetrievalSignals | undefined; }
const rows: Row[] = [];
for (const q of QUESTIONS) {
  const r = await engine.ask(q.q, { skipCache: true });
  rows.push({ q, signals: r.signals });
}

function evaluate(rule: Rule) {
  let ansTotal = 0, ansPassed = 0, unaTotal = 0, unaRefused = 0;
  const wrong: string[] = [];
  for (const { q, signals } of rows) {
    // No signals at all means retrieval returned nothing — a refusal under
    // every rule.
    const pass = signals ? gateRetrieval(signals, rule.thresholds).pass : false;
    if (q.answerable) {
      ansTotal++;
      if (pass) ansPassed++; else wrong.push(`refused: ${q.q}`);
    } else {
      unaTotal++;
      if (!pass) unaRefused++; else wrong.push(`answered: ${q.q}`);
    }
  }
  return {
    coverage: ansPassed / ansTotal,
    abstention: unaRefused / unaTotal,
    ansPassed, ansTotal, unaRefused, unaTotal, wrong,
  };
}

console.log("=".repeat(78));
console.log(`GATE 2 ON USER SOURCES — ${QUESTIONS.filter(q => q.answerable).length} answerable, ` +
            `${QUESTIONS.filter(q => !q.answerable).length} unanswerable, over 2 documents`);
console.log("=".repeat(78));
console.log("");
console.log("rule".padEnd(20) + "coverage".padEnd(14) + "abstention".padEnd(14) + "note");
console.log("-".repeat(78));

const results = RULES.map((r) => ({ rule: r, ...evaluate(r) }));
for (const r of results) {
  console.log(
    r.rule.name.padEnd(20) +
    `${(r.coverage * 100).toFixed(1)}% (${r.ansPassed}/${r.ansTotal})`.padEnd(14) +
    `${(r.abstention * 100).toFixed(1)}% (${r.unaRefused}/${r.unaTotal})`.padEnd(14) +
    r.rule.note,
  );
}

console.log("\n" + "-".repeat(78));
console.log("SIGNALS PER QUESTION (topScore / overlap / agreement / margin)");
console.log("-".repeat(78));
for (const { q, signals } of rows) {
  if (!signals) { console.log(`  ${q.answerable ? "A" : "U"}  —  no retrieval  ${q.q}`); continue; }
  console.log(
    `  ${q.answerable ? "A" : "U"}  ` +
    `${signals.topScore.toFixed(3)}  ${signals.lexicalOverlap.toFixed(2)}  ` +
    `${String(signals.strategyAgreement).padStart(2)}  ${signals.margin.toFixed(2)}   ${q.q}`,
  );
}

const detail = process.env.DETAIL === "1";
if (detail) {
  for (const r of results) {
    console.log(`\n--- ${r.rule.name} mistakes ---`);
    for (const w of r.wrong) console.log("   " + w);
  }
}
