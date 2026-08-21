/**
 * Cross-language tokeniser and hash check for the BM25 index.
 *
 * This one fails more quietly than the PCA and bit-packing parity tests it sits
 * beside. Those produce wrong numbers; this produces *no* numbers: if the
 * browser's tokeniser splits one character differently from the builder's, or
 * the two FNV implementations diverge on one byte, every lookup misses, the
 * lexical index returns nothing, and fusion carries on with six strategies
 * instead of seven. Nothing errors. The only visible symptom is a quality
 * number slightly lower than it should be — which is indistinguishable from
 * BM25 simply not helping much, the exact question the ablation is trying to
 * answer.
 *
 * Run `pipeline/src/lexical.py` first; it writes the fixture.
 */

import { readFileSync, existsSync } from "node:fs";
import { contentTokens } from "../src/retrieval/tokens";
import { fnv1a64 } from "../src/retrieval/lexical";

const FIXTURE = process.env.LEXPARITY ?? "/tmp/lexparity.json";

if (!existsSync(FIXTURE)) {
  console.error(`missing ${FIXTURE} — run: cd pipeline && .venv/bin/python src/lexical.py`);
  process.exit(1);
}

interface Case { text: string; tokens: string[]; hashes: [number, number][] }
const cases: Case[] = JSON.parse(readFileSync(FIXTURE, "utf8"));

let tokenFails = 0, hashFails = 0, tokensChecked = 0;
const examples: string[] = [];

for (const c of cases) {
  // The builder keeps duplicates (BM25 needs term frequency) and the browser
  // de-duplicates (gate 2 needs presence). Compare the sets, which is the
  // property that has to hold: the same term inventory on both sides.
  const mine = contentTokens(c.text);
  const theirs = new Set(c.tokens);

  const missing = [...theirs].filter((t) => !mine.has(t));
  const extra = [...mine].filter((t) => !theirs.has(t));
  if (missing.length || extra.length) {
    tokenFails++;
    if (examples.length < 5) {
      examples.push(
        `  "${c.text.slice(0, 48)}…"\n` +
        `    python-only: ${missing.slice(0, 6).join(", ") || "—"}\n` +
        `    js-only    : ${extra.slice(0, 6).join(", ") || "—"}`);
    }
  }

  // Hashes are checked against the builder's own dictionary ordering, which is
  // what the browser binary-searches.
  const uniq = [...new Set(c.tokens)];
  for (let i = 0; i < uniq.length && i < c.hashes.length; i++) {
    const got = fnv1a64(uniq[i]);
    const [hi, lo] = c.hashes[i];
    tokensChecked++;
    if (got.hi !== hi || got.lo !== lo) {
      hashFails++;
      if (examples.length < 8) {
        examples.push(`  hash "${uniq[i]}": js ${got.hi.toString(16)}${got.lo.toString(16).padStart(8, "0")} ` +
                      `vs py ${hi.toString(16)}${lo.toString(16).padStart(8, "0")}`);
      }
    }
  }
}

console.log(`fixtures      : ${cases.length}`);
console.log(`tokens hashed : ${tokensChecked}`);
console.log(`tokeniser     : ${cases.length - tokenFails}/${cases.length} agree`);
console.log(`hashes        : ${tokensChecked - hashFails}/${tokensChecked} agree`);
if (examples.length) console.log(`\n${examples.join("\n")}`);

const ok = tokenFails === 0 && hashFails === 0;
console.log(ok
  ? "\nPASS — the browser looks up exactly the terms the builder indexed"
  : "\nFAIL — lookups would silently miss; the lexical index would return nothing");
if (!ok) process.exit(1);
