/**
 * The tool-call harness, exercised against the real provider.
 *
 * The generation path is not prompt-in, text-out: the model answers by calling
 * a tool, names the excerpts it used, and may ask for one more search - which
 * the client executes, because the index is in the browser and the Worker
 * holds nothing but an API key. That is several moving parts across a network,
 * and the failure modes are quiet ones: a provider that ignores tools, a model
 * that cites excerpt 7 of 3, a search that never converges.
 *
 * So this runs the whole loop end to end and counts what happened:
 *
 *   answered / insufficient / unavailable   how turns ended
 *   tool calls                              how often a second search was asked for
 *   citations                               valid, fabricated, or absent
 *   gate 3                                  what survived grounding
 *
 * It calls a paid API, so it defaults to a small sample and paces itself. The
 * numbers that matter here are proportions and failures, not latency - the
 * generator is off the measured path by construction.
 *
 *   npx tsx bench/tools.ts                 15 queries
 *   npx tsx bench/tools.ts --limit 40      more
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { assembleIndex, indexBlobNames, type Manifest } from "../src/retrieval/loader";
import { E5Encoder } from "../src/retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG } from "../src/harness/rag";
import { synthesizeStream, resolveProvider, type Turn } from "../../worker/src/synthesize";
import { generate, type GenSource, type ToolResult } from "../src/answer/generate";

const INDEX_DIR = process.env.INDEX_DIR ?? "public/index";
const QUERIES = process.env.QUERIES ?? "../pipeline/data/subset/queries.hin.jsonl";
const argv = process.argv.slice(2);
const argOf = (n: string, d: number) => {
  const i = argv.indexOf(n);
  return i >= 0 && argv[i + 1] ? Number(argv[i + 1]) : d;
};
const LIMIT = argOf("--limit", 15);
/** Free-tier token budget is the binding constraint; see bench/precompute.ts. */
const GAP_MS = argOf("--gap", 9000);

function dotenv(path: string): Record<string, string> | null {
  if (!existsSync(path)) return null;
  const out: Record<string, string> = {};
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const m = /^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/.exec(line);
    if (m) out[m[1]] = m[2].trim().replace(/^["']|["']$/g, "");
  }
  return out;
}

function loadFromDisk() {
  const manifest: Manifest = JSON.parse(readFileSync(join(INDEX_DIR, "manifest.json"), "utf8"));
  const buf: Record<string, ArrayBuffer> = {};
  for (const n of indexBlobNames(manifest)) {
    const b = readFileSync(join(INDEX_DIR, n));
    buf[n] = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  }
  const passages: string[] = [];
  for (const sh of manifest.passageShards ?? ["passages.json"]) {
    for (const s of JSON.parse(readFileSync(join(INDEX_DIR, sh), "utf8")) as string[]) passages.push(s);
  }
  return assembleIndex(manifest, buf, passages);
}

/**
 * The loop, against a scripted provider.
 *
 * The live run below only exercises `search_corpus` if the model decides it
 * needs one, which on well-retrieved queries it correctly does not - so the
 * most important path would go untested precisely when everything is working.
 * This scripts the provider instead: turn one asks for a search, turn two
 * answers citing an excerpt that only exists because the search ran.
 *
 * It costs nothing, needs no key, and tests the part that is ours: the tool
 * dispatch, the transcript, the appended excerpts, and the bound on rounds.
 */
async function scriptedLoopTest(): Promise<boolean> {
  const sse = (obj: unknown) => `data: ${JSON.stringify(obj)}\n\n`;
  let turn = 0;
  const seen: { transcripts: number[]; sources: number[] } = { transcripts: [], sources: [] };

  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as {
      sources: GenSource[]; transcript?: Turn[];
    };
    seen.transcripts.push(body.transcript?.length ?? 0);
    seen.sources.push(body.sources.length);
    turn++;
    const script = turn === 1
      // Arguments split across two frames, as a real provider streams them.
      ? sse({ tool: { id: "call_1", name: "search_corpus", args: '{"query":"rephrased terms"}' } })
      : sse({ t: "The answer" }) + sse({ cited: [body.sources.length] }) + sse({ done: true });
    return new Response(script, { status: 200 });
  }) as typeof fetch;

  let executed = 0;
  const out = await generate("original question", [{ title: "doc", text: "first excerpt" }], {
    onDelta: () => {},
    onTool: async (name, args) => {
      executed++;
      return {
        sources: name === "search_corpus" && args.query === "rephrased terms"
          ? [{ title: "doc", text: "the excerpt the second search found" }]
          : [],
        summary: "Found 1 further excerpt.",
      };
    },
  }, { base: "", firstTokenTimeoutMs: 5000, totalTimeoutMs: 10_000, retries: 0, retryBaseMs: 1 });
  globalThis.fetch = realFetch;

  const checks: Array<[string, boolean]> = [
    ["the tool was dispatched to the client exactly once", executed === 1],
    ["the loop produced an answer", out.kind === "answer"],
    ["the tool call is reported", out.kind === "answer" && out.toolCalls === 1],
    ["turn 1 carried an empty transcript", seen.transcripts[0] === 0],
    ["turn 2 replayed the call and its result", seen.transcripts[1] === 2],
    ["turn 2 carried the appended excerpt", seen.sources[1] === 2],
    ["the answer's sources include what the search found",
      out.kind === "answer" && out.sources.length === 2
      && out.sources[1].text.includes("second search")],
    ["the citation resolves to the appended excerpt",
      out.kind === "answer" && out.cited.length === 1 && out.cited[0] === 2],
  ];
  console.log("=".repeat(72));
  console.log("TOOL LOOP - scripted provider (no API calls)");
  console.log("=".repeat(72));
  for (const [label, ok] of checks) console.log(`${ok ? "  ok  " : "  FAIL"} ${label}`);
  const passed = checks.every(([, ok]) => ok);
  console.log(passed ? "\nPASS - dispatch, transcript and citation indexing all hold\n"
                     : "\nFAIL\n");
  return passed;
}

async function main() {
  const scriptedOk = await scriptedLoopTest();
  if (argv.includes("--scripted-only")) process.exit(scriptedOk ? 0 : 1);

  const provider = resolveProvider(process.env as Record<string, string>)
    ?? resolveProvider(dotenv("../web/.env.local") ?? dotenv(".env.local") ?? {});
  if (!provider) {
    console.error("no generator configured - put GROQ_API_KEY in web/.env.local");
    process.exit(1);
  }

  const index = loadFromDisk();
  const enc = new E5Encoder();
  await enc.init();
  // The fitted threshold, as the app ships it. `DEFAULT_CONFIG` carries the
  // conservative placeholder for the case where calibration has never run, and
  // using it here would refuse nearly everything before a model was ever
  // called - measuring gate 2's placeholder rather than the tool loop.
  let cfg = { ...DEFAULT_CONFIG };
  try {
    const th = JSON.parse(readFileSync("public/thresholds.json", "utf8"));
    cfg = { ...cfg, thresholds: { ...cfg.thresholds,
      minTopScore: th.minTopScore, minAgreement: th.minAgreement,
      minLexicalOverlap: th.minLexicalOverlap ?? 0 } };
  } catch { /* placeholders */ }
  const engine = new RagEngine(index, enc, cfg);
  engine.setCorpusEnabled(true);
  await engine.warmup();

  const all = readFileSync(QUERIES, "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const sample = all.filter((q: { answerable: boolean }) => q.answerable).slice(0, LIMIT);

  console.log("=".repeat(72));
  console.log("TOOL-CALL HARNESS - live");
  console.log("=".repeat(72));
  console.log(`provider   : ${provider.label}`);
  console.log(`queries    : ${sample.length}\n`);

  let answered = 0, insufficient = 0, unavailable = 0;
  let toolCalls = 0, withCitations = 0, fabricated = 0, grounded = 0, ungrounded = 0;
  const reasons: Record<string, number> = {};

  /**
   * The same loop `ui/main.ts` runs, with the same tool executor.
   *
   * `generate` posts to a URL, so the fetch is redirected at the Worker's own
   * stream function here - no Worker process, no dev server, and no second
   * implementation of the protocol to keep in step.
   */
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    const href = typeof url === "string" ? url : url instanceof URL ? url.href : url.url;
    if (!href.endsWith("/synthesize")) return realFetch(url as never, init);
    const body = JSON.parse(String(init?.body ?? "{}")) as {
      query: string; sources: GenSource[]; transcript?: Turn[];
    };
    return new Response(
      synthesizeStream(provider, body.query, body.sources, body.transcript ?? []),
      { status: 200, headers: { "content-type": "text/event-stream" } });
  }) as typeof fetch;

  for (const q of sample) {
    const base = await engine.ask(q.query, { skipCache: true });
    if (base.status !== "answered") { process.stdout.write("·"); continue; }

    const sources: GenSource[] = base.citations.map((c) => ({
      title: c.source.kind === "user" ? (c.source.title ?? "your source") : "MS MARCO-XI",
      text: c.text,
    }));
    const shown = new Set(base.citations.map((c) => c.passageId));

    const out = await generate(q.query, sources, {
      onDelta: () => {},
      onTool: async (name, args): Promise<ToolResult> => {
        if (name !== "search_corpus") return { sources: [], summary: "Unknown tool." };
        const found = await engine.searchAgain(String(args.query ?? ""), shown);
        for (const c of found) shown.add(c.passageId);
        return {
          sources: found.map((c) => ({ title: "MS MARCO-XI", text: c.text })),
          summary: found.length ? `Found ${found.length} further excerpts.` : "Nothing new.",
        };
      },
    }, { base: "", firstTokenTimeoutMs: 20_000, totalTimeoutMs: 60_000, retries: 1, retryBaseMs: 500 });

    if (out.kind === "answer") {
      answered++;
      toolCalls += out.toolCalls;
      if (out.cited.length) withCitations++;
      const g = engine.verifyGenerated(out.text, out.sources, out.cited);
      if (g.pass) grounded++;
      else {
        ungrounded++;
        reasons[g.reason ?? "?"] = (reasons[g.reason ?? "?"] ?? 0) + 1;
        if (g.reason === "FABRICATED_CITATION") fabricated++;
      }
      process.stdout.write(out.toolCalls ? "T" : "a");
    } else if (out.kind === "insufficient") {
      insufficient++; process.stdout.write("i");
    } else {
      unavailable++; process.stdout.write("x");
      reasons[out.reason] = (reasons[out.reason] ?? 0) + 1;
    }

    await new Promise((r) => setTimeout(r, GAP_MS));
  }

  const n = answered + insufficient + unavailable;
  console.log("\n\n" + "-".repeat(72));
  console.log(`turns        : ${n}`);
  console.log(`  answered   : ${answered}`);
  console.log(`  insufficient: ${insufficient}   model read the passages and declined`);
  console.log(`  unavailable : ${unavailable}   provider or transport failed`);
  console.log(`tool calls   : ${toolCalls}   (search_corpus, executed client-side)`);
  console.log(`citations    : ${withCitations}/${answered} answers named their excerpts`);
  console.log(`  fabricated : ${fabricated}   cited an excerpt that was not supplied`);
  console.log(`gate 3       : ${grounded} passed, ${ungrounded} rejected`);
  if (Object.keys(reasons).length) console.log(`reasons      : ${JSON.stringify(reasons)}`);
  console.log("-".repeat(72));
  const ok = scriptedOk && answered > 0 && unavailable === 0;
  console.log(ok
    ? "PASS - the loop runs end to end against the real provider"
    : "FAIL - see reasons above");
  console.log("=".repeat(72));
  if (!ok) process.exit(1);
}

main().catch((e) => { console.error(e); process.exit(1); });
