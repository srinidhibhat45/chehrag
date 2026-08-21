/**
 * Answer generation — the "G" in RAG.
 *
 * Retrieval finds the passage that contains the answer; this turns it into one.
 * Without it the app shows the source line verbatim, which is a search result
 * rather than an answer.
 *
 * Generation cannot sit inside the 200 ms budget — a model round trip is
 * hundreds of milliseconds — so the two are timed and labelled separately:
 * `6.8 ms retrieval · 740 ms answer`. Retrieval remains the guarantee, runs
 * entirely in the browser, and is what the chip under each answer reports.
 *
 * The response streams, which is why this is not a plain JSON POST: the first
 * words land as the model produces them rather than after the whole answer
 * exists.
 *
 * Trust boundary: passage text comes from user documents and a shipped corpus
 * and is put in front of a language model. It is fenced as data in the prompt,
 * the model is told it is data, and the finished answer is checked back against
 * its passages (gate 3) before it is allowed to stand.
 *
 * The loop, and why it is here rather than in the Worker: the model answers by
 * calling a tool, and one of its tools is another search. The index lives in
 * this browser, so the search has to run here — the Worker holds the API key
 * and nothing else. So the Worker stays stateless and streams the tool call
 * back; this file executes it against the same sub-5 ms retrieval engine that
 * produced the first excerpts, appends the result to a transcript, and posts
 * again. At most one round, bounded by the Worker withholding the tool the
 * second time rather than by a counter here.
 */

/** One retrieved passage, with the document it came from. */
export interface GenSource {
  title: string;
  text: string;
}

/**
 * One prior step of the tool exchange.
 *
 * Mirrors the Worker's `Turn`. Deliberately duplicated rather than imported:
 * the browser bundle and the Worker are built separately and deployed
 * separately, and a shared type across that boundary would be a build-time
 * coupling standing in for a wire contract that is checked at runtime anyway.
 */
export type Turn =
  | { role: "assistant_tool"; id: string; name: string; args: string }
  | { role: "tool_result"; id: string; name: string; content: string };

export interface GenHandlers {
  /** Called with each fragment of answer text as it arrives. */
  onDelta(text: string): void;
  /**
   * Run a tool the model asked for. Returns the sources to add, and a line of
   * text describing the outcome for the model.
   *
   * Optional: with no executor the generator still runs, the tool is simply
   * never offered a result and the model is told so.
   */
  onTool?(name: string, args: Record<string, unknown>): Promise<ToolResult>;
  /** Called when a tool ran, for the telemetry line under the answer. */
  onToolNote?(note: string): void;
}

export interface ToolResult {
  /** Extra excerpts to append. Indices continue from the existing ones. */
  sources: GenSource[];
  /** What the model is told happened. Kept to one line. */
  summary: string;
}

export type GenOutcome =
  /**
   * A complete answer. `text` is the whole thing, already streamed via onDelta.
   *
   * `sources` is what the answer was actually written from, which after a
   * `search_corpus` call is not what was passed in. `cited` is the subset the
   * model says it used — gate 3 checks against those, so an answer citing an
   * excerpt that does not exist fails grounding rather than passing on the
   * strength of passages it never read.
   */
  | { kind: "answer"; text: string; ms: number; sources: GenSource[]; cited: number[]; toolCalls: number }
  /** The model read the passages and said they do not answer the question. */
  | { kind: "insufficient" }
  /** No generator is configured or reachable. The caller falls back. */
  | { kind: "unavailable"; reason: string };

/**
 * Where the generator lives.
 *
 * An empty base means same origin, which is what lets one code path serve both:
 * in development Vite answers `/synthesize` from a middleware reading
 * `web/.env.local`, and in production the deployed Worker serves the same route.
 * Neither ships the key to this file.
 */
export interface GenConfig {
  base: string;
  /** Abandon a generation that has produced nothing at all by this point. */
  firstTokenTimeoutMs: number;
  /** Hard ceiling on one generation, streaming or not. */
  totalTimeoutMs: number;
  /** Extra attempts after the first, for transport errors and 429/5xx only. */
  retries: number;
  /** First backoff step, doubled per attempt and jittered. */
  retryBaseMs: number;
};

export const DEFAULT_GEN_CONFIG: GenConfig = {
  base: "",
  firstTokenTimeoutMs: 12_000,
  totalTimeoutMs: 45_000,
  // Two attempts total. This is the slow path with an extractive answer already
  // on screen, so a retry costs a little latency and no correctness; more than
  // one would mostly serve to keep a visitor waiting on a provider that is down.
  retries: 1,
  retryBaseMs: 400,
};

/**
 * Ask the generator for an answer, streaming it back through `handlers`.
 *
 * Never throws. Every failure — no key, worker down, malformed stream, timeout —
 * returns `unavailable`, because the caller always holds a grounded extractive
 * answer to fall back on.
 */
export async function generate(
  query: string,
  sources: GenSource[],
  handlers: GenHandlers,
  cfg: GenConfig = DEFAULT_GEN_CONFIG,
): Promise<GenOutcome> {
  if (!sources.length) return { kind: "unavailable", reason: "no retrieved passages" };

  const started = performance.now();
  /**
   * Grows only by appending, so an excerpt's index never changes underneath a
   * citation the model already wrote.
   */
  const working = [...sources];
  const transcript: Turn[] = [];
  let toolCalls = 0;

  // Two rounds at most: the first, and one more after a search. The Worker
  // enforces the same bound from its side by withholding the tool once the
  // transcript shows it has run, so neither end depends on the other to
  // terminate.
  for (let round = 0; round < 2; round++) {
    const turn = await requestTurn(query, working, transcript, handlers, cfg);

    if (turn.kind === "tool") {
      if (!handlers.onTool) {
        // No executor wired up. Tell the model rather than silently looping.
        transcript.push({ role: "assistant_tool", id: turn.id, name: turn.name, args: turn.rawArgs });
        transcript.push({ role: "tool_result", id: turn.id, name: turn.name,
                          content: "That tool is unavailable. Answer from the excerpts you have." });
        continue;
      }
      let result: ToolResult;
      try {
        result = await handlers.onTool(turn.name, turn.args);
      } catch (err) {
        result = { sources: [], summary: `The search failed (${err instanceof Error ? err.message : "error"}).` };
      }
      toolCalls++;
      const from = working.length;
      for (const s of result.sources) working.push(s);
      const added = working.length - from;
      handlers.onToolNote?.(
        added
          ? `searched again for "${String(turn.args.query ?? "").slice(0, 60)}" — ${added} more excerpt${added === 1 ? "" : "s"}`
          : `searched again for "${String(turn.args.query ?? "").slice(0, 60)}" — nothing new`);
      transcript.push({ role: "assistant_tool", id: turn.id, name: turn.name, args: turn.rawArgs });
      transcript.push({
        role: "tool_result", id: turn.id, name: turn.name,
        content: added
          ? `${result.summary} They are excerpts ${from + 1}-${working.length}.`
          : result.summary,
      });
      continue;
    }

    if (turn.kind === "answer") {
      return { kind: "answer", text: turn.text, ms: performance.now() - started,
               sources: working, cited: turn.cited, toolCalls };
    }
    return turn;                     // insufficient | unavailable — both terminal
  }

  // Fell out of the loop: the model asked for a tool on its last allowed round.
  return { kind: "unavailable", reason: "tool loop did not converge" };
}

/** One turn of the exchange, in the shape the Worker's SSE protocol produces. */
type TurnOutcome =
  | { kind: "answer"; text: string; cited: number[] }
  | { kind: "tool"; id: string; name: string; args: Record<string, unknown>; rawArgs: string }
  | { kind: "insufficient" }
  | { kind: "unavailable"; reason: string };

async function requestTurn(
  query: string,
  sources: GenSource[],
  transcript: Turn[],
  handlers: GenHandlers,
  cfg: GenConfig,
): Promise<TurnOutcome> {
  const abort = new AbortController();

  // Two timers, because they fail for different reasons. Silence at the start
  // means the request never began — cold worker, bad key, network black hole —
  // and there is nothing to wait for. Silence afterwards means the model is
  // mid-sentence, and cutting it off discards an answer already arriving.
  let sawToken = false;
  const firstTokenTimer = setTimeout(() => {
    if (!sawToken) abort.abort(new Error("no first token"));
  }, cfg.firstTokenTimeoutMs);
  const totalTimer = setTimeout(() => abort.abort(new Error("generation timeout")), cfg.totalTimeoutMs);

  try {
    const res = await postWithRetry(
      `${cfg.base}/synthesize`,
      JSON.stringify({ query, sources, transcript }),
      abort.signal,
      cfg,
    );

    if (!res.ok || !res.body) {
      // 503 means no key is configured, which is a supported state rather than
      // a failure. The UI distinguishes it from a generator that is down.
      return { kind: "unavailable", reason: res.status === 503 ? "no generator configured" : `http ${res.status}` };
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let full = "";
    let cited: number[] = [];
    let insufficient = false;
    let serverError: string | null = null;
    let tool: { id: string; name: string; args: string } | null = null;

    // Minimal SSE: one JSON object per `data:` line, deliberately not the
    // upstream event schema. The browser should not track a vendor's stream
    // format, and a narrow protocol is one less thing to keep in step across
    // the two server implementations.
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // A chunk boundary can land mid-line, so the trailing partial is kept.
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        let ev: {
          t?: string; done?: boolean; insufficient?: boolean; error?: string;
          cited?: number[]; tool?: { id: string; name: string; args: string };
        };
        try { ev = JSON.parse(payload); } catch { continue; }

        if (ev.error) { serverError = ev.error; continue; }
        if (ev.insufficient) { insufficient = true; continue; }
        if (ev.tool) { tool = ev.tool; continue; }
        if (Array.isArray(ev.cited)) { cited = ev.cited; continue; }
        if (typeof ev.t === "string" && ev.t) {
          if (!sawToken) { sawToken = true; clearTimeout(firstTokenTimer); }
          full += ev.t;
          handlers.onDelta(ev.t);
        }
      }
    }

    if (serverError) return { kind: "unavailable", reason: serverError };
    if (tool) {
      // A tool call ends this turn even if text arrived alongside it.
      let args: Record<string, unknown> = {};
      try {
        const v = JSON.parse(tool.args || "{}");
        if (v && typeof v === "object") args = v as Record<string, unknown>;
      } catch { /* a malformed call still runs, with no arguments */ }
      return { kind: "tool", id: tool.id, name: tool.name, args, rawArgs: tool.args };
    }
    if (insufficient) return { kind: "insufficient" };

    const text = full.trim();
    if (!text) return { kind: "unavailable", reason: "empty generation" };
    return { kind: "answer", text, cited };
  } catch (err) {
    return { kind: "unavailable", reason: err instanceof Error ? err.message : String(err) };
  } finally {
    clearTimeout(firstTokenTimer);
    clearTimeout(totalTimer);
  }
}

/**
 * POST with a bounded retry.
 *
 * Only for faults that a second attempt can plausibly fix: a transport error,
 * or a 429/5xx from the provider. A 4xx is a request this client got wrong and
 * will keep getting wrong, and retrying a 401 just spends the rate limit twice.
 *
 * Jittered, because every visitor's page retries on the same schedule
 * otherwise, and a provider blip turns into a self-inflicted thundering herd
 * the moment it recovers.
 */
async function postWithRetry(
  url: string, body: string, signal: AbortSignal, cfg: GenConfig,
): Promise<Response> {
  let last: Response | null = null;
  for (let attempt = 0; attempt <= cfg.retries; attempt++) {
    if (attempt) {
      const wait = cfg.retryBaseMs * 2 ** (attempt - 1) * (0.5 + Math.random());
      await new Promise((r) => setTimeout(r, wait));
      if (signal.aborted) break;
    }
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
        signal,
      });
      if (res.ok || (res.status < 500 && res.status !== 429)) return res;
      last = res;
      // The body has to be drained or the connection is held open.
      await res.text().catch(() => "");
    } catch (err) {
      if (signal.aborted) throw err;
      last = null;
    }
  }
  return last ?? new Response(null, { status: 502 });
}
