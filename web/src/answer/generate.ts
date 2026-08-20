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
 */

/** One retrieved passage, with the document it came from. */
export interface GenSource {
  title: string;
  text: string;
}

export interface GenHandlers {
  /** Called with each fragment of answer text as it arrives. */
  onDelta(text: string): void;
}

export type GenOutcome =
  /** A complete answer. `text` is the whole thing, already streamed via onDelta. */
  | { kind: "answer"; text: string; ms: number }
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
}

export const DEFAULT_GEN_CONFIG: GenConfig = {
  base: "",
  firstTokenTimeoutMs: 12_000,
  totalTimeoutMs: 45_000,
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
    const res = await fetch(`${cfg.base}/synthesize`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, sources }),
      signal: abort.signal,
    });

    if (!res.ok || !res.body) {
      // 503 means no key is configured, which is a supported state rather than
      // a failure. The UI distinguishes it from a generator that is down.
      return { kind: "unavailable", reason: res.status === 503 ? "no generator configured" : `http ${res.status}` };
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let full = "";
    let insufficient = false;
    let serverError: string | null = null;

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
        let ev: { t?: string; done?: boolean; insufficient?: boolean; error?: string };
        try { ev = JSON.parse(payload); } catch { continue; }

        if (ev.error) { serverError = ev.error; continue; }
        if (ev.insufficient) { insufficient = true; continue; }
        if (typeof ev.t === "string" && ev.t) {
          if (!sawToken) { sawToken = true; clearTimeout(firstTokenTimer); }
          full += ev.t;
          handlers.onDelta(ev.t);
        }
      }
    }

    if (serverError) return { kind: "unavailable", reason: serverError };
    if (insufficient) return { kind: "insufficient" };

    const text = full.trim();
    if (!text) return { kind: "unavailable", reason: "empty generation" };
    return { kind: "answer", text, ms: performance.now() - started };
  } catch (err) {
    return { kind: "unavailable", reason: err instanceof Error ? err.message : String(err) };
  } finally {
    clearTimeout(firstTokenTimer);
    clearTimeout(totalTimer);
  }
}
