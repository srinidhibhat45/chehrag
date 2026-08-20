/**
 * Answer generation — the "G" in RAG.
 *
 * WHAT CHANGED AND WHY. This app used to display the retrieved passage itself
 * as the answer. Ask a biodata PDF "what is my name" and it replied
 * `name: srinidhi bhat, age:45, sex:M, faults: crying` — the source line,
 * verbatim. That is a search result, not an answer, and it puts the work of
 * reading the document back onto the person who asked precisely so they would
 * not have to. Generation is not a polish step on top of retrieval here; it is
 * the half of the system that answers the question.
 *
 * WHERE IT SITS RELATIVE TO THE BUDGET. Retrieval is still the 200 ms
 * guarantee, still entirely in this browser, and still what the stopwatch under
 * each answer reports. Generation cannot be inside that budget — a model round
 * trip is hundreds of milliseconds on a good day — and pretending otherwise
 * would make the latency claim a lie. So the two are timed separately and
 * labelled separately: `6.8 ms retrieval · 740 ms answer`. Nothing about the
 * retrieval number changes; what changes is that it is no longer the whole
 * story being told.
 *
 * The response streams, so the first words land in roughly the time it takes
 * the model to produce them rather than after the whole answer exists. That is
 * the entire reason this is not a plain JSON POST.
 *
 * TRUST BOUNDARY. Passage text comes from user documents and from a shipped
 * corpus, and it is put in front of a language model. It is fenced as data in
 * the prompt, the model is told it is data, and the finished answer is checked
 * back against the passages it was supposed to come from (gate 3) before it is
 * allowed to stand. A model that invents a fact fails that check.
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
 * Empty string means same origin, which is what makes one code path work in
 * both places: in development Vite serves `/synthesize` from a middleware that
 * reads the key out of `web/.env.local`, and in production the deployed Worker
 * serves the same route. Neither ever ships the key to this file.
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
 * comes back as `unavailable`, because the caller always has a grounded
 * extractive answer to fall back on and a thrown error would only turn a
 * degraded experience into a broken one.
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

  // Two timers, because they fail for different reasons and deserve different
  // patience. Silence at the start means the request never really began — a
  // cold worker, a bad key, a network black hole — and there is nothing to wait
  // for. Silence after the first token means the model is mid-sentence, and
  // cutting it off there would throw away an answer that is already arriving.
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
      // 503 is the configured-but-keyless case and is not an error condition —
      // it is the honest answer to "can you generate", and the UI says so
      // rather than showing a failure.
      return { kind: "unavailable", reason: res.status === 503 ? "no generator configured" : `http ${res.status}` };
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let full = "";
    let insufficient = false;
    let serverError: string | null = null;

    // Minimal SSE: one JSON object per `data:` line. Deliberately not the
    // upstream event schema — the browser has no business knowing the shape of
    // a model API response, and a narrow protocol is one less thing to keep in
    // step across two server implementations.
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
