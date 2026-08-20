import { defineConfig, loadEnv, type Plugin, type Connect } from "vite";
import type { ServerResponse } from "node:http";
import { synthesizeStream, resolveProvider, SSE_HEADERS } from "../worker/src/synthesize";

/**
 * Cross-origin isolation. This is what lets onnxruntime use SharedArrayBuffer
 * and run the embedding graph on multiple threads; without it transformers.js
 * falls back to single-threaded WASM and the dominant cost in the query budget
 * roughly triples, with no error to explain why.
 *
 * Must stay in step with `public/_headers`, which is the production copy.
 * `credentialless` rather than `require-corp`: the model weights come from the
 * Hugging Face CDN and the fonts from Google, and neither sets CORP.
 */
const ISOLATION_HEADERS = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "credentialless",
};

/**
 * The generator, served locally.
 *
 * In production `/synthesize` and `/health` are the Cloudflare Worker's, reached
 * through `VITE_WORKER_BASE`. In development that would mean the answer half of
 * a RAG system only works for someone who has deployed a Worker — so the whole
 * app would be demonstrable on a laptop *except* the part that answers, which
 * is precisely the part worth looking at.
 *
 * So the dev server mounts the same routes at the same paths. `VITE_WORKER_BASE`
 * stays empty locally, the client posts to a relative URL, and this answers it.
 * There is no second implementation to drift: the generator itself is imported
 * from the Worker's own source, so the prompt and the event protocol are
 * literally the same code in both places.
 *
 * Keys are read from `web/.env.local` with `loadEnv`'s empty prefix, which
 * means they are *server-side* variables. None is `VITE_`-prefixed, so none is
 * inlined into the bundle — the browser can call this endpoint but cannot read
 * what authenticates it.
 */
function generatorDevServer(env: Record<string, string>): Plugin {
  // Resolved once at startup, by the same function the Worker uses, so "which
  // generator am I talking to" has one answer and one place to change it.
  const provider = resolveProvider(env);

  const health: Connect.NextHandleFunction = (_req, res) => {
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({
      ok: true,
      // Speech still needs the Worker: Sarvam and ElevenLabs keys are its, and
      // one of the two exists to attach a header a browser cannot send. Saying
      // they are available here would be a lie the UI would repeat.
      stt: { sarvam: false },
      tts: { elevenlabs: false, sarvam: false },
      llm: !!provider,
      model: provider?.label ?? null,
      dev: true,
    }));
  };

  const synthesize: Connect.NextHandleFunction = async (req, res) => {
    if (req.method !== "POST") { res.statusCode = 405; res.end(); return; }
    if (!provider) {
      res.statusCode = 503;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ error: "llm_not_configured" }));
      return;
    }

    let raw = "";
    for await (const chunk of req) raw += chunk;

    let body: { query?: string; sources?: Array<{ title?: string; text?: string }> };
    try { body = JSON.parse(raw); } catch { res.statusCode = 400; res.end(); return; }

    const query = (body.query ?? "").trim();
    const sources = (body.sources ?? [])
      .filter((s) => typeof s?.text === "string" && s.text.trim())
      .map((s) => ({ title: String(s.title ?? "your source"), text: String(s.text) }));
    if (!query || !sources.length) { res.statusCode = 400; res.end(); return; }

    for (const [k, v] of Object.entries(SSE_HEADERS)) res.setHeader(k, v);
    await pipeToNode(synthesizeStream(provider, query, sources), res);
  };

  return {
    name: "chehrag-generator-dev",
    // Registered from `configureServer`/`configurePreviewServer` directly, so
    // these land ahead of Vite's own middleware — otherwise the SPA fallback
    // answers `/health` with index.html and the client decides, reasonably,
    // that no generator exists.
    configureServer(server) {
      server.middlewares.use("/health", health);
      server.middlewares.use("/synthesize", synthesize);
    },
    // `vite preview` serves the real build. Without this, the one command that
    // tests what actually ships would be the one command with no generator.
    configurePreviewServer(server) {
      server.middlewares.use("/health", health);
      server.middlewares.use("/synthesize", synthesize);
    },
  };
}

/** Web stream -> Node response, without pulling in a stream-interop dependency. */
async function pipeToNode(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(value);
      // Node coalesces small writes; an SSE frame that sits in a buffer is a
      // frame the browser has not received, which is the one thing streaming
      // exists to avoid.
      if (typeof (res as ServerResponse & { flush?: () => void }).flush === "function") {
        (res as ServerResponse & { flush: () => void }).flush();
      }
    }
  } catch {
    res.write(`data: ${JSON.stringify({ error: "generation failed" })}\n\n`);
  } finally {
    res.end();
  }
}

export default defineConfig(({ mode }) => {
  // Empty prefix: load every variable, not just `VITE_`-prefixed ones. This is
  // config code running in Node, so it can hold the key; nothing here puts it
  // anywhere the client bundle can reach.
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [generatorDevServer(env)],
    build: {
      target: "es2022",
      // Index blobs live in public/ and are copied verbatim. Never inline them —
      // base64 in JS would inflate them ~33% and block parsing.
      assetsInlineLimit: 0,
      chunkSizeWarningLimit: 2000,
    },
    server: { headers: ISOLATION_HEADERS },
    // `vite preview` serves the real build, but it does not read `public/_headers`
    // — that file is Cloudflare's. Without these the preview silently loses
    // cross-origin isolation, threaded WASM falls back to single-threaded, and the
    // embed stage measures ~3x slower than production for reasons that have
    // nothing to do with the build being tested.
    preview: { headers: ISOLATION_HEADERS },
    optimizeDeps: { exclude: ["@huggingface/transformers"] },
  };
});
