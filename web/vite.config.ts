import { defineConfig, loadEnv, type Plugin, type Connect } from "vite";
import type { ServerResponse } from "node:http";
import { synthesizeStream, resolveProvider, SSE_HEADERS, type Turn } from "../worker/src/synthesize";

/**
 * Cross-origin isolation, which is what lets onnxruntime use SharedArrayBuffer
 * and run the embedding graph on several threads. Without it transformers.js
 * falls back to single-threaded WASM and the dominant cost in the query budget
 * roughly triples, with no error to explain why.
 *
 * Must stay in step with `public/_headers`, the production copy.
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
 * In production `/synthesize` and `/health` belong to the Cloudflare Worker,
 * reached through `VITE_WORKER_BASE`. Without this, developing without a
 * deployed Worker would leave the whole app demonstrable on a laptop except the
 * part that answers.
 *
 * So the dev server mounts the same routes at the same paths: leave
 * `VITE_WORKER_BASE` empty, the client posts to a relative URL, and this answers
 * it. There is no second implementation to drift - the generator is imported
 * from the Worker's own source, so the prompt and the event protocol are the
 * same code in both places.
 *
 * Keys are read from `web/.env.local` with `loadEnv`'s empty prefix, making them
 * server-side variables. None is `VITE_`-prefixed, so none is inlined into the
 * bundle: the browser can call this endpoint but cannot read what authenticates
 * it.
 */
function generatorDevServer(env: Record<string, string>): Plugin {
  // Resolved once at startup by the same function the Worker uses, so "which
  // generator is this" has one answer and one place to change it.
  const provider = resolveProvider(env);

  const health: Connect.NextHandleFunction = (_req, res) => {
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({
      ok: true,
      // Speech still needs the Worker, which holds the Sarvam and ElevenLabs
      // keys and exists partly to attach a header a browser cannot send.
      // Claiming them here would be a claim the UI then repeats.
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

    let body: {
      query?: string;
      sources?: Array<{ title?: string; text?: string }>;
      transcript?: Turn[];
    };
    try { body = JSON.parse(raw); } catch { res.statusCode = 400; res.end(); return; }

    const query = (body.query ?? "").trim();
    const sources = (body.sources ?? [])
      .filter((s) => typeof s?.text === "string" && s.text.trim())
      .map((s) => ({ title: String(s.title ?? "your source"), text: String(s.text) }));
    if (!query || !sources.length) { res.statusCode = 400; res.end(); return; }

    // The tool transcript is replayed the same way the Worker replays it, so
    // the loop behaves identically in development and in production. No
    // validation here that the Worker does not also do - this server is bound
    // to localhost and its input comes from the page it is serving.
    for (const [k, v] of Object.entries(SSE_HEADERS)) res.setHeader(k, v);
    await pipeToNode(
      synthesizeStream(provider, query, sources, Array.isArray(body.transcript) ? body.transcript : []),
      res);
  };

  return {
    name: "chehrag-generator-dev",
    // Registered from `configureServer`/`configurePreviewServer` directly so
    // they land ahead of Vite's own middleware. Otherwise the SPA fallback
    // answers `/health` with index.html and the client concludes, reasonably,
    // that no generator exists.
    configureServer(server) {
      server.middlewares.use("/health", health);
      server.middlewares.use("/synthesize", synthesize);
    },
    // `vite preview` serves the real build, so without this the one command
    // that tests what ships would be the one command with no generator.
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
      // Node coalesces small writes, and an SSE frame sitting in a buffer is a
      // frame the browser has not received.
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
  // Empty prefix loads every variable, not just `VITE_`-prefixed ones. This is
  // config code running in Node, so it can hold the key; nothing here puts it
  // anywhere the client bundle can reach.
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [generatorDevServer(env)],
    build: {
      target: "es2022",
      // Index blobs live in public/ and are copied verbatim. Never inline them:
      // base64 in JS inflates them ~33% and blocks parsing.
      assetsInlineLimit: 0,
      chunkSizeWarningLimit: 2000,
    },
    server: { headers: ISOLATION_HEADERS },
    // `vite preview` serves the real build but does not read `public/_headers`,
    // which is Cloudflare's. Without these the preview loses cross-origin
    // isolation, threaded WASM falls back to single-threaded, and the embed
    // stage measures ~3x slower than production for reasons that have nothing
    // to do with the build being tested.
    preview: { headers: ISOLATION_HEADERS },
    optimizeDeps: { exclude: ["@huggingface/transformers"] },
  };
});
