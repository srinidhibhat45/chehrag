import { defineConfig } from "vite";

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

export default defineConfig({
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
});
