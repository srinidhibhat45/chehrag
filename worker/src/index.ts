/**
 * Cloudflare Worker — the only server-side component in the system.
 *
 * It exists for exactly two reasons, neither of which is latency:
 *
 *   1. Sarvam authenticates with an `api-subscription-key` HTTP header. Browsers
 *      cannot set custom headers on a WebSocket handshake, and shipping the key
 *      to the client would expose it. So the Worker terminates the browser
 *      socket and opens its own authenticated socket upstream.
 *
 *   2. Optional LLM answer synthesis, which runs *after* the 200ms budget has
 *      already been met by the extractive path.
 *
 * NOTHING here is in the measured retrieval path. If this Worker is down, voice
 * input and LLM polish stop working; typed questions still answer in full, at
 * full speed, because retrieval is entirely client-side.
 *
 * Free tier: 100k requests/day. Deploy with `wrangler deploy`.
 */

import { synthesizeStream, resolveProvider, SSE_HEADERS } from "./synthesize";

export interface Env {
  /* Index signature so `Env` satisfies `EnvLike` in synthesize.ts, which reads
     provider variables by name and must not depend on this interface. */
  [key: string]: string | undefined;
  SARVAM_API_KEY: string;
  ELEVENLABS_API_KEY?: string;
  ELEVENLABS_VOICE_ID?: string;
  /* Answer generation. Any one of these is enough; see synthesize.ts for how
     the provider is resolved and why Groq is the default. */
  GROQ_API_KEY?: string;
  ANTHROPIC_API_KEY?: string;
  /** Point at any other OpenAI-compatible endpoint, including a local one. */
  LLM_BASE_URL?: string;
  LLM_API_KEY?: string;
  LLM_PROVIDER?: string;
  LLM_MODEL?: string;
  ALLOWED_ORIGIN?: string;
}

/**
 * `https://`, not `wss://`, and that is not a typo.
 *
 * A Worker opens an upstream WebSocket by `fetch()`ing an **http(s)** URL with
 * `Upgrade: websocket` and reading `response.webSocket` off the 101. The
 * runtime's fetch has no `wss:` scheme at all — handed one it throws
 * `TypeError: Fetch API cannot load: wss://…` before a single packet moves.
 *
 * The browser-facing URL is still `wss://…/stt/stream`; only this hop upstream
 * is spelled http. Streaming STT was dead in production for exactly this
 * reason, and silently: see the `return await` note in the router for why the
 * exception never reached a log the client could see.
 */
const SARVAM_WS = "https://api.sarvam.ai/speech-to-text-realtime/ws";
const SARVAM_REST = "https://api.sarvam.ai/speech-to-text";
const SARVAM_TTS = "https://api.sarvam.ai/text-to-speech";
const ELEVEN_TTS = "https://api.elevenlabs.io/v1/text-to-speech";

/** Rachel — a stock ElevenLabs voice, so the deploy needs no voice setup. */
const DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM";

/**
 * CORS against a comma-separated allowlist.
 *
 * A list rather than a single value because Cloudflare Pages gives every
 * project two live origins — `chehrag.pages.dev` and a per-deploy
 * `<hash>.chehrag.pages.dev` — and a judge handed either one should not meet a
 * CORS error. Both are pinned explicitly; `*` stays available for local work
 * and is wrong in production, because `/fetch-url` makes outbound requests and
 * an open policy lets any site use this Worker as a proxy on your quota.
 */
function corsHeaders(env: Env, origin: string | null): Record<string, string> {
  const allowed = env.ALLOWED_ORIGIN ?? "*";
  const list = allowed.split(",").map((s) => s.trim()).filter(Boolean);
  const ok = allowed === "*" || (!!origin && list.includes(origin));
  return {
    "Access-Control-Allow-Origin": ok ? (origin ?? "*") : (list[0] ?? "*"),
    // The allowed origin now varies by request, so caches must key on it.
    Vary: "Origin",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const origin = req.headers.get("Origin");

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(env, origin) });
    }

    // `return await`, not `return`. A bare `return somePromise()` inside a
    // try block hands the promise to the caller *before* it settles, so a
    // rejection never reaches this catch: the runtime sees an uncaught
    // exception and answers "error code: 1101" with no CORS headers and no
    // log line the client can act on. That is exactly how a one-word bug in
    // handleSttStream stayed invisible in production.
    try {
      if (url.pathname === "/stt/stream") return await handleSttStream(req, env);
      if (url.pathname === "/stt/batch") return await handleSttBatch(req, env, origin);
      if (url.pathname === "/tts") return await handleTts(req, env, origin);
      if (url.pathname === "/synthesize") return await handleSynthesize(req, env, origin);
      if (url.pathname === "/fetch-url") return await handleFetchUrl(req, env, origin);
      if (url.pathname === "/health") {
        return Response.json(
          {
            ok: true,
            stt: { sarvam: !!env.SARVAM_API_KEY },
            tts: { elevenlabs: !!env.ELEVENLABS_API_KEY, sarvam: !!env.SARVAM_API_KEY },
            llm: !!resolveProvider(env),
            model: resolveProvider(env)?.label ?? null,
          },
          { headers: corsHeaders(env, origin) },
        );
      }
      return new Response("not found", { status: 404, headers: corsHeaders(env, origin) });
    } catch (err) {
      // Never leak internals to the client; the browser degrades on any 5xx.
      console.error("worker error", err);
      return Response.json(
        { error: "upstream_error" },
        { status: 502, headers: corsHeaders(env, origin) },
      );
    }
  },
};

/**
 * WebSocket relay: browser <-> Worker <-> Sarvam.
 *
 * Messages pass through untouched in both directions. The Worker's only job is
 * attaching the auth header the browser cannot send.
 */
async function handleSttStream(req: Request, env: Env): Promise<Response> {
  if (req.headers.get("Upgrade") !== "websocket") {
    return new Response("expected websocket upgrade", { status: 426 });
  }
  if (!env.SARVAM_API_KEY) {
    return new Response("STT not configured", { status: 503 });
  }

  const params = new URL(req.url).searchParams;
  const upstreamUrl = new URL(SARVAM_WS);
  // Forward only known-safe params; never reflect arbitrary client input upstream.
  for (const k of ["language_code", "model", "input_audio_codec", "sample_rate"]) {
    const v = params.get(k);
    if (v) upstreamUrl.searchParams.set(k, v);
  }
  if (!upstreamUrl.searchParams.has("model")) {
    upstreamUrl.searchParams.set("model", "saaras:v3-realtime");
  }

  const upstream = await fetch(upstreamUrl.toString(), {
    headers: {
      Upgrade: "websocket",
      "api-subscription-key": env.SARVAM_API_KEY,
    },
  });
  const server = upstream.webSocket;
  if (!server) return new Response("upstream refused websocket", { status: 502 });
  server.accept();

  const pair = new WebSocketPair();
  const [client, browser] = [pair[0], pair[1]];
  browser.accept();

  browser.addEventListener("message", (e: MessageEvent) => {
    try { server.send(e.data as string | ArrayBuffer); } catch { /* closing */ }
  });
  server.addEventListener("message", (e: MessageEvent) => {
    try { browser.send(e.data as string | ArrayBuffer); } catch { /* closing */ }
  });

  const closeBoth = (code?: number, reason?: string) => {
    // 1005/1006 are "no status" codes and are illegal to send explicitly.
    const safe = code && code >= 1000 && code !== 1005 && code !== 1006 ? code : 1000;
    try { server.close(safe, reason); } catch { /* already closed */ }
    try { browser.close(safe, reason); } catch { /* already closed */ }
  };
  browser.addEventListener("close", (e: CloseEvent) => closeBoth(e.code, e.reason));
  server.addEventListener("close", (e: CloseEvent) => closeBoth(e.code, e.reason));
  browser.addEventListener("error", () => closeBoth(1011, "browser error"));
  server.addEventListener("error", () => closeBoth(1011, "upstream error"));

  return new Response(null, { status: 101, webSocket: client });
}

/**
 * Batch transcription fallback.
 *
 * The streaming path is preferred (partials drive speculative retrieval), but
 * corporate proxies and some mobile networks break long-lived WebSockets. This
 * keeps voice working when they do.
 */
async function handleSttBatch(req: Request, env: Env, origin: string | null): Promise<Response> {
  if (req.method !== "POST") {
    return new Response("method not allowed", { status: 405, headers: corsHeaders(env, origin) });
  }
  if (!env.SARVAM_API_KEY) {
    return Response.json({ error: "stt_not_configured" },
      { status: 503, headers: corsHeaders(env, origin) });
  }

  const inBody = await req.formData();
  const file = inBody.get("file");
  if (!(file instanceof File)) {
    return Response.json({ error: "missing_file" },
      { status: 400, headers: corsHeaders(env, origin) });
  }
  // Cap upload size: a voice question is seconds, not minutes.
  if (file.size > 8 * 1024 * 1024) {
    return Response.json({ error: "file_too_large" },
      { status: 413, headers: corsHeaders(env, origin) });
  }

  const form = new FormData();
  form.append("file", file, "audio.webm");
  form.append("model", (inBody.get("model") as string) || "saaras:v3");
  const lang = inBody.get("language_code") as string | null;
  if (lang) form.append("language_code", lang);

  const res = await fetch(SARVAM_REST, {
    method: "POST",
    headers: { "api-subscription-key": env.SARVAM_API_KEY },
    body: form,
  });
  if (!res.ok) {
    return Response.json({ error: "stt_failed", status: res.status },
      { status: 502, headers: corsHeaders(env, origin) });
  }
  const data = await res.json<{ transcript?: string; language_code?: string }>();
  return Response.json(
    { transcript: data.transcript ?? "", language: data.language_code ?? null },
    { headers: corsHeaders(env, origin) },
  );
}

/**
 * Text to speech.
 *
 * Two providers behind one endpoint, chosen by the client from the language of
 * the answer — ElevenLabs where it has the language, Sarvam for the eight Indic
 * languages ElevenLabs does not cover. See `web/src/tts/speak.ts` for why that
 * split exists rather than a single provider.
 *
 * The response body is the audio itself, streamed straight through. Buffering
 * it here to inspect or re-encode would add the whole generation time to the
 * first byte, and this is the one place in the system where a user is waiting
 * on a network round trip in real time.
 */
async function handleTts(req: Request, env: Env, origin: string | null): Promise<Response> {
  const headers = corsHeaders(env, origin);
  if (req.method !== "POST") {
    return new Response("method not allowed", { status: 405, headers });
  }

  const body = await req.json<{ text?: string; provider?: string; language?: string }>()
    .catch(() => ({} as { text?: string; provider?: string; language?: string }));

  // Capped rather than rejected: a long answer should still be spoken, just
  // truncated. Also bounds spend per request on both providers.
  const text = (body.text ?? "").trim().slice(0, 900);
  if (!text) return Response.json({ error: "empty_text" }, { status: 400, headers });

  const provider = body.provider === "sarvam" ? "sarvam" : "elevenlabs";
  const audioHeaders = {
    ...headers,
    "content-type": "audio/mpeg",
    // Same text, same voice, same audio. Worth caching at the edge — repeat
    // questions in a demo are the common case.
    "cache-control": "public, max-age=3600",
  };

  if (provider === "sarvam") {
    if (!env.SARVAM_API_KEY) {
      return Response.json({ error: "sarvam_tts_not_configured" }, { status: 503, headers });
    }
    const res = await fetch(SARVAM_TTS, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "api-subscription-key": env.SARVAM_API_KEY,
      },
      body: JSON.stringify({
        text,
        target_language_code: body.language ?? "hi-IN",
        model: "bulbul:v2",
        speaker: "anushka",
        enable_preprocessing: true,
      }),
    });
    if (!res.ok) {
      return Response.json({ error: "sarvam_tts_failed", status: res.status },
        { status: 502, headers });
    }
    // Sarvam returns base64 WAV in JSON rather than an audio body, so this one
    // genuinely has to be buffered — there is no stream to pass through.
    const data = await res.json<{ audios?: string[] }>();
    const b64 = data.audios?.[0];
    if (!b64) return Response.json({ error: "sarvam_tts_empty" }, { status: 502, headers });
    const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new Response(bin, { headers: { ...audioHeaders, "content-type": "audio/wav" } });
  }

  if (!env.ELEVENLABS_API_KEY) {
    return Response.json({ error: "elevenlabs_not_configured" }, { status: 503, headers });
  }
  const voice = env.ELEVENLABS_VOICE_ID || DEFAULT_VOICE_ID;
  const res = await fetch(`${ELEVEN_TTS}/${voice}/stream?output_format=mp3_22050_32`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "xi-api-key": env.ELEVENLABS_API_KEY,
      accept: "audio/mpeg",
    },
    body: JSON.stringify({
      text,
      // flash v2.5 — ~75 ms model latency and 32 languages. The higher-quality
      // multilingual_v2 costs roughly half a second more per request, which for
      // a spoken reply is the difference between a conversation and a wait.
      model_id: "eleven_flash_v2_5",
      language_code: body.language || undefined,
      voice_settings: { stability: 0.4, similarity_boost: 0.75, speed: 1.0 },
    }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    return Response.json({ error: "elevenlabs_failed", status: res.status, detail: detail.slice(0, 300) },
      { status: 502, headers });
  }
  return new Response(res.body, { headers: audioHeaders });
}

/**
 * Answer synthesis — the generation half of the RAG loop.
 *
 * This used to be described as "LLM polish", an optional rewrite layered over
 * an answer the browser had already produced. It is not optional any more, and
 * calling it polish was what let the product ship echoing its own sources back
 * at people. The browser retrieves; this generates; neither is the whole
 * system on its own.
 *
 * It is still not on the 200 ms clock, and that has not changed: retrieval is
 * what carries the latency guarantee, it is measured separately, and it is
 * reported separately in the interface. What changed is that the interface no
 * longer presents a retrieved passage as though it were an answer.
 *
 * The response streams. See `synthesize.ts` for the prompt, the event protocol
 * and why both live in a host-agnostic module.
 */
async function handleSynthesize(req: Request, env: Env, origin: string | null): Promise<Response> {
  const cors = corsHeaders(env, origin);

  if (req.method !== "POST") {
    return new Response("method not allowed", { status: 405, headers: cors });
  }
  const provider = resolveProvider(env);
  if (!provider) {
    // 503 rather than 500: nothing is broken, the capability was never
    // configured. The client treats it as "no generator" and keeps its
    // extractive answer rather than showing a failure.
    return Response.json({ error: "llm_not_configured" }, { status: 503, headers: cors });
  }

  let body: { query?: string; sources?: Array<{ title?: string; text?: string }> };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "bad_request" }, { status: 400, headers: cors });
  }

  const query = (body.query ?? "").trim();
  const sources = (body.sources ?? [])
    .filter((s) => typeof s?.text === "string" && s.text.trim())
    .map((s) => ({ title: String(s.title ?? "your source"), text: String(s.text) }));

  if (!query || !sources.length) {
    return Response.json({ error: "bad_request" }, { status: 400, headers: cors });
  }

  const stream = synthesizeStream(provider, query, sources);

  return new Response(stream, { headers: { ...cors, ...SSE_HEADERS } });
}

/**
 * Fetch a URL the user asked to add as a source, and return it as plain text.
 *
 * This exists because the browser cannot: reading the body of an arbitrary
 * cross-origin page is exactly what CORS forbids. It is not on any latency
 * path — a source is fetched once, when it is added.
 *
 * It is, however, the one endpoint that makes an outbound request to an address
 * the user supplies, which makes it the one place server-side request forgery
 * could get in. Guards, in order:
 *
 *   - http/https only; no file:, data:, gopher: or blob:
 *   - literal private, loopback, link-local and metadata addresses refused
 *   - redirects followed manually, with every hop re-checked, because a public
 *     hostname is free to redirect to 169.254.169.254
 *   - response capped, timed out, and content-type checked before reading
 *
 * The Worker holds two API keys. It must never become a way to read anything
 * that is only reachable from where it runs.
 */
const MAX_FETCH_BYTES = 6 * 1024 * 1024;
const FETCH_TIMEOUT_MS = 12_000;
const MAX_REDIRECTS = 4;

/** Addresses that are never a legitimate public web page. */
function isBlockedHost(hostname: string): boolean {
  const h = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (h === "localhost" || h.endsWith(".localhost") || h.endsWith(".internal") ||
      h.endsWith(".local") || h === "metadata.google.internal") {
    return true;
  }
  // IPv4 literals.
  const v4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(h);
  if (v4) {
    const [a, b] = [Number(v4[1]), Number(v4[2])];
    if (a === 0 || a === 10 || a === 127) return true;              // this/private/loopback
    if (a === 169 && b === 254) return true;                         // link-local + cloud metadata
    if (a === 172 && b >= 16 && b <= 31) return true;                // private
    if (a === 192 && b === 168) return true;                         // private
    if (a === 100 && b >= 64 && b <= 127) return true;               // carrier NAT
    if (a >= 224) return true;                                       // multicast + reserved
    return false;
  }
  // IPv6 literals: loopback, unique-local, link-local, and v4-mapped forms.
  if (h === "::1" || h === "::" ) return true;
  if (/^f[cd][0-9a-f]{2}:/.test(h) || /^fe[89ab][0-9a-f]:/.test(h)) return true;
  if (/^::ffff:/.test(h)) return true;
  return false;
}

const TEXTUAL = /^(text\/|application\/(xhtml\+xml|xml|json|.*\+json))/i;

async function handleFetchUrl(req: Request, env: Env, origin: string | null): Promise<Response> {
  const headers = corsHeaders(env, origin);
  if (req.method !== "POST") {
    return new Response("method not allowed", { status: 405, headers });
  }

  const body = await req.json<{ url?: string }>().catch(() => ({} as { url?: string }));
  let target: URL;
  try {
    target = new URL(body.url ?? "");
  } catch {
    return Response.json({ error: "bad_url" }, { status: 400, headers });
  }

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    let res: Response | null = null;
    for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
      if (target.protocol !== "http:" && target.protocol !== "https:") {
        return Response.json({ error: "unsupported_scheme" }, { status: 400, headers });
      }
      // Re-checked on EVERY hop. Checking only the first URL is the classic
      // SSRF hole: a public host can redirect straight to link-local.
      if (isBlockedHost(target.hostname)) {
        return Response.json({ error: "blocked_host" }, { status: 400, headers });
      }

      res = await fetch(target.toString(), {
        redirect: "manual",
        signal: ctrl.signal,
        headers: {
          // Identify honestly and ask for text. Some sites serve a different
          // (JS-only) page to unknown agents; that is their call to make.
          "User-Agent": "Chehrag/1.0 (+source ingestion; https://github.com/)",
          "Accept": "text/html,text/plain,application/xhtml+xml,text/markdown;q=0.9,*/*;q=0.1",
          "Accept-Language": "en,hi;q=0.8",
        },
      });

      if (res.status >= 300 && res.status < 400) {
        const loc = res.headers.get("location");
        if (!loc) break;
        target = new URL(loc, target);
        continue;
      }
      break;
    }

    if (!res) return Response.json({ error: "too_many_redirects" }, { status: 400, headers });
    if (!res.ok) {
      return Response.json({ error: "upstream_status", status: res.status },
        { status: 502, headers });
    }

    const ctype = res.headers.get("content-type") ?? "";
    if (!TEXTUAL.test(ctype)) {
      return Response.json({ error: "unsupported_type", contentType: ctype },
        { status: 415, headers });
    }
    const declared = Number(res.headers.get("content-length") ?? 0);
    if (declared > MAX_FETCH_BYTES) {
      return Response.json({ error: "too_large" }, { status: 413, headers });
    }

    const raw = await readCapped(res, MAX_FETCH_BYTES);
    if (raw === null) return Response.json({ error: "too_large" }, { status: 413, headers });

    const html = /html|xml/i.test(ctype);
    return Response.json(
      {
        title: html ? titleOf(raw) : target.pathname.split("/").filter(Boolean).pop() || target.hostname,
        text: html ? stripHtml(raw) : raw,
        contentType: ctype,
        finalUrl: target.toString(),
      },
      { headers },
    );
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    return Response.json({ error: aborted ? "timeout" : "fetch_failed" },
      { status: 502, headers });
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Read a body with a hard byte ceiling.
 *
 * `content-length` is a claim, not a guarantee — it can be absent, wrong, or
 * deliberately understated. Streaming and counting is the only version that
 * actually bounds memory.
 */
async function readCapped(res: Response, max: number): Promise<string | null> {
  const reader = res.body?.getReader();
  if (!reader) return "";
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > max) { await reader.cancel(); return null; }
    chunks.push(value);
  }
  const buf = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) { buf.set(c, at); at += c.byteLength; }
  // Non-fatal: a page with one bad byte should still be readable, and
  // U+FFFD in the output is what `looksBinary` on the client keys off.
  return new TextDecoder("utf-8", { fatal: false, ignoreBOM: false }).decode(buf);
}

function titleOf(html: string): string {
  const og = /<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']+)["']/i.exec(html);
  if (og) return decodeEntities(og[1]).trim().slice(0, 120);
  const t = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html);
  return t ? decodeEntities(t[1]).trim().slice(0, 120) : "";
}

/**
 * HTML to text, on the server.
 *
 * Workers have no DOMParser, so this is regex-based — acceptable because the
 * output is only ever embedded and displayed as text, never re-rendered as
 * markup. Order matters: script and style bodies must go before tags are
 * stripped, or their contents survive as "text".
 */
function stripHtml(html: string): string {
  return decodeEntities(
    html
      .replace(/<!--[\s\S]*?-->/g, " ")
      .replace(/<(script|style|noscript|svg|template)\b[\s\S]*?<\/\1\s*>/gi, " ")
      .replace(/<\/(p|div|li|tr|h[1-6]|section|article|blockquote|pre)\s*>/gi, "\n")
      .replace(/<br\s*\/?>/gi, "\n")
      .replace(/<[^>]+>/g, " "),
  )
    .replace(/[ \t ]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function decodeEntities(s: string): string {
  return s
    .replace(/&nbsp;/gi, " ").replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<").replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"').replace(/&#0?39;|&apos;/gi, "'")
    .replace(/&#(\d+);/g, (_, d) => safeChar(Number(d)))
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => safeChar(parseInt(h, 16)));
}

function safeChar(code: number): string {
  return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : "";
}
