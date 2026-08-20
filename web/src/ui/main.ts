/**
 * App wiring.
 *
 * Two timing paths, kept strictly separate, because conflating them is how
 * latency claims become dishonest:
 *
 *   FAST PATH (measured, budget 200 ms) — embed, retrieve, fuse, rescore,
 *   guard, extract. Entirely in this browser, across both the shipped corpus
 *   and the user's own sources. This is what the chip under each answer
 *   reports, and it is the only thing that chip reports.
 *
 *   SLOW PATH (unmeasured, best effort) — Sarvam speech-to-text before the
 *   question exists, ElevenLabs or Sarvam speech after the answer already
 *   exists, and the LLM rewrite after that. None of it may touch the measured
 *   span, and none of it is allowed to be a precondition for answering: with
 *   the Worker unreachable, typed questions still answer at full speed.
 *
 * The status dots in the top bar are load-bearing for honesty. They report what
 * the Worker says it actually has keys for, so the interface never claims a
 * Sarvam or ElevenLabs integration it has not confirmed.
 */

import { loadIndex } from "../retrieval/loader";
import { createEncoder, type BatchEncoder } from "../retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG, USER_BASE, type RagAnswer, type RagConfig } from "../harness/rag";
import { percentiles, CircuitBreaker } from "../harness/pipeline";
import { SourceStore } from "../sources/store";
import { SourcesPanel } from "./sources-panel";
import { Chat, type BotHandle } from "./chat";
import { Orb } from "./orb";
import { pickStt, sttAvailability, type SttEngine, type SttKind } from "../stt";
import { Speaker, scriptLanguage, type TtsProvider } from "../tts/speak";

const WORKER_BASE = import.meta.env.VITE_WORKER_BASE ?? "";

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

const app = $("app");
const orb = new Orb($("orb"), $("orb-caption"), $("orb-sub"));
const llmBreaker = new CircuitBreaker(3, 20_000);
const speaker = new Speaker({
  workerBase: WORKER_BASE,
  onNotice: (m) => { $("voice-note").textContent = m; },
});

let engine: RagEngine;
let encoder: BatchEncoder;
let store: SourceStore;
let chat: Chat;

/** Set while a query is in flight, so voice partials don't stack up. */
let asking = false;
/** Session latency, for the studio readout. */
const sessionTimes: number[] = [];

// ---------------------------------------------------------------------------
// theme
// ---------------------------------------------------------------------------

const savedTheme = localStorage.getItem("chehrag-theme");
if (savedTheme === "light" || savedTheme === "dark") {
  document.documentElement.dataset.theme = savedTheme;
}
$("theme-btn").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme
    ?? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = cur === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("chehrag-theme", next);
  orb.syncTheme();
});

// ---------------------------------------------------------------------------
// what is searchable
// ---------------------------------------------------------------------------

const CORPUS_KEY = "chehrag-corpus";

/** Opt-in, and remembered. Absent means off — a first visit starts empty. */
function corpusPref(): boolean { return localStorage.getItem(CORPUS_KEY) === "1"; }

function setCorpus(on: boolean): void {
  localStorage.setItem(CORPUS_KEY, on ? "1" : "0");
  engine?.setCorpusEnabled(on);
  syncSources();
}

/**
 * Reflect what is actually searchable, in the rail and in the welcome.
 *
 * Tolerates a missing welcome: `Chat` removes that block as soon as the first
 * question is asked, and this runs again every time a source changes.
 */
function syncSources(): void {
  const on = engine?.corpusOn ?? corpusPref();

  $("corpus-card").dataset.enabled = on ? "1" : "0";
  const toggle = $<HTMLButtonElement>("corpus-toggle");
  toggle.setAttribute("aria-pressed", String(on));
  toggle.setAttribute("aria-label",
    on ? "Stop searching the Hindi sample set" : "Search the Hindi sample set");

  // The sample questions are questions about the Hindi corpus. Offering them
  // while it is switched off would be offering a button that cannot work.
  const suggest = document.getElementById("suggest");
  if (suggest) suggest.hidden = !on;
  const hint = document.getElementById("welcome-hint");
  if (hint) hint.hidden = on || !!store?.hasEnabled;
}

// ---------------------------------------------------------------------------
// languages
// ---------------------------------------------------------------------------

/**
 * What the user can say they are speaking.
 *
 * `auto` is first and is the default because Sarvam genuinely identifies the
 * language and returns it with the transcript. On the browser-speech fallback
 * there is no such thing, so the note under the picker says so rather than
 * letting `auto` quietly mean "Hindi".
 */
const LANGS: Array<{ tag: string; label: string }> = [
  { tag: "auto",  label: "Detect automatically" },
  { tag: "hi-IN", label: "हिन्दी — Hindi" },
  { tag: "bn-IN", label: "বাংলা — Bengali" },
  { tag: "ta-IN", label: "தமிழ் — Tamil" },
  { tag: "te-IN", label: "తెలుగు — Telugu" },
  { tag: "mr-IN", label: "मराठी — Marathi" },
  { tag: "gu-IN", label: "ગુજરાતી — Gujarati" },
  { tag: "kn-IN", label: "ಕನ್ನಡ — Kannada" },
  { tag: "ml-IN", label: "മലയാളം — Malayalam" },
  { tag: "pa-IN", label: "ਪੰਜਾਬੀ — Punjabi" },
  { tag: "od-IN", label: "ଓଡ଼ିଆ — Odia" },
  { tag: "en-IN", label: "English" },
];

function wireLangs(): void {
  const sel = $<HTMLSelectElement>("lang-sel");
  for (const l of LANGS) {
    const o = document.createElement("option");
    o.value = l.tag;
    o.textContent = l.label;
    sel.append(o);
  }
  sel.value = localStorage.getItem("chehrag-lang") ?? "auto";
  sel.addEventListener("change", () => {
    localStorage.setItem("chehrag-lang", sel.value);
    updateVoiceNote();
  });
}

function askLang(): string { return $<HTMLSelectElement>("lang-sel").value; }

// ---------------------------------------------------------------------------
// service status
// ---------------------------------------------------------------------------

interface Wiring {
  sttSarvam: boolean;
  sttBrowser: boolean;
  ttsEleven: boolean;
  ttsSarvam: boolean;
  llm: boolean;
}
let wiring: Wiring = {
  sttSarvam: false, sttBrowser: false, ttsEleven: false, ttsSarvam: false, llm: false,
};

/**
 * Ask the Worker what it actually has keys for.
 *
 * Deliberately not inferred from `VITE_WORKER_BASE` being set. A Worker can be
 * deployed with a Sarvam key and no ElevenLabs key, or either one can be
 * revoked; the only reliable source for "is the voice requirement live" is the
 * Worker telling us.
 */
async function probeWiring(): Promise<void> {
  const local = sttAvailability(WORKER_BASE);
  wiring.sttBrowser = local.browser;

  if (WORKER_BASE) {
    try {
      const r = await fetch(`${WORKER_BASE}/health`);
      const h = await r.json() as {
        stt?: { sarvam?: boolean };
        tts?: { elevenlabs?: boolean; sarvam?: boolean };
        llm?: boolean;
      };
      wiring.sttSarvam = !!h.stt?.sarvam;
      wiring.ttsEleven = !!h.tts?.elevenlabs;
      wiring.ttsSarvam = !!h.tts?.sarvam;
      wiring.llm = !!h.llm;
    } catch { /* Worker unreachable — everything stays off */ }
  }
  await speaker.probe();
  renderWiring();
}

/**
 * Reflect what the Worker actually confirmed.
 *
 * There used to be a row of lit/unlit dots here naming each provider. That is
 * an instrument panel, not a product: it asks the person using this to care
 * which vendor transcribes them. The honesty it existed for is kept — the note
 * under the voice pickers says in words which voice will speak and whether
 * speech input is the real path or the browser's fallback — but it is stated
 * where it changes what someone would do, and nowhere else.
 */
function renderWiring(): void {
  // Never before the index is up: an enabled mic on a booting app records a
  // question there is nothing to answer with.
  $<HTMLButtonElement>("mic-btn").disabled = !micReady() || app.dataset.stage === "boot";
  updateVoiceNote();
}

function updateVoiceNote(): void {
  const sel = $<HTMLSelectElement>("voice-sel");
  const parts: string[] = [];
  if (sel.value === "off") {
    parts.push("Answers are not spoken.");
  } else {
    const lang = askLang() === "auto" ? "hi-IN" : askLang();
    const route = speaker.routeFor(lang);
    const NAME: Record<TtsProvider, string> = {
      elevenlabs: "ElevenLabs",
      sarvam: "Sarvam",
      browser: "your browser's built-in voice",
      none: "nothing — no voice is available here",
    };
    parts.push(`Answers are read aloud by ${NAME[route]}.`);
  }
  if (!wiring.sttSarvam && wiring.sttBrowser) {
    parts.push("Questions are heard by your browser's own recogniser.");
  }
  if (!wiring.sttSarvam && askLang() === "auto") {
    parts.push("It can't detect the language on its own, so it will assume Hindi — pick one above to be sure.");
  }
  $("voice-note").textContent = parts.join(" ");
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  const t0 = performance.now();
  wireLangs();
  void probeWiring();
  checkIsolation();

  // The index and the model download in parallel. They are independent, and
  // serialising them would double the wait on a first visit for no reason.
  const indexPromise = loadIndex("/index", (loaded, total, name) => {
    // Cap the ring at 90% until the encoder is also up — a full ring next to a
    // disabled input is a lie the user notices.
    orb.setCharge(Math.min(0.9, loaded / total));
    $("orb-sub").textContent =
      `${(loaded / 1e6).toFixed(0)} MB of ~${(total / 1e6).toFixed(0)} MB · ${name}`;
  });
  const encoderPromise = createEncoder();

  orb.set("dormant", "Lighting the lamp…");
  const [index, enc] = await Promise.all([indexPromise, encoderPromise]);
  encoder = enc;
  orb.setCharge(0.95);
  $("orb-sub").textContent = "warming up…";

  // Thresholds are fitted offline by bench/calibrate.ts. If that has not been
  // run, fall back to the defaults rather than blocking the app.
  let cfg: RagConfig = { ...DEFAULT_CONFIG };
  try {
    const t = await (await fetch("/thresholds.json")).json();
    // Spread over the defaults rather than replacing them: calibrate.ts fits
    // three of these fields and does not know about the rest, so a literal
    // object here would silently reset every threshold it omits to undefined.
    cfg = { ...cfg, thresholds: {
      ...cfg.thresholds,
      minTopScore: t.minTopScore, minAgreement: t.minAgreement,
      minLexicalOverlap: t.minLexicalOverlap ?? 0,
    } };
  } catch { /* defaults */ }

  engine = new RagEngine(index, encoder, cfg);

  store = new SourceStore({
    encoder,
    pcaMean: index.pcaMean,
    pcaComp: index.pcaComp,
    dim: index.manifest.dim,
    codeWords: index.manifest.codeWords,
    model: index.manifest.model,
    workerBase: WORKER_BASE,
  });
  engine.attachSources(store);
  // Applied before warmup so the console handle and the rail agree from the
  // first frame. Warmup forces the corpus on internally regardless — it is
  // warming code paths, not answering questions.
  engine.setCorpusEnabled(corpusPref());
  new SourcesPanel(store, { onChange: () => { engine.invalidate(); syncSources(); } });
  store.subscribe(() => syncSources());
  // Sources added in a previous session replay from IndexedDB. Not awaited on
  // the critical path — the corpus is already searchable without them.
  void store.restore().catch(() => { /* corrupt store: start empty */ });

  chat = new Chat($("thread"), { onSpeak: (text, btn) => void speakAnswer(text, btn) });

  await engine.warmup();
  orb.setCharge(1);

  const chunks = Object.values(index.manifest.strategies).reduce((a, s) => a + s.count, 0);
  $("corpus-meta").textContent =
    `${index.manifest.numPassages.toLocaleString()} passages · ${chunks.toLocaleString()} chunks · ` +
    `${Object.keys(index.manifest.strategies).length} strategies`;
  syncSources();
  $("src-stat").textContent = `ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`;

  app.dataset.stage = "idle";
  $<HTMLInputElement>("q").disabled = false;
  $<HTMLButtonElement>("go").disabled = false;
  $<HTMLButtonElement>("mic-btn").disabled = !(wiring.sttSarvam || wiring.sttBrowser);

  orb.set("idle", "Ready", micReady()
    ? "tap the lamp, or the mic, and ask"
    : "no microphone available — typing works");
  $("composer-note").textContent = orb.offMainThread
    ? "Searched in this browser. The lamp draws on its own thread, so it costs a question nothing."
    : "Searched in this browser. The lamp holds still while a question is timed.";
  $<HTMLInputElement>("q").focus();
}

function micReady(): boolean { return wiring.sttSarvam || wiring.sttBrowser; }

/**
 * The one hosting mistake that costs the budget.
 *
 * Threaded WASM needs `SharedArrayBuffer`, which needs cross-origin isolation,
 * which needs COOP + COEP on the response that served this page. Get the
 * headers wrong and onnxruntime silently drops to a single thread: the embed
 * stage — the dominant cost in the query — roughly triples, nothing errors, and
 * the only symptom is that the numbers are worse than they were on localhost.
 *
 * `public/_headers` is the production copy of `vite.config.ts`'s dev headers.
 * If the host does not honour it, this is the line that says so.
 */
function checkIsolation(): void {
  if (crossOriginIsolated) return;
  console.warn(
    "[chehrag] This page is NOT cross-origin isolated, so WASM is running " +
    "single-threaded and every query will be roughly 3x slower than it needs " +
    "to be. Serve it with:\n" +
    "  Cross-Origin-Opener-Policy: same-origin\n" +
    "  Cross-Origin-Embedder-Policy: credentialless\n" +
    "(see web/public/_headers — Cloudflare Pages reads it automatically).",
  );
}

// ---------------------------------------------------------------------------
// ask
// ---------------------------------------------------------------------------

async function ask(text: string, opts: { voice?: boolean } = {}): Promise<void> {
  const q = text.trim();
  if (!q || !engine || asking) return;
  asking = true;
  $<HTMLInputElement>("q").value = "";
  $("transcript").hidden = true;

  chat.addUser(q, opts.voice);
  const handle = chat.addPending();
  orb.set("thinking", "Looking…", "");

  try {
    // Bulk embedding is paused for the duration. Ingestion already yields
    // between small batches, but pausing removes even that interference — the
    // budget is a promise, and a promise that only holds when nothing else is
    // happening is not one.
    encoder.pauseBulk();
    // On the inline-WebGL fallback the fire shares this thread, so it stops for
    // the measured span. On the worker path this is a no-op.
    orb.freeze();

    const res = await engine.ask(q, { skipCache: true });

    handle.resolve(res, {
      userChunks: store?.activeChunks ?? 0,
      truncated: store?.truncated ?? false,
    });
    // Only questions that were actually searched belong in the session figures.
    // A refusal for want of a source never reached the encoder, and folding its
    // 0.1 ms into the readout would flatter every number in it.
    if (res.refusal !== "NO_SOURCES") {
      sessionTimes.push(res.totalMs);
      renderSessionStats();
    }

    const answered = res.status === "answered";
    orb.set(answered ? "answered" : "refused", answered ? "Answered" : "Declined", "");

    // Refusing for want of a source and leaving the user to find the Add
    // button themselves is a dead end. Put the panel in front of them.
    if (res.refusal === "NO_SOURCES") openAddPanel();

    if (answered && $<HTMLSelectElement>("voice-sel").value !== "off" && opts.voice) {
      // Spoken questions get spoken answers without being asked twice; typed
      // ones get a button, because someone typing in an office did not ask for
      // audio.
      void speakAnswer(res.answer, null, handle);
    }
    if (answered && wiring.llm) void synthesize(q, res, handle);
  } catch (err) {
    chat.addNotice(`Something went wrong: ${err instanceof Error ? err.message : String(err)}`);
    orb.set("refused", "Failed", "");
  } finally {
    orb.thaw();
    encoder.resumeBulk();
    asking = false;
  }
}

function renderSessionStats(): void {
  if (!sessionTimes.length) return;
  const p = percentiles(sessionTimes);
  $("stat-p50").textContent = p.p50.toFixed(1);
  $("stat-p100").textContent = p.p100.toFixed(1);
  $("stat-n").textContent = String(sessionTimes.length);
}

/** Off the fast path. Verified against retrieved context before display. */
async function synthesize(query: string, base: RagAnswer, handle: BotHandle): Promise<void> {
  try {
    const data = await llmBreaker.call(async () => {
      const res = await fetch(`${WORKER_BASE}/synthesize`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query, contexts: base.citations.map((c) => c.text) }),
      });
      if (!res.ok) throw new Error(`synthesize ${res.status}`);
      return res.json() as Promise<{ answer: string | null }>;
    });
    if (!data.answer) return;
    // Gate 3: a fluent model can quietly add a fact that was never in the
    // sources. If it drifted, we keep the extractive answer and say nothing.
    if (!engine.verifySynthesis(data.answer, base.citations).pass) return;
    handle.polish(data.answer);
  } catch { /* worker down or circuit open — the extractive answer already stands */ }
}

// ---------------------------------------------------------------------------
// speaking
// ---------------------------------------------------------------------------

/**
 * Speak an answer.
 *
 * The language comes from the STT result when there is one — a model that heard
 * the speaker knows more than any heuristic — and otherwise from the script of
 * the answer text. See `scriptLanguage` for why that is script detection and
 * does not pretend to be more.
 */
let lastHeardLanguage: string | null = null;

async function speakAnswer(
  text: string,
  btn: HTMLElement | null,
  handle?: BotHandle,
): Promise<void> {
  const lang = lastHeardLanguage ?? scriptLanguage(text);
  if (btn) btn.textContent = "Speaking…";
  try {
    const r = await speaker.speak(text, lang);
    const NAME: Record<TtsProvider, string> = {
      elevenlabs: "ElevenLabs", sarvam: "Sarvam", browser: "browser voice", none: "no voice",
    };
    handle?.setVoice(`spoken by ${NAME[r.provider]} · ${r.firstAudioMs.toFixed(0)} ms to audio`);
    if (r.el) {
      orb.set("speaking", "Speaking", "");
      orb.listenToAudio(r.el);
      r.el.addEventListener("ended", () => {
        orb.stopListening();
        orb.set("idle", "Ready", "");
      }, { once: true });
    }
  } catch (err) {
    handle?.setVoice(`couldn't speak: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    if (btn) btn.textContent = "Speak";
  }
}

// ---------------------------------------------------------------------------
// voice input
// ---------------------------------------------------------------------------

let stt: SttEngine | null = null;
let sttKind: SttKind = "none";
let recording = false;
let lastPartial = "";

async function toggleMic(): Promise<void> {
  const live = $("transcript");
  if (recording) { await stopMic(); return; }

  if (!micReady()) {
    live.hidden = false;
    live.dataset.error = "1";
    live.textContent =
      "No speech input available. Set VITE_WORKER_BASE with a Sarvam key, or use a Chromium browser.";
    return;
  }

  live.hidden = false;
  delete live.dataset.error;
  live.dataset.partial = "1";
  live.textContent = "Listening…";
  lastPartial = "";
  lastHeardLanguage = null;
  orb.set("listening", "Listening", "tap again when you're done");
  $("mic-btn").dataset.on = "1";

  const picked = pickStt({
    workerBase: wiring.sttSarvam ? WORKER_BASE : "",
    languageCode: askLang(),
    kind: wiring.sttSarvam ? "sarvam" : "browser",
    onEvent: (e) => {
      if (e.type === "partial") {
        live.dataset.partial = "1";
        live.textContent = e.text;
        // Speculative retrieval: answer the prefix while they are still
        // talking, so the final answer is already warm when they stop. Cached
        // and unmeasured — it never appears in the reported numbers.
        if (e.text.length > 12 && e.text !== lastPartial && !asking) {
          lastPartial = e.text;
          void engine.ask(e.text).catch(() => { /* speculative — failure is free */ });
        }
      } else if (e.type === "final") {
        live.dataset.partial = "0";
        live.textContent = e.text;
        if (e.language) lastHeardLanguage = e.language;
        void stopMic().then(() => ask(e.text, { voice: true }));
      } else if (e.type === "error") {
        live.dataset.error = "1";
        live.textContent = e.message;
        void stopMic();
      }
    },
  });

  if (!picked) {
    live.dataset.error = "1";
    live.textContent = "No speech recogniser available in this browser.";
    return;
  }
  stt = picked.engine;
  sttKind = picked.kind;

  try {
    await stt.start();
    recording = true;
    // The fire pulses with the voice. Without a stream (permission granted but
    // Web Audio unavailable) it simply stays lit without the pulse.
    const stream = stt.micStream;
    if (stream) orb.listenTo(stream);
    if (sttKind === "browser") {
      live.textContent = "Listening… (browser recogniser — Sarvam key not configured)";
    }
  } catch (err) {
    live.dataset.error = "1";
    live.textContent = `Microphone unavailable — ${err instanceof Error ? err.message : String(err)}`;
    $("mic-btn").dataset.on = "0";
    orb.set("idle", "Ready", "");
  }
}

async function stopMic(): Promise<void> {
  if (!recording && !stt) return;
  recording = false;
  $("mic-btn").dataset.on = "0";
  orb.stopListening();
  if (orb.current === "listening") orb.set("thinking", "Looking…", "");
  const s = stt;
  stt = null;
  await s?.stop().catch(() => { /* already closed */ });
}

// ---------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  void ask($<HTMLInputElement>("q").value);
});
// Implicit form submission covers Enter in every browser, but not every
// embedded webview or automation layer dispatches the default action. This
// costs nothing and removes the failure mode where the key does nothing at all.
$("q").addEventListener("keydown", (e) => {
  const ev = e as KeyboardEvent;
  if (ev.key === "Enter" && !ev.isComposing) {
    ev.preventDefault();
    void ask($<HTMLInputElement>("q").value);
  }
});
$("orb").addEventListener("click", () => void toggleMic());
$("mic-btn").addEventListener("click", () => void toggleMic());
for (const chip of document.querySelectorAll<HTMLButtonElement>(".chip")) {
  chip.addEventListener("click", () => void ask(chip.dataset.q ?? ""));
}
$("corpus-toggle").addEventListener("click", () => setCorpus(!(engine?.corpusOn ?? false)));
$("try-corpus").addEventListener("click", () => setCorpus(true));
$("voice-sel").addEventListener("change", () => {
  if ($<HTMLSelectElement>("voice-sel").value === "off") speaker.stop();
  updateVoiceNote();
});

// -- add-a-source panel ------------------------------------------------------
function openAddPanel(): void {
  $("add-panel").hidden = false;
  $("add-btn").setAttribute("aria-expanded", "true");
  $("rail").dataset.hilite = "1";
  setTimeout(() => { delete $("rail").dataset.hilite; }, 1600);
}

const addBtn = $<HTMLButtonElement>("add-btn");
addBtn.addEventListener("click", () => {
  const open = $("add-panel").hidden;
  $("add-panel").hidden = !open;
  addBtn.setAttribute("aria-expanded", String(open));
  if (open) $<HTMLTextAreaElement>("paste-text").focus();
});

// -- responsive pane toggles -------------------------------------------------
function wirePane(btnId: string, paneId: string): void {
  const btn = $<HTMLButtonElement>(btnId);
  const pane = $(paneId);
  const scrim = $("scrim");
  const close = () => {
    delete pane.dataset.open;
    btn.setAttribute("aria-expanded", "false");
    if (!$("rail").dataset.open && !$("studio").dataset.open) scrim.hidden = true;
  };
  btn.addEventListener("click", () => {
    const open = pane.dataset.open === "1";
    if (open) { close(); return; }
    pane.dataset.open = "1";
    btn.setAttribute("aria-expanded", "true");
    scrim.hidden = false;
  });
  scrim.addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && pane.dataset.open === "1") close();
  });
}
wirePane("rail-toggle", "rail");
wirePane("studio-toggle", "studio");

boot().catch((e) => {
  orb.set("refused", "The lamp wouldn't light",
    e instanceof Error ? e.message : String(e));
});

// Exposed for debugging in the console; not used by the app itself.
Object.assign(window as unknown as Record<string, unknown>, {
  chehrag: {
    get engine() { return engine; },
    get store() { return store; },
    get wiring() { return wiring; },
    USER_BASE,
  },
});
