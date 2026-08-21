/**
 * App wiring. Two timing paths, kept apart.
 *
 *   fast (measured, 200 ms)   embed, retrieve, fuse, rescore, guard, extract.
 *                             All in the browser, corpus and user sources
 *                             alike. This is what the chip under an answer
 *                             reports, and all it reports.
 *   slow (best effort)        Sarvam STT before the question exists, the model
 *                             that writes the answer, TTS afterwards. None of
 *                             it may touch the measured span.
 *
 * With no generator configured the app still answers, showing the matching
 * passage labelled as a quotation. That is a degraded mode and it says so.
 *
 * Service availability comes from the Worker's /health, so the UI never claims
 * an integration it has not confirmed.
 */

import { loadIndex } from "../retrieval/loader";
import { createEncoder, type BatchEncoder } from "../retrieval/encoder";
import { RagEngine, DEFAULT_CONFIG, USER_BASE, type RagAnswer, type RagConfig } from "../harness/rag";
import { percentiles, CircuitBreaker } from "../harness/pipeline";
import { SourceStore } from "../sources/store";
import { SourcesPanel } from "./sources-panel";
import { Chat, type BotHandle } from "./chat";
import { Orb } from "./orb";
import { Curtain } from "./curtain";
import { pickStt, sttAvailability, type SttEngine, type SttKind } from "../stt";
import { Speaker, scriptLanguage, type TtsProvider } from "../tts/speak";
import { generate, DEFAULT_GEN_CONFIG, type GenSource } from "../answer/generate";
import { loadPrecomputed, type PrecomputedAnswers } from "../answer/precomputed";

const WORKER_BASE = import.meta.env.VITE_WORKER_BASE ?? "";

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

const app = $("app");
const orb = new Orb($("orb"), $("orb-caption"), $("orb-sub"));
// Constructed at module scope rather than inside boot(): it is on screen from
// the initial HTML, with the app already inert behind it.
const curtain = new Curtain($("curtain"), app);
const llmBreaker = new CircuitBreaker(3, 20_000);
const speaker = new Speaker({
  workerBase: WORKER_BASE,
  onNotice: (m) => { $("voice-note").textContent = m; },
});

let engine: RagEngine;
let encoder: BatchEncoder;
let store: SourceStore;
let chat: Chat;
/** Answers written offline for the shipped corpus. Null when none shipped. */
let precomputed: PrecomputedAnswers | null = null;

/** Set while a query is in flight, so voice partials do not stack up. */
let asking = false;
/** Session latency, for the studio readout. */
const sessionTimes: number[] = [];

// -- theme ------------------------------------------------------------------

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

// -- what is searchable -----------------------------------------------------

const CORPUS_KEY = "chehrag-corpus";

/**
 * Is the shipped MS MARCO-XI dataset searchable?
 *
 * On by default, so the app can answer from the first question rather than
 * opening empty. Provenance is handled by attribution instead: every answer
 * names the document it came from, and a corpus passage says so.
 *
 * Reads `!== "0"` rather than `=== "1"`, so an explicit opt-out is remembered
 * while an absent preference means on.
 */
function corpusPref(): boolean { return localStorage.getItem(CORPUS_KEY) !== "0"; }

function setCorpus(on: boolean): void {
  localStorage.setItem(CORPUS_KEY, on ? "1" : "0");
  engine?.setCorpusEnabled(on);
  syncSources();
}

/**
 * Reflect what is searchable, in the rail and in the welcome.
 *
 * Tolerates a missing welcome: `Chat` removes that block once the first question
 * is asked, and this runs again on every source change.
 */
function syncSources(): void {
  const on = engine?.corpusOn ?? corpusPref();

  $("corpus-card").dataset.enabled = on ? "1" : "0";
  const toggle = $<HTMLButtonElement>("corpus-toggle");
  toggle.setAttribute("aria-pressed", String(on));
  toggle.setAttribute("aria-label",
    on ? "Stop searching MS MARCO-XI" : "Search MS MARCO-XI");
}

// -- languages --------------------------------------------------------------

/**
 * What the user can declare they are speaking.
 *
 * `auto` is the default because Sarvam identifies the language and returns it
 * with the transcript. The browser-speech fallback cannot, so the note under the
 * picker says so rather than letting `auto` silently mean Hindi.
 */
const LANGS: Array<{ tag: string; label: string }> = [
  { tag: "auto",  label: "Detect automatically" },
  { tag: "hi-IN", label: "हिन्दी - Hindi" },
  { tag: "bn-IN", label: "বাংলা - Bengali" },
  { tag: "ta-IN", label: "தமிழ் - Tamil" },
  { tag: "te-IN", label: "తెలుగు - Telugu" },
  { tag: "mr-IN", label: "मराठी - Marathi" },
  { tag: "gu-IN", label: "ગુજરાતી - Gujarati" },
  { tag: "kn-IN", label: "ಕನ್ನಡ - Kannada" },
  { tag: "ml-IN", label: "മലയാളം - Malayalam" },
  { tag: "pa-IN", label: "ਪੰਜਾਬੀ - Punjabi" },
  { tag: "od-IN", label: "ଓଡ଼ିଆ - Odia" },
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

// -- the two halves of voice ------------------------------------------------

/**
 * Are answers read aloud?
 *
 * On by default, for every question rather than only spoken ones. Speech is the
 * output this is built around, and the keyboard is a way in rather than a
 * different mode.
 */
let speakOn = localStorage.getItem("chehrag-speak") !== "0";

function setSpeak(on: boolean): void {
  speakOn = on;
  localStorage.setItem("chehrag-speak", on ? "1" : "0");
  const btn = $<HTMLButtonElement>("speak-btn");
  btn.setAttribute("aria-pressed", String(on));
  btn.setAttribute("aria-label", on ? "Stop reading answers aloud" : "Read answers aloud");
  $("speak-label").textContent = on ? "Answers spoken" : "Answers silent";
  if (!on) speaker.stop();
  updateVoiceNote();
}

/**
 * Show the keyboard.
 *
 * Hidden until asked for: a text field beside a microphone is what people reach
 * for out of habit, and reaching for it here means never discovering that
 * speaking works. Once opened it stays open for the session.
 */
function setTyping(on: boolean): void {
  const stage = $("stage");
  if (stage.dataset.typing === (on ? "1" : "0")) return;
  stage.dataset.typing = on ? "1" : "0";
  $<HTMLFormElement>("ask-form").hidden = !on;
  const btn = $<HTMLButtonElement>("type-btn");
  btn.setAttribute("aria-expanded", String(on));
  $("type-label").textContent = on ? "Hide the keyboard" : "Type instead";
  if (on && app.dataset.stage !== "boot") $<HTMLInputElement>("q").focus();
}

// -- service status ---------------------------------------------------------

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
 * Ask the Worker which keys it holds.
 *
 * Not inferred from `VITE_WORKER_BASE` being set: a Worker can be deployed with
 * a Sarvam key and no ElevenLabs key, and either can be revoked later.
 */
async function probeWiring(): Promise<void> {
  const local = sttAvailability(WORKER_BASE);
  wiring.sttBrowser = local.browser;

  // Probed unconditionally rather than only when `VITE_WORKER_BASE` is set.
  // With no Worker configured this is a same-origin request, which the dev
  // server answers - that is what lets `npm run dev` produce real answers from a
  // key in `web/.env.local`. In production without a Worker the SPA fallback
  // returns HTML, `json()` throws, and everything stays off.
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
  } catch { /* no generator reachable - everything stays off */ }
  await speaker.probe();
  renderWiring();
}

/**
 * Reflect what the Worker confirmed.
 *
 * Stated in words under the voice pickers - which voice will speak, and whether
 * speech input is the real path or the browser's fallback - rather than as a row
 * of per-provider status dots, which asks the reader to care about the vendor.
 */
function renderWiring(): void {
  // Never before the index is up: a live lamp on a booting app would record a
  // question there is nothing to answer with.
  $<HTMLButtonElement>("orb").disabled = app.dataset.stage === "boot";
  // Speech is the primary path, so its absence changes the stage rather than
  // leaving a disabled button: the keyboard opens and stays open.
  $("stage").dataset.mic = micReady() ? "1" : "0";
  if (!micReady() && app.dataset.stage !== "boot") setTyping(true);
  updateVoiceNote();
}

function updateVoiceNote(): void {
  const parts: string[] = [];
  if (!speakOn) {
    parts.push("Answers are not spoken.");
  } else {
    const lang = askLang() === "auto" ? "hi-IN" : askLang();
    const route = speaker.routeFor(lang);
    const NAME: Record<TtsProvider, string> = {
      elevenlabs: "ElevenLabs",
      sarvam: "Sarvam",
      browser: "your browser's built-in voice",
      none: "nothing - no voice is available here",
    };
    parts.push(`Answers are read aloud by ${NAME[route]}.`);
  }
  if (!wiring.sttSarvam && wiring.sttBrowser) {
    parts.push("Questions are heard by your browser's own recogniser.");
  }
  if (!wiring.sttSarvam && askLang() === "auto") {
    parts.push("It can't detect the language on its own, so it will assume Hindi - pick one above to be sure.");
  }
  $("voice-note").textContent = parts.join(" ");
}

// -- boot -------------------------------------------------------------------

async function boot(): Promise<void> {
  const t0 = performance.now();
  wireLangs();
  // Before the probe, so a remembered "answers silent" shows from the first
  // frame rather than after the index finishes downloading.
  setSpeak(speakOn);
  void probeWiring();
  checkIsolation();

  // Independent, so they download in parallel; serialising them would double
  // the wait on a first visit.
  const indexPromise = loadIndex("/index", (loaded, total, name) => {
    // Capped at 90% until the encoder is also up: a full ring beside a disabled
    // input reads as finished when it is not.
    orb.setCharge(Math.min(0.9, loaded / total));
    $("orb-sub").textContent =
      `${(loaded / 1e6).toFixed(0)} MB of ~${(total / 1e6).toFixed(0)} MB · ${name}`;
    curtain.indexProgress(loaded, total, name);
  });
  const encoderPromise = createEncoder((loaded, total) => curtain.modelProgress(loaded, total));

  // Ticked off individually rather than after the `Promise.all` below: they
  // finish at different times, and the step list exists to show which one is
  // outstanding. Rejections are swallowed here only - `Promise.all` still sees
  // them, and `boot()`'s caller reports them.
  void indexPromise.then(() => curtain.stepDone("index"), () => { /* reported by boot() */ });
  void encoderPromise.then(() => curtain.stepDone("model"), () => { /* reported by boot() */ });

  orb.set("dormant", "Lighting the lamp...");
  const [index, enc] = await Promise.all([indexPromise, encoderPromise]);
  encoder = enc;
  orb.setCharge(0.95);
  $("orb-sub").textContent = "warming up...";
  curtain.stepRun("warm", "running");

  // Fitted offline by `bench/calibrate.ts`. If that has never run, fall back to
  // the defaults rather than blocking the app.
  let cfg: RagConfig = { ...DEFAULT_CONFIG };
  try {
    const t = await (await fetch("/thresholds.json")).json();
    // Spread over the defaults rather than replacing them: calibrate.ts fits
    // three of these fields and knows nothing about the rest, so a literal
    // object here would reset every threshold it omits to undefined.
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
  // first frame. Warmup forces the corpus on internally regardless, since it
  // warms code paths rather than answering questions.
  engine.setCorpusEnabled(corpusPref());
  new SourcesPanel(store, { onChange: () => { engine.invalidate(); syncSources(); } });
  store.subscribe(() => syncSources());
  // Sources from a previous session replay from IndexedDB. Not awaited: the
  // corpus is already searchable without them.
  void store.restore().catch(() => { /* corrupt store: start empty */ });

  // Answers written offline for the shipped corpus, if this build ships any.
  // Not awaited either: a miss costs a live generation, which is what every
  // question cost before the store existed.
  void loadPrecomputed("/answers", index.manifest.builtAt, index.manifest.dim)
    .then((p) => {
      precomputed = p;
      if (p) console.info(`[chehrag] ${p.size.toLocaleString()} precomputed answers (${p.model})`);
    })
    .catch(() => { /* no store shipped */ });

  chat = new Chat($("thread"), { onSpeak: (text, btn) => void speakAnswer(text, btn) });

  await engine.warmup();
  orb.setCharge(1);
  curtain.stepDone("warm");

  const chunks = Object.values(index.manifest.strategies).reduce((a, s) => a + s.count, 0);
  // The lexical index is counted here but not in `manifest.strategies`: it
  // indexes passages rather than chunks, so it has no chunk count to add, and
  // saying "6 indices" while searching 7 would be wrong in the one place a
  // visitor can check.
  const indices = Object.keys(index.manifest.strategies).length + (index.lexical ? 1 : 0);
  $("corpus-meta").textContent =
    `${index.manifest.numPassages.toLocaleString()} passages · ${chunks.toLocaleString()} chunks · ` +
    `${indices} indices`;
  syncSources();
  $("src-stat").textContent = `ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`;

  app.dataset.stage = "idle";
  $<HTMLInputElement>("q").disabled = false;
  $<HTMLButtonElement>("go").disabled = false;
  renderWiring();
  // Last: raising the curtain claims the app is usable, so everything behind it
  // has to be.
  curtain.lift();

  orb.set("idle", "Ready", micReady()
    ? "tap the lamp, or press space, and speak"
    : "no microphone here - the keyboard is open below");
  // Not the text field, which is hidden. Focusing the lamp means the first
  // Enter or Space starts listening.
  if (micReady()) $<HTMLButtonElement>("orb").focus();
}

function micReady(): boolean { return wiring.sttSarvam || wiring.sttBrowser; }

/**
 * The one hosting mistake that costs the budget.
 *
 * Threaded WASM needs `SharedArrayBuffer`, which needs cross-origin isolation,
 * which needs COOP + COEP on the response serving this page. With the headers
 * wrong, onnxruntime drops to a single thread: the embed stage roughly triples,
 * nothing errors, and the only symptom is worse numbers than on localhost.
 *
 * `public/_headers` is the production copy of `vite.config.ts`'s dev headers.
 * This is what reports a host that does not honour it.
 */
function checkIsolation(): void {
  if (crossOriginIsolated) return;
  console.warn(
    "[chehrag] This page is NOT cross-origin isolated, so WASM is running " +
    "single-threaded and every query will be roughly 3x slower than it needs " +
    "to be. Serve it with:\n" +
    "  Cross-Origin-Opener-Policy: same-origin\n" +
    "  Cross-Origin-Embedder-Policy: credentialless\n" +
    "(see web/public/_headers - Cloudflare Pages reads it automatically).",
  );
}

// -- ask --------------------------------------------------------------------

async function ask(text: string, opts: { voice?: boolean } = {}): Promise<void> {
  const q = text.trim();
  if (!q || !engine || asking) return;
  asking = true;
  $<HTMLInputElement>("q").value = "";
  $("transcript").hidden = true;
  // The lamp shrinks once there is something to read above it.
  app.dataset.asked = "1";

  chat.addUser(q, opts.voice);
  const handle = chat.addPending();
  orb.set("thinking", "Looking...", "");

  try {
    // Bulk embedding is paused for the duration. Ingestion already yields
    // between small batches; pausing removes even that interference, so the
    // budget holds while a document is indexing.
    encoder.pauseBulk();
    // On the inline-WebGL fallback the fire shares this thread, so it stops for
    // the measured span. A no-op on the worker path.
    orb.freeze();

    const res = await engine.ask(q, { skipCache: true });

    const answered = res.status === "answered";

    // Looked up here rather than inside `writeAnswer`, for two reasons.
    //
    // `engine.lastQueryVector` is overwritten by the next question, and
    // `writeAnswer` is deliberately not awaited - so a speculative retrieval
    // fired by a voice partial could replace the vector before an async lookup
    // read it, and the answer would be matched against the wrong question.
    //
    // And a stored answer needs no generator at all, so whether one exists has
    // to be known before deciding what this message is waiting for.
    const topPassage = res.citations[0]?.passageId;
    const stored = answered && precomputed && topPassage !== undefined
      ? precomputed.lookup(engine.lastQueryVector, topPassage)
      : null;

    // `generating: true` holds the message in a waiting state rather than
    // painting the retrieved passage, which would flash on screen and then be
    // replaced by the real answer.
    const willGenerate = answered && (!!stored || wiring.llm);

    handle.resolve(res, {
      userChunks: store?.activeChunks ?? 0,
      truncated: store?.truncated ?? false,
      generating: willGenerate,
    });
    // Only questions that were searched belong in the session figures. A
    // refusal for want of a source never reached the encoder, and folding its
    // 0.1 ms in would flatter every number here.
    //
    // Generation time is excluded for the same reason: this readout measures
    // retrieval, and a network round trip is not part of it.
    if (res.refusal !== "NO_SOURCES") {
      sessionTimes.push(res.totalMs);
      renderSessionStats();
    }

    orb.set(answered ? "answered" : "refused", answered ? "Answered" : "Declined", "");

    // Refusing for want of a source and leaving the user to find the Add button
    // is a dead end, so the panel opens itself.
    if (res.refusal === "NO_SOURCES") openAddPanel();

    // Speaking waits for the written answer: a generated sentence reads aloud
    // far better than a raw passage. Independent of `opts.voice`, because speech
    // is the output rather than a reply in kind to a spoken question.
    const speakWhenReady = answered && speakOn;

    if (stored) {
      // Written offline for this exact question against this exact passage.
      // Retrieval still ran and `res.totalMs` still reports it; what is skipped
      // is the round trip. See `answer/precomputed.ts`.
      handle.beginGeneration();
      handle.streamDelta(stored.answer);
      handle.endPrecomputed();
      if (speakWhenReady) void speakAnswer(stored.answer, null, handle);
    } else if (willGenerate) {
      void writeAnswer(q, res, handle, speakWhenReady);
    } else if (answered) {
      // No generator configured. The passage still contains the answer, but it
      // is a passage, and is labelled as one rather than presented as something
      // the system wrote.
      handle.fallBackToExtract(
        "No answer writer is configured, so this is the passage that matched - quoted, not answered.",
      );
      if (speakWhenReady) void speakAnswer(res.answer, null, handle);
    }
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
  $("stat-p70").textContent = p.p70.toFixed(1);
  $("stat-p100").textContent = p.p100.toFixed(1);
  $("stat-n").textContent = String(sessionTimes.length);
}

/**
 * Write the answer. Off the measured path: this is a round trip to a model.
 *
 * Three endings, each leaving something usable on screen:
 *
 *   answer         streamed into the bubble, then checked against the passages
 *                  it came from (gate 3) before it stands
 *   insufficient   the model read the passages and says they do not answer the
 *                  question, a refusal the retrieval gate missed
 *   unavailable    no key, worker down, timeout. The retrieved passage is shown
 *                  instead, labelled as a quotation
 */
async function writeAnswer(
  query: string,
  base: RagAnswer,
  handle: BotHandle,
  speak: boolean,
): Promise<void> {
  const sources: GenSource[] = base.citations.map((c) => ({
    title: c.source.kind === "user" ? (c.source.title ?? "your source") : "MS MARCO-XI",
    text: c.text,
  }));

  let started = false;
  // Passages already in front of the model, so a second search adds rather than
  // repeats.
  const shown = new Set(base.citations.map((c) => c.passageId));

  const outcome = await llmBreaker.call(() =>
    generate(query, sources, {
      onDelta: (t) => {
        if (!started) { started = true; handle.beginGeneration(); }
        handle.streamDelta(t);
      },
      // The model's one tool, executed here because the index is here. It runs
      // the same sub-5 ms retrieval path as the original question.
      onTool: async (name, args) => {
        if (name !== "search_corpus") return { sources: [], summary: "Unknown tool." };
        const q = String(args.query ?? "").trim();
        if (!q) return { sources: [], summary: "No query was given." };
        const found = await engine.searchAgain(q, shown);
        for (const c of found) shown.add(c.passageId);
        return {
          sources: found.map((c) => ({
            title: c.source.kind === "user" ? (c.source.title ?? "your source") : "MS MARCO-XI",
            text: c.text,
          })),
          summary: found.length
            ? `Found ${found.length} further excerpt${found.length === 1 ? "" : "s"}.`
            : "That search found nothing beyond the excerpts you already have.",
        };
      },
      onToolNote: (note) => handle.setToolNote(note),
    }, { ...DEFAULT_GEN_CONFIG, base: WORKER_BASE }),
  ).catch((): { kind: "unavailable"; reason: string } => (
    // The breaker is open after repeated failures: the system is declining to
    // make a call it expects to fail.
    { kind: "unavailable", reason: "generator unavailable" }
  ));

  if (outcome.kind === "answer") {
    // Gate 3. A fluent model can add a fact that was never in the passages.
    // Checked after streaming rather than before: holding the whole answer back
    // to verify it costs the entire benefit of streaming, and a rare retraction
    // is cheaper than a universal delay.
    //
    // Checked against what the model says it used, not against everything
    // retrieved - and `verifyGenerated` fails an answer citing an excerpt that
    // was never supplied, which is a fabricated citation however well the prose
    // happens to match.
    if (engine.verifyGenerated(outcome.text, outcome.sources, outcome.cited).pass) {
      handle.endGeneration(outcome.ms);
      if (speak) void speakAnswer(outcome.text, null, handle);
      return;
    }
    handle.fallBackToExtract(
      "I couldn't trace all of that back to your sources, so here's the passage it came from instead.",
    );
  } else if (outcome.kind === "insufficient") {
    handle.fallBackToExtract(
      "This is the closest thing I found, but I couldn't turn it into a direct answer to your question.",
    );
  } else {
    // "Not configured" and "configured but not answering" are different facts
    // about the setup, and reporting the first sends the user to check a key
    // that is fine.
    handle.fallBackToExtract(
      outcome.reason === "no generator configured"
        ? "No answer writer is set up, so this is the passage that matched - quoted, not answered."
        : `I couldn't reach the answer writer (${outcome.reason}), so this is the passage that matched - quoted, not answered.`,
    );
  }

  if (speak) void speakAnswer(handle.currentAnswer(), null, handle);
}

// -- speaking ---------------------------------------------------------------

/**
 * Speak an answer.
 *
 * The language comes from the STT result when there is one, since a model that
 * heard the speaker knows more than any heuristic, and otherwise from the script
 * of the answer text. See `scriptLanguage`.
 */
let lastHeardLanguage: string | null = null;

async function speakAnswer(
  text: string,
  btn: HTMLElement | null,
  handle?: BotHandle,
): Promise<void> {
  const lang = lastHeardLanguage ?? scriptLanguage(text);
  if (btn) btn.textContent = "Speaking...";
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

// -- voice input ------------------------------------------------------------

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
    setTyping(true);
    return;
  }

  live.hidden = false;
  delete live.dataset.error;
  live.dataset.partial = "1";
  live.textContent = "Listening...";
  lastPartial = "";
  lastHeardLanguage = null;
  orb.set("listening", "Listening", "tap the lamp again when you're done");
  $("stage").dataset.listening = "1";

  const picked = pickStt({
    workerBase: wiring.sttSarvam ? WORKER_BASE : "",
    languageCode: askLang(),
    kind: wiring.sttSarvam ? "sarvam" : "browser",
    onEvent: (e) => {
      if (e.type === "partial") {
        live.dataset.partial = "1";
        live.textContent = e.text;
        // Speculative retrieval: answer the prefix while the user is still
        // talking, so the final answer is warm when they stop. Cached and
        // unmeasured, so it never appears in the reported numbers.
        if (e.text.length > 12 && e.text !== lastPartial && !asking) {
          lastPartial = e.text;
          void engine.ask(e.text).catch(() => { /* speculative - failure is free */ });
        }
      } else if (e.type === "final") {
        live.dataset.partial = "0";
        live.textContent = e.text;
        if (e.language) lastHeardLanguage = e.language;
        void stopMic().then(() => ask(e.text, { voice: true }));
      } else if (e.type === "notice") {
        // The engine changed how it works and is still working. Report it and
        // keep listening; tearing it down here would destroy the fallback it
        // has just set up.
        live.dataset.partial = "1";
        live.textContent = e.message;
        orb.set("listening", "Listening", "tap the lamp again when you're done");
      } else if (e.type === "error") {
        live.dataset.error = "1";
        live.textContent = e.message;
        // Back to Ready rather than "Looking...": nothing was heard, so nothing
        // is being looked up, and the caption is the app's status line.
        void stopMic("idle");
        // Speech has visibly failed, so open the keyboard rather than leaving
        // a lamp that just stopped working as the only way in.
        setTyping(true);
      }
    },
  });

  if (!picked) {
    live.dataset.error = "1";
    live.textContent = "No speech recogniser available in this browser.";
    return;
  }
  // Held locally as well as in `stt`, because `start()` is a suspension point
  // and the engine can be torn down inside it: a recogniser may report an error
  // or a final transcript before it reports that it started, and `stopMic` nulls
  // `stt` from that callback. Reading the module-level variable after the await
  // would then dereference null.
  const sttEngine = picked.engine;
  stt = sttEngine;
  sttKind = picked.kind;

  try {
    await sttEngine.start();
    if (stt !== sttEngine) return;   // superseded or stopped while starting
    recording = true;
    // The fire pulses with the voice. Without a stream - permission granted but
    // Web Audio unavailable - it stays lit without the pulse.
    const stream = sttEngine.micStream;
    if (stream) orb.listenTo(stream);
    if (sttKind === "browser") {
      live.textContent = "Listening... (browser recogniser - Sarvam key not configured)";
    }
  } catch (err) {
    live.dataset.error = "1";
    live.textContent = `Microphone unavailable - ${err instanceof Error ? err.message : String(err)}`;
    $("stage").dataset.listening = "0";
    orb.set("idle", "Ready", "");
    // A refused microphone is the one case where the keyboard has to appear
    // unasked, since there is otherwise no way in at all.
    setTyping(true);
  }
}

async function stopMic(next: "thinking" | "idle" = "thinking"): Promise<void> {
  if (!recording && !stt) return;
  recording = false;
  $("stage").dataset.listening = "0";
  orb.stopListening();
  if (orb.current === "listening") {
    if (next === "thinking") orb.set("thinking", "Looking...", "");
    else orb.set("idle", "Ready", "tap the lamp, or press space, and speak");
  }
  const s = stt;
  stt = null;
  await s?.stop().catch(() => { /* already closed */ });

  // A recogniser asked to stop is not obliged to emit a final, and the browser
  // one sometimes does not. Without this the caption sits on "Looking..." for a
  // search that never started.
  if (next === "thinking") {
    setTimeout(() => {
      if (!asking && !recording && orb.current === "thinking") {
        orb.set("idle", "Ready", "tap the lamp, or press space, and speak");
      }
    }, 2500);
  }
}

// -- events -----------------------------------------------------------------

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  void ask($<HTMLInputElement>("q").value);
});
// Implicit form submission covers Enter in every browser, but not every
// embedded webview or automation layer dispatches the default action.
$("q").addEventListener("keydown", (e) => {
  const ev = e as KeyboardEvent;
  if (ev.key === "Enter" && !ev.isComposing) {
    ev.preventDefault();
    void ask($<HTMLInputElement>("q").value);
  }
});
$("orb").addEventListener("click", () => void toggleMic());

/**
 * Space is the microphone, so the app is reachable without a mouse. It yields to
 * anything that already wants the key: a focused field, a focused button (where
 * the browser's own activation fires), and a held key repeating.
 */
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
  if (app.dataset.stage === "boot" || !micReady()) return;
  const el = document.activeElement as HTMLElement | null;
  if (el && (el.isContentEditable
    || ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(el.tagName))) return;
  e.preventDefault();
  void toggleMic();
});

$("corpus-toggle").addEventListener("click", () => setCorpus(!(engine?.corpusOn ?? false)));
$("speak-btn").addEventListener("click", () => setSpeak(!speakOn));
$("type-btn").addEventListener("click",
  () => setTyping($("stage").dataset.typing !== "1"));

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
  const message = e instanceof Error ? e.message : String(e);
  orb.set("refused", "The lamp wouldn't light", message);
  // The orb is behind the curtain, so reporting only there would leave the
  // visitor watching a stalled bar with no explanation.
  curtain.fail(message);
});

// Exposed for console debugging. Not used by the app itself.
Object.assign(window as unknown as Record<string, unknown>, {
  chehrag: {
    get engine() { return engine; },
    get store() { return store; },
    get wiring() { return wiring; },
    USER_BASE,
  },
});
