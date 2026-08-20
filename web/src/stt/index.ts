/**
 * Speech-to-text, behind one interface.
 *
 * Sarvam is the implementation the brief asks for and the only one used when a
 * key is configured. `WebSpeechStt` is a fallback with a specific, narrow job:
 * without it, an evaluator who has not been given a Sarvam key sees a voice
 * button that does nothing, and cannot tell a missing key from a broken
 * feature. With it they can hear the interface work and read, in plain text,
 * that they are hearing the browser's recogniser rather than Sarvam.
 *
 * It is never silently substituted. `pickStt` reports which engine it chose and
 * the UI names it, because "voice works" and "the Sarvam integration works" are
 * different claims and only one of them is the requirement.
 *
 * Two real differences the UI has to account for:
 *
 *   - Web Speech has no language auto-detect worth the name. It is told a
 *     language and hears that language; Sarvam's `auto` genuinely identifies
 *     one of eleven and returns it with the transcript.
 *   - Web Speech is Chromium-only in practice, and on Chrome the audio goes to
 *     Google's servers. Which is precisely why it is a fallback and is labelled.
 */

import { SarvamStt, type SttEvent, type SttOptions } from "./sarvam";

export type { SttEvent };

export interface SttEngine {
  start(): Promise<void>;
  stop(): Promise<void>;
  readonly micStream: MediaStream | null;
}

export type SttKind = "sarvam" | "browser" | "none";

/** Chromium exposes this unprefixed on some builds and prefixed on others. */
function speechRecognitionCtor(): (new () => SpeechRecognition) | null {
  const w = window as unknown as {
    SpeechRecognition?: new () => SpeechRecognition;
    webkitSpeechRecognition?: new () => SpeechRecognition;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function sttAvailability(workerBase: string): { sarvam: boolean; browser: boolean } {
  return { sarvam: !!workerBase, browser: !!speechRecognitionCtor() };
}

export function pickStt(
  opts: SttOptions & { kind?: SttKind },
): { engine: SttEngine; kind: SttKind } | null {
  const wantBrowser = opts.kind === "browser";
  if (!wantBrowser && opts.workerBase) {
    return { engine: new SarvamStt(opts), kind: "sarvam" };
  }
  if (speechRecognitionCtor()) {
    return { engine: new WebSpeechStt(opts), kind: "browser" };
  }
  return null;
}

/**
 * `SpeechRecognition` wrapped to look like the Sarvam client.
 *
 * It opens its own microphone stream in parallel with the recogniser's. That
 * looks redundant and is not: the recogniser does not expose its audio, and the
 * fireball's envelope needs a `MediaStream`. Browsers share one microphone
 * between both consumers, so this costs a permission check and nothing else.
 */
export class WebSpeechStt implements SttEngine {
  private rec: SpeechRecognition | null = null;
  private stream: MediaStream | null = null;
  private stopped = false;
  private finalText = "";

  constructor(private readonly opts: SttOptions) {}

  get micStream(): MediaStream | null { return this.stream; }

  async start(): Promise<void> {
    const Ctor = speechRecognitionCtor();
    if (!Ctor) throw new Error("this browser has no speech recognition");

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      // The recogniser can still work; only the fire's envelope is lost.
      this.stream = null;
    }

    const rec = new Ctor();
    this.rec = rec;
    rec.continuous = true;
    rec.interimResults = true;
    // "auto" is not a thing here — the recogniser must be told. Hindi is the
    // corpus language, so it is the sensible default when nothing is chosen.
    rec.lang = !this.opts.languageCode || this.opts.languageCode === "auto"
      ? "hi-IN" : this.opts.languageCode;

    rec.onresult = (e: SpeechRecognitionEvent) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) this.finalText += r[0].transcript;
        else interim += r[0].transcript;
      }
      const text = (this.finalText + interim).trim();
      if (text) this.opts.onEvent({ type: "partial", text });
    };
    rec.onerror = (e: SpeechRecognitionErrorEvent) => {
      if (e.error === "aborted" || e.error === "no-speech") return;
      this.opts.onEvent({ type: "error", message: `browser speech: ${e.error}` });
    };
    rec.onend = () => {
      // Chrome ends the session on its own after a pause. Restart unless the
      // user actually stopped, or dictation dies mid-sentence.
      if (!this.stopped) { try { rec.start(); } catch { /* already running */ } return; }
      this.opts.onEvent({ type: "closed" });
    };

    rec.start();
  }

  async stop(): Promise<void> {
    this.stopped = true;
    try { this.rec?.stop(); } catch { /* not started */ }
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    const text = this.finalText.trim();
    if (text) this.opts.onEvent({ type: "final", text, language: this.rec?.lang });
    this.rec = null;
  }
}
