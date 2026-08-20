/**
 * Spoken answers, routed across two providers because for this corpus they are
 * not interchangeable:
 *
 *   ElevenLabs `eleven_flash_v2_5` — ~75 ms model latency across 32 languages,
 *   but of the fourteen in MSMARCO-XI it covers only Hindi and Tamil.
 *
 *   Sarvam `bulbul:v2` — native Indic, covering Bengali, Gujarati, Kannada,
 *   Malayalam, Marathi, Odia, Punjabi and Telugu, exactly the set ElevenLabs
 *   leaves out.
 *
 * So a language goes to ElevenLabs when ElevenLabs speaks it and to Sarvam
 * otherwise, which is the only arrangement under which every language in the
 * corpus can be spoken at all.
 *
 * The browser's `speechSynthesis` sits under both as a last resort, labelled as
 * a fallback wherever it is used, so the interface stays demonstrable with no
 * keys configured.
 *
 * None of this is inside the 200 ms budget or measured against it: speech
 * happens after an answer already exists on screen.
 */

export type TtsProvider = "elevenlabs" | "sarvam" | "browser" | "none";

export interface SpeakResult {
  provider: TtsProvider;
  /** Time from request to the first audible sample, ms. */
  firstAudioMs: number;
  /** The element actually playing, so the orb can read its envelope. */
  el: HTMLAudioElement | null;
}

/** Languages `eleven_flash_v2_5` speaks, restricted to the ones this corpus can
 *  produce. Everything else in MSMARCO-XI routes to Sarvam. */
const ELEVEN_LANGS = new Set(["en", "hi", "ta"]);

/** Sarvam `bulbul:v2` target languages, as BCP-47 tags the API expects. */
const SARVAM_TAGS: Record<string, string> = {
  hi: "hi-IN", bn: "bn-IN", ta: "ta-IN", te: "te-IN", mr: "mr-IN",
  gu: "gu-IN", kn: "kn-IN", ml: "ml-IN", pa: "pa-IN", od: "od-IN",
  or: "od-IN", en: "en-IN",
};

export interface SpeakerOptions {
  workerBase: string;
  /** Called when the provider actually used differs from what was expected. */
  onNotice?: (msg: string) => void;
}

export class Speaker {
  private current: HTMLAudioElement | null = null;
  private currentUrl: string | null = null;
  private abort: AbortController | null = null;
  private available: { elevenlabs: boolean; sarvam: boolean } = {
    elevenlabs: false, sarvam: false,
  };
  private probed = false;

  constructor(private readonly opts: SpeakerOptions) {}

  /**
   * Ask the Worker which voice back-ends have keys. Done once at boot, so the UI
   * can name the provider up front rather than discovering it at play time.
   */
  async probe(): Promise<{ elevenlabs: boolean; sarvam: boolean }> {
    if (this.probed) return this.available;
    this.probed = true;
    if (!this.opts.workerBase) return this.available;
    try {
      const r = await fetch(`${this.opts.workerBase}/health`);
      const h = await r.json() as { tts?: { elevenlabs?: boolean; sarvam?: boolean } };
      this.available = {
        elevenlabs: !!h.tts?.elevenlabs,
        sarvam: !!h.tts?.sarvam,
      };
    } catch { /* worker down — browser voice only */ }
    return this.available;
  }

  /** Which provider a given language would actually use, right now. */
  routeFor(lang: string): TtsProvider {
    const base = lang.split("-")[0].toLowerCase();
    if (this.available.elevenlabs && ELEVEN_LANGS.has(base)) return "elevenlabs";
    if (this.available.sarvam && SARVAM_TAGS[base]) return "sarvam";
    if (this.available.elevenlabs) return "elevenlabs";   // multilingual model may still cope
    if (typeof speechSynthesis !== "undefined") return "browser";
    return "none";
  }

  stop(): void {
    this.abort?.abort();
    this.abort = null;
    if (this.current) {
      this.current.pause();
      this.current.src = "";
      this.current = null;
    }
    if (this.currentUrl) { URL.revokeObjectURL(this.currentUrl); this.currentUrl = null; }
    if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
  }

  /**
   * Speak `text`. Resolves once audio has started rather than finished, so the
   * caller can wire up the orb envelope without blocking.
   */
  async speak(text: string, lang: string): Promise<SpeakResult> {
    this.stop();
    const t0 = performance.now();
    const body = text.trim().slice(0, 900);
    if (!body) return { provider: "none", firstAudioMs: 0, el: null };

    const route = this.routeFor(lang);
    if (route === "elevenlabs" || route === "sarvam") {
      try {
        const el = await this.speakRemote(body, lang, route);
        return { provider: route, firstAudioMs: performance.now() - t0, el };
      } catch (err) {
        if ((err as Error).name === "AbortError") throw err;
        this.opts.onNotice?.(
          `${route} voice unavailable (${(err as Error).message}) — using the browser voice`,
        );
      }
    }
    const ok = this.speakBrowser(body, lang);
    return {
      provider: ok ? "browser" : "none",
      firstAudioMs: performance.now() - t0,
      el: null,
    };
  }

  /**
   * Fetch synthesised audio through the Worker and play it.
   *
   * A whole-blob fetch rather than MediaSource streaming. A spoken answer is one
   * or two sentences, under four seconds of audio, and at that length MSE —
   * codec strings, buffer appends, the per-browser quirks matrix — buys a couple
   * of hundred milliseconds for a class of failure that is silent and hard to
   * reproduce. The Worker still streams from the provider; only this hop is
   * buffered.
   */
  private async speakRemote(text: string, lang: string, provider: TtsProvider): Promise<HTMLAudioElement> {
    const ctrl = new AbortController();
    this.abort = ctrl;

    const res = await fetch(`${this.opts.workerBase}/tts`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        text,
        provider,
        language: provider === "sarvam"
          ? (SARVAM_TAGS[lang.split("-")[0].toLowerCase()] ?? "hi-IN")
          : lang.split("-")[0].toLowerCase(),
      }),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`tts ${res.status}${detail ? ` ${detail.slice(0, 120)}` : ""}`);
    }

    const blob = await res.blob();
    if (!blob.size) throw new Error("empty audio");

    const url = URL.createObjectURL(blob);
    // A fresh element per utterance: `createMediaElementSource` may be called
    // only once per element, and the orb needs a source node each time.
    const el = new Audio(url);
    el.preload = "auto";
    this.current = el;
    this.currentUrl = url;
    el.addEventListener("ended", () => {
      if (this.currentUrl === url) { URL.revokeObjectURL(url); this.currentUrl = null; }
    }, { once: true });

    await el.play();
    return el;
  }

  /** Last resort, and labelled as one wherever it is used. */
  private speakBrowser(text: string, lang: string): boolean {
    if (typeof speechSynthesis === "undefined") return false;
    try {
      const u = new SpeechSynthesisUtterance(text);
      u.lang = lang.includes("-") ? lang : `${lang}-IN`;
      const match = speechSynthesis.getVoices()
        .find((v) => v.lang.toLowerCase().startsWith(lang.split("-")[0].toLowerCase()));
      if (match) u.voice = match;
      u.rate = 1.0;
      speechSynthesis.speak(u);
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Guess the language of an answer from its script.
 *
 * Script detection, not language detection: Devanagari is shared by Hindi,
 * Marathi, Nepali and Sanskrit, and this cannot separate them. It exists to
 * route audio to a voice that will not mangle the text, and script is the
 * property that matters for that.
 *
 * A Sarvam transcript's `language_code` is used instead when there is one, since
 * a model that heard the speaker knows more than any script heuristic.
 */
export function scriptLanguage(text: string): string {
  const counts: Array<[string, number]> = [
    ["bn", count(text, /[ঀ-৿]/g)],
    ["gu", count(text, /[઀-૿]/g)],
    ["pa", count(text, /[਀-੿]/g)],
    ["or", count(text, /[଀-୿]/g)],
    ["ta", count(text, /[஀-௿]/g)],
    ["te", count(text, /[ఀ-౿]/g)],
    ["kn", count(text, /[ಀ-೿]/g)],
    ["ml", count(text, /[ഀ-ൿ]/g)],
    ["ur", count(text, /[؀-ۿ]/g)],
    ["hi", count(text, /[ऀ-ॿ]/g)],
  ];
  let best = "en", bestN = 0;
  for (const [code, n] of counts) if (n > bestN) { best = code; bestN = n; }
  // A stray Devanagari character in an English sentence must not flip the voice.
  return bestN >= 3 ? best : "en";
}

function count(s: string, re: RegExp): number {
  return (s.match(re) ?? []).length;
}
