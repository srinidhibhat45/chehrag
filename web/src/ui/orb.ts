/**
 * The orb — main-thread side.
 *
 * Owns the canvas and the microphone envelope, and knows nothing about how the
 * fire is drawn. Three rendering paths, in preference order:
 *
 *   1. **Worker + OffscreenCanvas.** The canvas is transferred once and the
 *      main thread never touches a frame again. This is the one that keeps the
 *      latency claim honest.
 *   2. **In-thread WebGL.** For browsers with WebGL2 but no `OffscreenCanvas`
 *      transfer (Safari was late here, and some privacy extensions block worker
 *      construction outright). Here the renderer is explicitly *stopped* for the
 *      duration of every measured query — see `freeze()`/`thaw()` — on both
 *      the worker and the inline path, so neither can inflate the numbers.
 *   3. **No WebGL at all.** A CSS ember, static. It carries state through colour
 *      only. Nothing breaks.
 *
 * `prefers-reduced-motion` is treated as path 3 regardless of capability. A
 * churning fireball is exactly the kind of thing that setting exists to stop.
 */

import { FireRenderer, type FireInput, type FireState } from "./fire";

export type { FireState };

type Mode = "worker" | "inline" | "static";

export class Orb {
  private canvas: HTMLCanvasElement;
  private worker: Worker | null = null;
  private inline: FireRenderer | null = null;
  private mode: Mode = "static";

  private state: FireState = "dormant";
  private charge = 0;
  private frozen = false;

  // microphone
  private audio: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private mediaSrc: MediaStreamAudioSourceNode | null = null;
  private bins: Uint8Array<ArrayBuffer> | null = null;
  private ampRaf = 0;
  private smoothed = 0;
  private settle = 0;

  constructor(
    private readonly host: HTMLElement,
    private readonly caption: HTMLElement,
    private readonly sub: HTMLElement,
  ) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "orb-canvas";
    this.canvas.setAttribute("aria-hidden", "true");
    host.prepend(this.canvas);

    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!reduced) this.boot();

    host.dataset.mode = this.mode;

    const ro = new ResizeObserver(() => this.pushResize());
    ro.observe(host);
    document.addEventListener("visibilitychange", () => {
      const visible = !document.hidden;
      if (this.mode === "worker") this.worker?.postMessage({ type: "visibility", visible });
      else if (this.mode === "inline") visible && !this.frozen ? this.inline?.start() : this.inline?.stop();
    });
  }

  private boot(): void {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const { w, h } = this.size();

    // Path 1.
    if (typeof this.canvas.transferControlToOffscreen === "function") {
      try {
        const worker = new Worker(new URL("./fire.worker.ts", import.meta.url), { type: "module" });
        const off = this.canvas.transferControlToOffscreen();
        worker.postMessage(
          { type: "init", canvas: off, w, h, dpr, light: this.isLight() },
          [off],
        );
        worker.onmessage = (e: MessageEvent<{ type: string; message?: string }>) => {
          // The worker starts fine but the GL context can still fail inside it,
          // and the canvas is already transferred by then — it cannot be handed
          // back. Degrade to the static ember rather than showing a dead square.
          if (e.data.type === "error") {
            this.worker?.terminate();
            this.worker = null;
            this.mode = "static";
            this.host.dataset.mode = "static";
          }
        };
        this.worker = worker;
        this.mode = "worker";
        return;
      } catch {
        // Worker construction blocked, or the canvas was already transferred.
      }
    }

    // Path 2.
    try {
      this.inline = new FireRenderer(this.canvas);
      this.inline.setLight(this.isLight());
      this.inline.resize(w, h, dpr);
      this.inline.start();
      this.mode = "inline";
    } catch {
      this.mode = "static";   // Path 3.
    }
  }

  private size(): { w: number; h: number } {
    const r = this.host.getBoundingClientRect();
    return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
  }

  private isLight(): boolean {
    const t = document.documentElement.dataset.theme;
    if (t === "light") return true;
    if (t === "dark") return false;
    return matchMedia("(prefers-color-scheme: light)").matches;
  }

  private pushResize(): void {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const { w, h } = this.size();
    if (this.mode === "worker") this.worker?.postMessage({ type: "resize", w, h, dpr });
    else this.inline?.resize(w, h, dpr);
  }

  syncTheme(): void {
    const on = this.isLight();
    if (this.mode === "worker") this.worker?.postMessage({ type: "light", on });
    else this.inline?.setLight(on);
  }

  private push(input: Partial<FireInput>): void {
    if (this.mode === "worker") this.worker?.postMessage({ type: "state", input });
    else this.inline?.set(input);
  }

  // -- state ---------------------------------------------------------------

  /** 0..1 index load progress. */
  setCharge(v: number): void {
    this.charge = v < 0 ? 0 : v > 1 ? 1 : v;
    this.push({ charge: this.charge });
  }

  set(state: FireState, caption?: string, sub?: string): void {
    if (this.settle) { clearTimeout(this.settle); this.settle = 0; }

    this.state = state;
    this.host.dataset.state = state;
    this.push({ state });
    if (caption !== undefined) this.caption.textContent = caption;
    if (sub !== undefined) this.sub.textContent = sub;

    // `answered` and `refused` are momentary; anything arriving after them
    // cancels the return to idle so the orb never flips back mid-transition.
    if (state === "answered" || state === "refused") {
      this.settle = window.setTimeout(() => {
        if (this.state === state) this.set("idle");
      }, state === "answered" ? 1500 : 2600);
    }
  }

  get current(): FireState { return this.state; }

  /**
   * Stop the fire for the duration of a measured query — inline path only.
   *
   * On the worker path this is a no-op by construction, which is the entire
   * point of the worker path. On the inline fallback a frame costs the main
   * thread real time, and a benchmark run that quietly included it would be
   * measuring the animation as much as the pipeline.
   */
  freeze(): void {
    if (this.frozen) return;
    this.frozen = true;
    if (this.mode === "inline") this.inline?.stop();
    // Also paused on the worker path. Off the main thread is not the same as
    // free: the worker still wants a core and the GPU, and a query racing the
    // fire for either is a query that can miss its budget on hardware slower
    // than the one this was written on.
    else if (this.mode === "worker") this.worker?.postMessage({ type: "freeze", on: true });
  }

  thaw(): void {
    if (!this.frozen) return;
    this.frozen = false;
    if (document.hidden) return;
    if (this.mode === "inline") this.inline?.start();
    else if (this.mode === "worker") this.worker?.postMessage({ type: "freeze", on: false });
  }

  /** True when the fire genuinely costs the query path nothing. */
  get offMainThread(): boolean { return this.mode === "worker"; }
  get renderMode(): Mode { return this.mode; }

  // -- amplitude -----------------------------------------------------------

  /**
   * Drive the fire from the microphone while the user speaks.
   *
   * Runs on the main thread because that is where the `MediaStream` lives, but
   * only ever *while recording* — never during a measured query, because you
   * cannot be speaking and have already asked at the same time.
   */
  listenTo(stream: MediaStream): void {
    this.stopListening();
    try {
      const Ctx = window.AudioContext
        ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.audio = new Ctx();
      this.analyser = this.audio.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.75;
      this.mediaSrc = this.audio.createMediaStreamSource(stream);
      this.mediaSrc.connect(this.analyser);
      this.bins = new Uint8Array(this.analyser.frequencyBinCount);
    } catch {
      return;   // no Web Audio: the orb still shows the listening state
    }
    this.pump();
  }

  /**
   * Drive the fire from TTS playback, so the fire "speaks" the answer.
   * Same envelope maths, different source node.
   */
  listenToAudio(el: HTMLAudioElement): void {
    this.stopListening();
    try {
      const Ctx = window.AudioContext
        ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.audio = new Ctx();
      this.analyser = this.audio.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.72;
      const src = this.audio.createMediaElementSource(el);
      src.connect(this.analyser);
      // Must also reach the speakers: routing through an analyser alone
      // silences the element.
      this.analyser.connect(this.audio.destination);
      this.bins = new Uint8Array(this.analyser.frequencyBinCount);
    } catch {
      return;
    }
    this.pump();
  }

  private pump(): void {
    const tick = () => {
      if (!this.analyser || !this.bins) return;
      this.analyser.getByteTimeDomainData(this.bins);
      let sum = 0;
      for (let i = 0; i < this.bins.length; i++) {
        const v = (this.bins[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / this.bins.length);
      // Speech RMS sits low; this curve lifts normal speaking into the visible
      // range without a loud room pegging it at 1. Attack fast, release slow —
      // the reverse reads as a strobe.
      const shaped = clamp01(Math.pow(rms * 4.2, 0.7));
      this.smoothed += (shaped - this.smoothed) * (shaped > this.smoothed ? 0.5 : 0.12);
      this.push({ amp: this.smoothed });
      this.ampRaf = requestAnimationFrame(tick);
    };
    this.ampRaf = requestAnimationFrame(tick);
  }

  stopListening(): void {
    if (this.ampRaf) { cancelAnimationFrame(this.ampRaf); this.ampRaf = 0; }
    this.mediaSrc?.disconnect();
    this.mediaSrc = null;
    this.analyser = null;
    this.bins = null;
    void this.audio?.close().catch(() => { /* already closed */ });
    this.audio = null;
    this.smoothed = 0;
    this.push({ amp: 0 });
  }
}

function clamp01(v: number): number { return v < 0 ? 0 : v > 1 ? 1 : v; }
