/**
 * The first-run curtain.
 *
 * Modal because the block is real: the lamp and composer are enabled at the end
 * of `boot()` and SourcesPanel does not exist until the encoder and store do.
 *
 * It names the three things boot() awaits - index and model in parallel, then
 * warm-up - because the wait is an index arriving in the browser, paid once so
 * no later question touches the network. Named, that is a reason to wait.
 *
 * `lift()` runs after the inputs are enabled, never before, and a boot that
 * stalls says so rather than leaving a bar creeping toward a hundred it will
 * never reach.
 */

export type CurtainStep = "index" | "model" | "warm";

/**
 * How much of the bar each phase owns.
 *
 * The index and the model download in parallel and both must finish before
 * anything works, so they split the bar evenly rather than by byte count, which
 * varies per browser and cache state. Giving either one the whole bar would
 * freeze it whenever the other was the slow half.
 *
 * Warm-up is ~150 ms of a multi-second boot but keeps the last sliver: a bar at
 * 100% while the app is still unusable is the same error in the other direction.
 */
const INDEX_SHARE = 0.44;
const MODEL_SHARE = 0.44;

/** No byte and no phase change for this long: report it rather than sit there. */
const STALL_MS = 30_000;

const MB = (b: number): string => (b / 1e6).toFixed(b < 1e7 ? 1 : 0);
const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

export class Curtain {
  private readonly steps = new Map<CurtainStep, HTMLElement>();
  private readonly values = new Map<CurtainStep, HTMLElement>();

  private readonly head: HTMLElement;
  private readonly bar: HTMLElement;
  private readonly fill: HTMLElement;
  private readonly foot: HTMLElement;
  private readonly error: HTMLElement;
  private readonly retry: HTMLButtonElement;
  private readonly sr: HTMLElement;

  /** True while every byte so far has come out of IndexedDB. */
  private allCached = true;
  /** 0..1 per parallel phase, so the bar is their weighted sum. */
  private indexFrac = 0;
  private modelFrac = 0;
  private warmDone = false;
  private lastChange = performance.now();
  private stallTimer = 0;
  private done = false;

  /**
   * `app` carries `inert` from the initial HTML and is released by `lift()`.
   *
   * Where `inert` is unsupported the attribute is ignored and the block degrades
   * to what the curtain does on its own: an opaque, full-bleed layer above
   * everything. The pointer is still stopped; only tabbing behind it survives.
   */
  constructor(
    private readonly root: HTMLElement,
    private readonly app: HTMLElement,
  ) {
    const pick = <T extends HTMLElement>(id: string): T => {
      const el = document.getElementById(id);
      if (!el) throw new Error(`curtain: missing #${id}`);
      return el as T;
    };
    this.head = pick("curtain-h");
    this.bar = pick("curtain-bar");
    this.fill = pick("curtain-fill");
    this.foot = pick("curtain-foot");
    this.error = pick("curtain-error");
    this.retry = pick<HTMLButtonElement>("curtain-retry");
    this.sr = pick("curtain-sr");

    for (const li of this.root.querySelectorAll<HTMLElement>(".curtain-step")) {
      const step = li.dataset.step as CurtainStep;
      this.steps.set(step, li);
      this.values.set(step, pick(`curtain-v-${step}`));
    }

    this.retry.addEventListener("click", () => location.reload());
    this.stallTimer = window.setInterval(() => this.checkStall(), 5_000);
    // Tells the inline fallback in index.html that this module is alive, so it
    // can distinguish "the code never loaded" from "the download is slow".
    this.root.dataset.wired = "1";
  }

  /**
   * Index download progress. `label` is the loader's own and carries `(cached)`
   * for a blob replayed from IndexedDB, which is the difference between
   * downloading 138 MB and reading what is already there.
   */
  indexProgress(loaded: number, total: number, label: string): void {
    if (this.done) return;
    if (!label.includes("(cached)")) this.allCached = false;
    this.touch();

    this.indexFrac = loaded / Math.max(1, total);
    this.paint();
    this.value("index", this.allCached
      ? `${MB(loaded)} MB from cache`
      : `${MB(loaded)} of ~${MB(total)} MB`);

    if (this.allCached) {
      this.foot.textContent = "Opening from cache - this is the fast path.";
    }
  }

  /**
   * Model download progress, in cumulative bytes over its files.
   *
   * Never called on a warm cache, since transformers.js reports nothing for a
   * file it does not fetch. The absence of these calls is not zero progress, so
   * the row is left saying "starting" until `stepDone` ticks it.
   */
  modelProgress(loaded: number, total: number): void {
    if (this.done || total <= 0) return;
    this.touch();
    this.modelFrac = loaded / total;
    this.paint();
    // The bytes finishing is not the phase finishing: ONNX still has to build
    // the session and run a warm-up inference, seconds of work on a slow
    // machine. Left on "135 of ~135 MB" that reads as a step that failed to
    // tick.
    this.value("model", loaded >= total ? "preparing" : `${MB(loaded)} of ~${MB(total)} MB`);
  }

  /** A phase finished. Ticks it, and advances the bar if it owns any of it. */
  stepDone(step: CurtainStep, note?: string): void {
    if (this.done) return;
    this.touch();
    this.steps.get(step)?.setAttribute("data-state", "done");
    this.value(step, note ?? (step === "index" && this.allCached ? "cached" : "ready"));
    // A finished phase is complete whether or not it ever reported a byte, so
    // the bar jumps rather than creeping toward a number that never arrives.
    if (step === "index") this.indexFrac = 1;
    if (step === "model") this.modelFrac = 1;
    if (step === "warm") this.warmDone = true;
    this.paint();
    this.say(`${this.label(step)} ready`);
  }

  /** A phase started. Only `warm` needs this; the other two start with the page. */
  stepRun(step: CurtainStep, note = "working"): void {
    if (this.done) return;
    this.touch();
    this.steps.get(step)?.setAttribute("data-state", "run");
    this.value(step, note);
    this.say(`${this.label(step)}`);
  }

  /**
   * Boot failed. Reported here rather than only on the orb, which is behind this
   * - a curtain that never lifts is the worst way to surface an error.
   */
  fail(message: string): void {
    if (this.done) return;
    window.clearInterval(this.stallTimer);
    this.root.dataset.state = "failed";
    // The heading is this dialog's accessible name, so leaving it on "Lighting
    // the lamp" would have a screen reader announce a failed boot as one in
    // progress. Same words the orb uses for the same condition.
    this.head.textContent = "The lamp wouldn't light";
    this.error.textContent = message;
    this.error.hidden = false;
    this.foot.textContent = "Nothing was uploaded and nothing was lost - this is a load failure.";
    this.retry.hidden = false;
    this.retry.focus();
    this.say(`Chehrag could not start. ${message}`);
  }

  /** Boot finished and the controls behind are live. */
  lift(): void {
    if (this.done) return;
    this.done = true;
    window.clearInterval(this.stallTimer);
    this.setBar(1);
    this.app.removeAttribute("inert");
    this.root.dataset.state = "lifted";
    this.say("Ready. Ask a question.");
    // Removed from the accessibility tree once the fade has run, so a keyboard
    // user never lands on a panel that is no longer visible.
    const drop = (): void => { this.root.hidden = true; };
    this.root.addEventListener("transitionend", drop, { once: true });
    window.setTimeout(drop, 600);
  }

  // -- internals -----------------------------------------------------------

  /** Recompute the bar from whatever every phase has reported so far. */
  private paint(): void {
    this.setBar(this.warmDone
      ? 1
      : clamp01(this.indexFrac) * INDEX_SHARE + clamp01(this.modelFrac) * MODEL_SHARE);
  }

  private setBar(fraction: number): void {
    const pct = Math.max(0, Math.min(1, fraction)) * 100;
    this.fill.style.width = `${pct.toFixed(1)}%`;
    this.bar.setAttribute("aria-valuenow", String(Math.round(pct)));
  }

  private value(step: CurtainStep, text: string): void {
    const el = this.values.get(step);
    if (el) el.textContent = text;
  }

  private label(step: CurtainStep): string {
    return step === "index" ? "Search index"
      : step === "model" ? "Language model"
      : "Warming up";
  }

  private touch(): void { this.lastChange = performance.now(); }

  private say(text: string): void { this.sr.textContent = text; }

  /**
   * A stalled download is not a failed one - the fetch may still be alive - so
   * this reports an abnormal wait and offers a retry rather than claiming
   * failure.
   */
  private checkStall(): void {
    if (this.done || this.root.dataset.state === "failed") return;
    if (performance.now() - this.lastChange < STALL_MS) return;
    this.foot.textContent =
      "This is taking longer than it should. The connection may have dropped.";
    this.retry.hidden = false;
  }
}
