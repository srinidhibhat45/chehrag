/**
 * The conversation.
 *
 * Answers are messages in a thread, not a single result panel that gets
 * overwritten. That is not a cosmetic choice: the interesting comparisons in
 * this system are *between* answers — this question was refused and that one
 * was not, this one took 6 ms and that one 40 — and a panel that replaces
 * itself destroys the only evidence a person has for judging it.
 *
 * The latency reading rides with the answer it describes, as a small chip in
 * the message footer. It used to be a whole screen of its own, which had the
 * pathology of any dedicated metrics tab: the number was somewhere you had to
 * go and look, at which point it was no longer attached to the thing it was a
 * fact about. Expanded, the chip shows the per-stage breakdown inline.
 *
 * Everything is built with DOM calls rather than innerHTML. Passage text is
 * corpus data and user-supplied document text, and the one interesting way to
 * attack a system like this is to get markup out of a retrieved passage and
 * into the page.
 */

import type { RagAnswer } from "../harness/rag";
import { describePlan } from "../harness/deadline";

/** Refusals decided in gate 1, before any vector is computed. */
const INPUT_GATE_REFUSALS = new Set([
  "EMPTY", "TOO_SHORT", "GIBBERISH", "UNSAFE", "INJECTION", "NOT_A_QUESTION",
]);

const REFUSAL_HINT: Record<string, string> = {
  INJECTION:     "that was an instruction, not a question",
  GIBBERISH:     "couldn't make that out",
  NOT_A_QUESTION:"try phrasing it as a question",
  TOO_SHORT:     "too short to retrieve on",
  UNSAFE:        "declined",
  LOW_CONFIDENCE:"nothing in the sources matched well enough",
  NO_AGREEMENT:  "the chunking strategies disagreed",
  UNGROUNDED:    "couldn't ground an answer in the retrieved text",
  NO_SOURCES:    "nothing added yet",
};

export interface BotHandle {
  /** Replace the placeholder with a finished answer. */
  resolve(r: RagAnswer, opts: { userChunks: number; truncated: boolean }): void;
  /** Attach the LLM rewrite, once it has passed the grounding gate. */
  polish(text: string): void;
  /** Show which voice spoke, or why none did. */
  setVoice(label: string): void;
  readonly el: HTMLElement;
}

export interface ChatHooks {
  /** Speak this answer aloud. Returns a label describing what spoke it. */
  onSpeak(text: string, el: HTMLElement): void;
}

export class Chat {
  private welcome: HTMLElement | null;

  constructor(
    private readonly thread: HTMLElement,
    private readonly hooks: ChatHooks,
  ) {
    this.welcome = thread.querySelector(".welcome");
  }

  private clearWelcome(): void {
    if (this.welcome) { this.welcome.remove(); this.welcome = null; }
  }

  /** Append the user's question. `voice` marks it as spoken rather than typed. */
  addUser(text: string, voice = false): void {
    this.clearWelcome();
    const msg = div("msg msg-user");
    const bubble = div("bubble");
    bubble.textContent = text;
    if (voice) bubble.dataset.voice = "1";
    msg.append(bubble);
    this.thread.append(msg);
    this.scroll();
  }

  /**
   * Append a placeholder for an answer that is still being computed.
   *
   * The placeholder exists because retrieval is fast enough that a spinner
   * would flash — but not so fast that nothing at all should acknowledge the
   * question. The wick pulses; that is the entire loading state.
   */
  addPending(): BotHandle {
    this.clearWelcome();
    const msg = div("msg msg-bot");
    msg.dataset.status = "pending";
    const dot = div("msg-dot");
    const body = div("msg-body");
    const text = document.createElement("p");
    text.className = "answer-text";
    text.textContent = "Looking…";
    body.append(text);
    msg.append(dot, body);
    this.thread.append(msg);
    this.scroll();

    return {
      el: msg,
      resolve: (r, opts) => {
        msg.dataset.status = r.status;
        body.replaceChildren();
        this.renderAnswer(body, r, opts);
        this.scroll();
      },
      polish: (t) => {
        const p = div("polish");
        const label = div("polish-label");
        label.append(
          txt("Rewritten by the language model"),
          el("em", " after the fast answer, off the clock"),
        );
        const para = document.createElement("p");
        para.className = "polish-text";
        para.textContent = t;
        p.append(label, para);
        body.append(p);
        this.scroll();
      },
      setVoice: (label) => {
        let note = body.querySelector<HTMLElement>(".meta-note");
        if (!note) {
          note = div("meta-note");
          body.querySelector(".msg-meta")?.append(note);
        }
        note.textContent = label;
      },
    };
  }

  /** A message that isn't an answer — an error, a notice. */
  addNotice(text: string): void {
    this.clearWelcome();
    const msg = div("msg msg-bot");
    msg.dataset.status = "refused";
    const body = div("msg-body");
    const p = document.createElement("p");
    p.className = "answer-text";
    p.textContent = text;
    body.append(p);
    msg.append(div("msg-dot"), body);
    this.thread.append(msg);
    this.scroll();
  }

  // -- answer body ---------------------------------------------------------

  private renderAnswer(
    body: HTMLElement,
    r: RagAnswer,
    opts: { userChunks: number; truncated: boolean },
  ): void {
    const answered = r.status === "answered";

    const text = document.createElement("p");
    text.className = "answer-text";
    text.textContent = r.answer;
    body.append(text);

    if (!answered && r.refusal) {
      const why = div("refusal-why");
      why.textContent = REFUSAL_HINT[r.refusal] ?? r.refusal.toLowerCase().replace(/_/g, " ");
      body.append(why);
    }

    // Nothing was searched, so there is nothing to time, cite or speak. A
    // stopwatch here would be advertising speed for doing no work.
    if (r.refusal === "NO_SOURCES") return;

    const meta = div("msg-meta");
    body.append(meta);

    // -- the speed chip ----------------------------------------------------
    const within = r.totalMs < 200;
    const speed = document.createElement("button");
    speed.className = "speed";
    speed.type = "button";
    speed.dataset.over = within ? "0" : "1";
    speed.setAttribute("aria-expanded", "false");
    speed.append(txt(r.totalMs.toFixed(1)), el("span", "ms", "speed-unit"));
    speed.title = within
      ? `${r.totalMs.toFixed(1)} ms of the 200 ms budget — ${(200 / Math.max(r.totalMs, 0.01)).toFixed(0)}x headroom. Click for the stage breakdown.`
      : `${r.totalMs.toFixed(1)} ms — over the 200 ms budget.`;
    meta.append(speed);

    const stagesPanel = this.stagesPanel(r, opts);
    stagesPanel.hidden = true;
    body.append(stagesPanel);
    speed.addEventListener("click", () => {
      const open = stagesPanel.hidden;
      stagesPanel.hidden = !open;
      speed.setAttribute("aria-expanded", String(open));
    });

    // -- citations ---------------------------------------------------------
    if (r.citations.length) {
      const cites = this.citesPanel(r);
      cites.hidden = true;
      const btn = metaButton(
        `${r.citations.length} source${r.citations.length === 1 ? "" : "s"}`,
      );
      btn.setAttribute("aria-expanded", "false");
      btn.addEventListener("click", () => {
        const open = cites.hidden;
        cites.hidden = !open;
        btn.setAttribute("aria-expanded", String(open));
      });
      meta.append(btn);
      body.append(cites);
    }

    if (answered) {
      // The raw confidence used to be printed here. It is the number the
      // refusal gate is thresholded on, so it is real — but "conf 0.569" asks
      // the reader to know what good looks like, and nothing on the page tells
      // them. It survives where it means something: as the score on each
      // retrieved passage, next to the passage it scores.
      const speak = metaButton("Speak");
      speak.addEventListener("click", () => this.hooks.onSpeak(r.answer, speak));
      meta.append(speak);
    } else if (INPUT_GATE_REFUSALS.has(r.refusal ?? "")) {
      const n = div("meta-note");
      n.textContent = "stopped at the input gate — no embedding spent";
      meta.append(n);
    }
  }

  /**
   * The per-stage breakdown, plus the 200 ms track.
   *
   * The track is always exactly 200 ms wide. That is the point: it is read
   * against the requirement, not against the other bars, so a fast answer looks
   * fast instead of looking like a full bar.
   */
  private stagesPanel(
    r: RagAnswer,
    opts: { userChunks: number; truncated: boolean },
  ): HTMLElement {
    const panel = div("stages-panel");

    const track = div("track");
    track.setAttribute("role", "img");
    track.setAttribute("aria-label", `${r.totalMs.toFixed(1)} milliseconds of the 200 millisecond budget`);
    const fill = div("track-fill");
    fill.style.width = `${Math.max(0.6, Math.min(100, (r.totalMs / 200) * 100))}%`;
    fill.dataset.over = r.totalMs < 200 ? "0" : "1";
    const ticks = div("track-ticks");
    ticks.setAttribute("aria-hidden", "true");
    for (const [left, label] of [["25%", "50"], ["50%", "100"], ["75%", "150"]] as const) {
      const s = document.createElement("span");
      s.style.left = left;
      s.append(document.createElement("i"), txt(label));
      ticks.append(s);
    }
    const cap = document.createElement("span");
    cap.className = "track-cap";
    cap.append(document.createElement("i"), txt("200 ms budget"));
    ticks.append(cap);
    track.append(fill, ticks);
    panel.append(track);

    const max = Math.max(...r.telemetry.map((s) => s.ms), 0.01);
    for (const s of r.telemetry) {
      const row = div("stage-row");
      row.append(el("span", s.name, "stage-name"));

      const bar = div("stage-bar");
      const bf = div("stage-fill");
      bf.dataset.outcome = s.outcome;
      bf.style.width = s.outcome === "skipped" ? "100%" : `${Math.max(1, (s.ms / max) * 100)}%`;
      bar.append(bf);
      row.append(bar);

      row.append(el("span", s.outcome === "skipped" ? "skipped" : s.ms.toFixed(2), "stage-ms"));
      panel.append(row);
    }

    const parts = [`plan: ${describePlan(r.plan)}`];
    if (opts.userChunks) parts.push(`${opts.userChunks.toLocaleString()} of your chunks scanned`);
    if (opts.truncated) parts.push("user-source scan hit its cap");
    if (r.plan.degraded && r.plan.reason) parts.push(r.plan.reason);
    parts.push("retrieval only — speech and LLM rewrite are off this clock");
    const note = document.createElement("p");
    note.className = "meter-plan";
    note.textContent = parts.join(" · ");
    panel.append(note);

    return panel;
  }

  private citesPanel(r: RagAnswer): HTMLElement {
    const panel = div("cites-panel");
    for (const c of r.citations) {
      const cell = div("cite");
      if (c.source.kind === "user") cell.dataset.user = "1";

      const meta = div("cite-meta");
      if (c.source.kind === "user") {
        const badge = el("span", c.source.title ?? "your source", "src-badge");
        badge.title = c.source.title ?? "";
        meta.append(badge);
      }
      for (const s of c.strategies) meta.append(el("span", s, "tag"));
      meta.append(el("span", c.score.toFixed(4), "cite-score"));

      const text = div("cite-text");
      text.textContent = c.text;
      cell.append(meta, text);

      if (c.text.length > 220) {
        const more = document.createElement("button");
        more.className = "cite-more";
        more.type = "button";
        more.textContent = "Show full passage";
        more.addEventListener("click", () => {
          const open = cell.classList.toggle("is-open");
          more.textContent = open ? "Show less" : "Show full passage";
        });
        cell.append(more);
      }
      panel.append(cell);
    }
    return panel;
  }

  /** Keep the newest message in view without yanking the user off a citation
   *  they are mid-read of. */
  private scroll(): void {
    const nearBottom =
      this.thread.scrollHeight - this.thread.scrollTop - this.thread.clientHeight < 220;
    if (nearBottom) this.thread.scrollTop = this.thread.scrollHeight;
  }
}

// -- tiny DOM helpers --------------------------------------------------------

function div(cls: string): HTMLElement {
  const d = document.createElement("div");
  d.className = cls;
  return d;
}

function el(tag: string, text: string, cls?: string): HTMLElement {
  const e = document.createElement(tag);
  e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

function txt(s: string): Text { return document.createTextNode(s); }

function metaButton(label: string): HTMLButtonElement {
  const b = document.createElement("button");
  b.className = "meta-btn";
  b.type = "button";
  b.textContent = label;
  return b;
}
