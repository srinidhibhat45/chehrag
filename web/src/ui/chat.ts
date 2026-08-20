/**
 * The conversation.
 *
 * Answers are messages in a thread rather than one result panel that gets
 * overwritten. The interesting comparisons here are between answers — this
 * question was refused and that one was not, this took 6 ms and that took 40 —
 * and a panel that replaces itself destroys the evidence for both.
 *
 * The bubble holds a written answer to the question, and directly under it,
 * never behind a disclosure triangle, the document that answer came from. The
 * retrieved passages sit one click below: evidence, not the answer.
 *
 * The latency reading rides with the answer it describes, as a chip in the
 * message footer rather than on a metrics screen of its own, where it would no
 * longer be attached to the thing it is a fact about. Expanding it shows the
 * per-stage breakdown inline.
 *
 * Retrieval and generation are two numbers and are never summed: retrieval runs
 * in the browser under the 200 ms budget, and the answer is written by a model
 * over the network. One combined figure would misreport both.
 *
 * Everything is built with DOM calls rather than innerHTML. Passage text is
 * corpus data and user-supplied document text, and the obvious attack on a
 * system like this is markup escaping a retrieved passage into the page.
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
  /** Replace the placeholder with the outcome of retrieval. */
  resolve(r: RagAnswer, opts: { userChunks: number; truncated: boolean; generating: boolean }): void;
  /** First token of a generated answer has arrived — clear the waiting state. */
  beginGeneration(): void;
  /** Append a fragment of the generated answer. */
  streamDelta(text: string): void;
  /** Generation finished and passed the grounding gate. */
  endGeneration(ms: number): void;
  /**
   * The answer came from the precomputed store rather than from a model call.
   * Labelled rather than timed: there is no generation figure to report, and
   * showing one would describe a round trip that never happened.
   */
  endPrecomputed(): void;
  /**
   * Generation did not produce a usable answer. `note` says why in the user's
   * terms, and the retrieved passage is shown instead, labelled as a quotation.
   */
  fallBackToExtract(note: string): void;
  /** Show which voice spoke, or why none did. */
  setVoice(label: string): void;
  /** Whatever text is currently being presented as the answer. */
  currentAnswer(): string;
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
   * Retrieval is fast enough that a spinner would only flash, but not so fast
   * that nothing should acknowledge the question. The wick pulses; that is the
   * whole loading state.
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

    // Captured on resolve, so the generation callbacks below can reach what
    // they update without re-querying the DOM on every token.
    let answerEl: HTMLElement = text;
    let extractive = "";
    let generated = "";
    let streaming = false;
    let metaEl: HTMLElement | null = null;

    /** Appears only once there is a settled answer to speak. */
    const addSpeakButton = () => {
      if (!metaEl || metaEl.querySelector(".speak-btn")) return;
      const speak = metaButton("Speak");
      speak.classList.add("speak-btn");
      speak.addEventListener("click", () =>
        this.hooks.onSpeak(generated || extractive, speak));
      metaEl.append(speak);
    };

    return {
      el: msg,
      currentAnswer: () => generated || extractive,

      resolve: (r, opts) => {
        msg.dataset.status = r.status;
        extractive = r.answer;
        body.replaceChildren();
        const parts = this.renderAnswer(body, r, opts);
        answerEl = parts.answerEl;
        metaEl = parts.metaEl;
        if (r.status === "answered" && !opts.generating) addSpeakButton();
        this.scroll();
      },

      beginGeneration: () => {
        streaming = true;
        generated = "";
        answerEl.replaceChildren();
        answerEl.dataset.state = "streaming";
        this.scroll();
      },

      streamDelta: (t) => {
        if (!streaming) return;
        generated += t;
        answerEl.textContent = generated;
        this.scroll();
      },

      endGeneration: (ms) => {
        streaming = false;
        delete answerEl.dataset.state;
        answerEl.textContent = generated.trim();
        if (metaEl) {
          const chip = el("span", `${Math.round(ms)} ms answer`, "gen-ms");
          chip.title =
            "Time for the model to write the answer. Separate from the retrieval " +
            "figure beside it, which is the part of this system that carries the " +
            "200 ms guarantee.";
          metaEl.insertBefore(chip, metaEl.querySelector(".meta-btn"));
        }
        addSpeakButton();
        this.scroll();
      },

      endPrecomputed: () => {
        streaming = false;
        delete answerEl.dataset.state;
        answerEl.textContent = generated.trim();
        if (metaEl) {
          const chip = el("span", "written ahead", "gen-ms");
          chip.dataset.precomputed = "1";
          chip.title =
            "This answer was written offline for the shipped corpus and stored, " +
            "so no model was called. The retrieval figure beside it is unchanged " +
            "— that work still ran for this question.";
          metaEl.insertBefore(chip, metaEl.querySelector(".meta-btn"));
        }
        addSpeakButton();
        this.scroll();
      },

      fallBackToExtract: (note) => {
        streaming = false;
        generated = "";
        delete answerEl.dataset.state;
        answerEl.textContent = extractive;
        // Marked as a quotation rather than presented as an answer: a raw
        // passage shown without saying so is a wrong answer that looks right.
        answerEl.dataset.quoted = "1";
        const why = div("answer-note");
        why.textContent = note;
        answerEl.after(why);
        addSpeakButton();
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
    opts: { userChunks: number; truncated: boolean; generating: boolean },
  ): { answerEl: HTMLElement; metaEl: HTMLElement | null } {
    const answered = r.status === "answered";

    const text = document.createElement("p");
    text.className = "answer-text";
    if (answered && opts.generating) {
      // Retrieval is done and generation has not produced a token yet. The
      // alternative is flashing the raw passage on screen for half a second
      // before replacing it with the answer.
      text.textContent = "Reading your sources…";
      text.dataset.state = "waiting";
    } else {
      text.textContent = r.answer;
    }
    body.append(text);

    if (!answered && r.refusal) {
      const why = div("refusal-why");
      why.textContent = REFUSAL_HINT[r.refusal] ?? r.refusal.toLowerCase().replace(/_/g, " ");
      body.append(why);
    }

    // Nothing was searched, so there is nothing to time, cite or speak.
    if (r.refusal === "NO_SOURCES") return { answerEl: text, metaEl: null };

    // -- attribution -------------------------------------------------------
    //
    // Above the metadata row and outside any disclosure: which document said so
    // is not a diagnostic, it is half of what makes the answer usable. It
    // doubles as the control that opens the passages.
    let cites: HTMLElement | null = null;
    if (r.citations.length) {
      cites = this.citesPanel(r);
      cites.hidden = true;

      const attrib = document.createElement("button");
      attrib.className = "answer-source";
      attrib.type = "button";
      attrib.setAttribute("aria-expanded", "false");
      attrib.append(txt("from "));
      for (const node of sourceLabels(r)) attrib.append(node);
      attrib.append(el("span", r.citations.length === 1 ? "1 passage" : `${r.citations.length} passages`, "src-count"));
      attrib.addEventListener("click", () => {
        const open = cites!.hidden;
        cites!.hidden = !open;
        attrib.setAttribute("aria-expanded", String(open));
      });
      body.append(attrib);
    }

    const meta = div("msg-meta");
    body.append(meta);

    // -- the retrieval stopwatch -------------------------------------------
    const within = r.totalMs < 200;
    const speed = document.createElement("button");
    speed.className = "speed";
    speed.type = "button";
    speed.dataset.over = within ? "0" : "1";
    speed.setAttribute("aria-expanded", "false");
    speed.append(
      txt(r.totalMs.toFixed(1)),
      el("span", "ms", "speed-unit"),
      el("span", "retrieval", "speed-what"),
    );
    speed.title = within
      ? `${r.totalMs.toFixed(1)} ms of the 200 ms budget to find the passages — ${(200 / Math.max(r.totalMs, 0.01)).toFixed(0)}x headroom. Click for the stage breakdown.`
      : `${r.totalMs.toFixed(1)} ms — over the 200 ms retrieval budget.`;
    meta.append(speed);

    const stagesPanel = this.stagesPanel(r, opts);
    stagesPanel.hidden = true;
    speed.addEventListener("click", () => {
      const open = stagesPanel.hidden;
      stagesPanel.hidden = !open;
      speed.setAttribute("aria-expanded", String(open));
    });

    if (!answered && INPUT_GATE_REFUSALS.has(r.refusal ?? "")) {
      meta.append(el("div", "stopped at the input gate — no embedding spent", "meta-note"));
    }

    body.append(stagesPanel);
    if (cites) body.append(cites);

    return { answerEl: text, metaEl: meta };
  }

  /**
   * The per-stage breakdown, plus the 200 ms track.
   *
   * The track is always exactly 200 ms wide, so it reads against the budget
   * rather than against the other bars and a fast answer looks fast.
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
    parts.push("retrieval only — writing the answer, and speech, are off this clock");
    const note = document.createElement("p");
    note.className = "meter-plan";
    note.textContent = parts.join(" · ");
    panel.append(note);

    return panel;
  }

  private citesPanel(r: RagAnswer): HTMLElement {
    const panel = div("cites-panel");
    panel.append(el("p", "The passages the answer was written from.", "cites-lede"));
    for (const c of r.citations) {
      const cell = div("cite");
      if (c.source.kind === "user") cell.dataset.user = "1";

      const meta = div("cite-meta");
      meta.append(el("span", sourceLabel(c.source), "src-badge"));
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

  /** Keep the newest message in view without pulling the reader off a citation
   *  they are part-way through. */
  private scroll(): void {
    const nearBottom =
      this.thread.scrollHeight - this.thread.scrollTop - this.thread.clientHeight < 220;
    if (nearBottom) this.thread.scrollTop = this.thread.scrollHeight;
  }
}

// -- attribution helpers -----------------------------------------------------

/** Named rather than described: the dataset has a name, and citing it is the
 *  point of a citation. */
function sourceLabel(s: { kind: "corpus" | "user"; title?: string }): string {
  return s.kind === "user" ? (s.title || "your source") : "MS MARCO-XI";
}

/**
 * The document names behind an answer, de-duplicated in rank order.
 *
 * Named rather than counted — "from biodata.pdf" is actionable where "from 3
 * sources" is not. Past two documents it becomes a count anyway, since a line of
 * six filenames stops being a sentence.
 */
function sourceLabels(r: RagAnswer): Node[] {
  const seen: string[] = [];
  for (const c of r.citations) {
    const label = sourceLabel(c.source);
    if (!seen.includes(label)) seen.push(label);
  }
  const out: Node[] = [];
  const shown = seen.slice(0, 2);
  shown.forEach((label, i) => {
    if (i > 0) out.push(txt(seen.length > 2 ? ", " : " and "));
    out.push(el("span", label, "src-name"));
  });
  if (seen.length > 2) out.push(txt(` and ${seen.length - 2} more`));
  return out;
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
