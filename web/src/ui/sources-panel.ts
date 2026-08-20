/**
 * The sources rail.
 *
 * Three ways in — paste, file, link — and one list out. The list reports how
 * many passages and chunks each source produced, because "added successfully"
 * says nothing about whether a dropped PDF had a text layer.
 *
 * A permanent rail rather than a modal drawer: the sources are the subject, so
 * they stay visible while an answer is read, and one can be switched off and the
 * question asked again without losing your place.
 *
 * Rendering rebuilds the list on every change. With a handful of sources that is
 * cheaper than diffing, and it happens off the query path.
 */

import type { Source, SourceStore } from "../sources/store";
import { SourceCancelled } from "../sources/store";
import { IngestError } from "../sources/ingest";

const KIND_LABEL: Record<Source["kind"], string> = {
  paste: "TXT", file: "FILE", url: "WEB",
};

export interface PanelHooks {
  /** Called whenever the indexed corpus changes, so caches can be dropped. */
  onChange: () => void;
}

export class SourcesPanel {
  private rail = byId("rail");
  private list = byId("src-list");
  private empty = byId("src-empty");
  private stat = byId("src-stat");
  private error = byId("add-error");

  constructor(
    private readonly store: SourceStore,
    private readonly hooks: PanelHooks,
  ) {
    this.wireModes();
    this.wirePaste();
    this.wireFiles();
    this.wireUrl();
    store.subscribe(() => this.render());
    this.render();
  }

  // -- add modes -----------------------------------------------------------

  private wireModes(): void {
    const panes: Record<string, HTMLElement> = {
      paste: byId("pane-paste"), file: byId("pane-file"), url: byId("pane-url"),
    };
    for (const btn of document.querySelectorAll<HTMLButtonElement>(".seg-btn")) {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.mode!;
        for (const b of document.querySelectorAll<HTMLButtonElement>(".seg-btn")) {
          const on = b === btn;
          b.classList.toggle("is-on", on);
          b.setAttribute("aria-selected", String(on));
        }
        for (const [k, el] of Object.entries(panes)) el.hidden = k !== mode;
        this.clearError();
      });
    }
  }

  private wirePaste(): void {
    const text = byId<HTMLTextAreaElement>("paste-text");
    const title = byId<HTMLInputElement>("paste-title");
    const btn = byId<HTMLButtonElement>("paste-add");
    btn.addEventListener("click", () => {
      const body = text.value;
      void this.run(btn, () => this.store.addPaste(body, title.value), () => {
        text.value = ""; title.value = "";
      });
    });
  }

  private wireFiles(): void {
    const drop = byId("drop");
    const input = byId<HTMLInputElement>("file-input");

    input.addEventListener("change", () => {
      const files = [...(input.files ?? [])];
      input.value = "";               // so the same file can be re-added
      void this.addFiles(files);
    });

    // The whole rail is a drop target, not just the dashed box, so a dragged
    // file does not have to be aimed. Dropping also opens the add panel if it
    // was collapsed, so progress is visible.
    for (const ev of ["dragenter", "dragover"] as const) {
      this.rail.addEventListener(ev, (e) => {
        e.preventDefault();
        drop.classList.add("is-over");
      });
    }
    for (const ev of ["dragleave", "drop"] as const) {
      this.rail.addEventListener(ev, (e) => {
        e.preventDefault();
        if (ev === "dragleave" && this.rail.contains(e.relatedTarget as Node)) return;
        drop.classList.remove("is-over");
      });
    }
    this.rail.addEventListener("drop", (e) => {
      const files = [...(e.dataTransfer?.files ?? [])];
      if (files.length) { this.openAddPanel(); void this.addFiles(files); }
    });
  }

  private openAddPanel(): void {
    byId("add-panel").hidden = false;
    byId("add-btn").setAttribute("aria-expanded", "true");
  }

  private async addFiles(files: File[]): Promise<void> {
    this.clearError();
    // Sequential rather than parallel: they queue behind the same encoder
    // either way, and one at a time keeps the progress bars meaningful.
    for (const f of files) {
      try {
        await this.store.addFile(f);
        this.hooks.onChange();
      } catch (err) {
        this.showError(err);
      }
    }
  }

  private wireUrl(): void {
    const input = byId<HTMLInputElement>("url-input");
    const btn = byId<HTMLButtonElement>("url-add");
    const go = () => {
      const url = input.value;
      void this.run(btn, () => this.store.addUrl(url), () => { input.value = ""; });
    };
    btn.addEventListener("click", go);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  }

  /** Shared add-flow: disable the button, run, report, refresh. */
  private async run(btn: HTMLButtonElement, fn: () => Promise<Source>, ok: () => void): Promise<void> {
    this.clearError();
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Adding…";
    try {
      await fn();
      ok();
      this.hooks.onChange();
    } catch (err) {
      this.showError(err);
    } finally {
      btn.disabled = false;
      btn.textContent = label;
    }
  }

  private showError(err: unknown): void {
    if (err instanceof SourceCancelled) return;
    const msg = err instanceof IngestError ? err.message
      : err instanceof Error ? err.message : String(err);
    const hint = err instanceof IngestError ? err.hint : undefined;
    this.error.replaceChildren();
    const b = document.createElement("b");
    b.textContent = msg;
    this.error.append(b);
    if (hint) {
      const s = document.createElement("span");
      s.textContent = hint;
      this.error.append(s);
    }
    this.error.hidden = false;
    this.openAddPanel();
  }

  private clearError(): void { this.error.hidden = true; }

  // -- rendering -----------------------------------------------------------

  private render(): void {
    const sources = this.store.list();
    const ready = sources.filter((s) => s.status === "ready");
    const enabled = ready.filter((s) => s.enabled);

    this.stat.textContent = ready.length
      ? `${enabled.length} of ${ready.length} on · ` +
        `${enabled.reduce((a, s) => a + s.chunks, 0).toLocaleString()} chunks searched`
      : this.stat.textContent.startsWith("ready in") ? this.stat.textContent : "";

    this.empty.hidden = sources.length > 0;
    this.list.replaceChildren();
    for (const s of sources) this.list.append(this.row(s));
  }

  private row(s: Source): HTMLElement {
    const el = document.createElement("div");
    el.className = "src";
    el.dataset.status = s.status;
    el.dataset.enabled = s.enabled ? "1" : "0";

    const kind = document.createElement("span");
    kind.className = "src-kind";
    kind.textContent = KIND_LABEL[s.kind];

    const body = document.createElement("div");
    body.className = "src-body";
    const title = document.createElement("div");
    title.className = "src-title";
    title.textContent = s.title;
    title.title = s.title;
    const meta = document.createElement("div");
    meta.className = "src-meta";
    if (s.status === "error") {
      meta.dataset.error = "1";
      meta.textContent = s.error ?? "failed";
    } else if (s.status === "indexing") {
      meta.textContent = `indexing ${Math.round(s.progress * 100)}%`;
    } else {
      const bits = [
        `${s.passages.toLocaleString()} passages`,
        `${s.chunks.toLocaleString()} chunks`,
      ];
      if (s.note) bits.push(s.note);
      meta.textContent = bits.join(" · ");
    }
    meta.title = meta.textContent;
    body.append(title, meta);

    const actions = document.createElement("div");
    actions.className = "src-actions";

    if (s.status === "ready") {
      const toggle = document.createElement("button");
      toggle.className = "src-toggle";
      toggle.type = "button";
      toggle.setAttribute("aria-pressed", String(s.enabled));
      toggle.setAttribute("aria-label", s.enabled ? `Stop searching ${s.title}` : `Search ${s.title}`);
      toggle.addEventListener("click", () => {
        this.store.setEnabled(s.id, !s.enabled);
        this.hooks.onChange();
      });
      actions.append(toggle);
    }

    const x = document.createElement("button");
    x.className = "src-x";
    x.type = "button";
    x.setAttribute("aria-label", s.status === "indexing" ? `Cancel ${s.title}` : `Remove ${s.title}`);
    x.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
            d="M6 6l12 12M18 6L6 18"/></svg>`;
    x.addEventListener("click", () => {
      this.store.remove(s.id);
      this.hooks.onChange();
    });
    actions.append(x);

    el.append(kind, body, actions);

    if (s.status === "indexing") {
      const bar = document.createElement("div");
      bar.className = "src-progress";
      const fill = document.createElement("i");
      fill.style.width = `${Math.round(s.progress * 100)}%`;
      bar.append(fill);
      el.append(bar);
    }
    return el;
  }
}

function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
}
