/**
 * Turning whatever the user adds into plain text.
 *
 * Pasted text, a dropped file and a URL all reduce to a title and a body.
 * Retrieval never sees the original format, so format knowledge is confined to
 * this file.
 *
 * Heavy extractors are dynamically imported: someone who only pastes text should
 * not pay for the PDF parser, which is larger than the rest of the app.
 *
 * None of this is on the 200ms path. Ingestion happens once per source; queries
 * only touch the vectors it produced.
 */

export type SourceKind = "paste" | "file" | "url";

export interface Extracted {
  title: string;
  text: string;
  /** Shown in the sources panel, so the user can see what was actually parsed. */
  note?: string;
}

export class IngestError extends Error {
  constructor(message: string, readonly hint?: string) {
    super(message);
    this.name = "IngestError";
  }
}

/** Hard ceiling per source. Well above any realistic document, low enough that
 *  one pathological paste cannot exhaust memory or the embedding queue. */
export const MAX_CHARS = 600_000;
const MAX_FILE_BYTES = 32 * 1024 * 1024;

/**
 * Collapse the whitespace damage each extractor produces in its own way: PDFs
 * emit line breaks mid-sentence, HTML emits runs of non-breaking spaces, DOCX
 * emits none at all. Sentence splitting downstream assumes normal spacing, so
 * normalising here keeps the chunker from having to know where text came from.
 */
function normalise(raw: string): string {
  return raw
    .replace(/\r\n?/g, "\n")
    .replace(/ /g, " ")
    // Join lines that were wrapped mid-sentence, keep real paragraph breaks.
    .replace(/([^\n।.!?:;])\n(?!\n)/g, "$1 ")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function clamp(text: string): { text: string; truncated: boolean } {
  if (text.length <= MAX_CHARS) return { text, truncated: false };
  return { text: text.slice(0, MAX_CHARS), truncated: true };
}

// -- paste ------------------------------------------------------------------

export function ingestPaste(raw: string, title?: string): Extracted {
  const norm = normalise(raw);
  if (norm.length < 20) {
    throw new IngestError("That's too short to be a useful source.",
      "Paste at least a paragraph.");
  }
  const { text, truncated } = clamp(norm);
  return {
    title: title?.trim() || firstLineTitle(norm),
    text,
    note: truncated ? `truncated to ${MAX_CHARS.toLocaleString()} characters` : undefined,
  };
}

/** A title from the text itself beats "Untitled 3" in the sources list. */
function firstLineTitle(text: string): string {
  const line = text.split("\n").find((l) => l.trim().length > 0) ?? "Pasted text";
  const t = line.trim().replace(/^#+\s*/, "");
  return t.length > 64 ? `${t.slice(0, 61)}…` : t;
}

// -- files ------------------------------------------------------------------

const PLAIN_EXT = /\.(txt|md|markdown|csv|tsv|log|rst|org|tex|srt|vtt)$/i;
const CODE_EXT = /\.(json|jsonl|ya?ml|toml|ini|xml|py|ts|tsx|js|jsx|go|rs|java|c|h|cpp|sh|sql)$/i;
const HTML_EXT = /\.(html?|xhtml)$/i;

export async function ingestFile(file: File): Promise<Extracted> {
  if (file.size > MAX_FILE_BYTES) {
    throw new IngestError(`${file.name} is ${(file.size / 1e6).toFixed(0)} MB.`,
      `The limit is ${MAX_FILE_BYTES / 1e6} MB.`);
  }
  const name = file.name;
  const title = name.replace(/\.[^.]+$/, "");

  let text: string;
  let note: string | undefined;

  if (/\.pdf$/i.test(name)) {
    const r = await extractPdf(file);
    text = r.text;
    note = r.note;
  } else if (/\.docx$/i.test(name)) {
    text = await extractDocx(file);
  } else if (HTML_EXT.test(name)) {
    text = extractHtml(await file.text());
  } else if (/\.jsonl?$/i.test(name)) {
    text = extractJson(await file.text());
  } else if (PLAIN_EXT.test(name) || CODE_EXT.test(name) || file.type.startsWith("text/")) {
    text = await file.text();
  } else {
    // Unknown extension: try it as text rather than refusing outright, but
    // reject binary — decoded binary is noise that would poison retrieval.
    const guess = await file.text();
    if (looksBinary(guess)) {
      throw new IngestError(`Can't read ${name}.`,
        "Supported: PDF, DOCX, TXT, MD, HTML, CSV, JSON and other text formats.");
    }
    text = guess;
  }

  const norm = normalise(text);
  if (norm.length < 20) {
    throw new IngestError(`${name} had no readable text.`,
      "If it's a scanned PDF, the pages are images — run OCR first.");
  }
  const clamped = clamp(norm);
  return {
    title,
    text: clamped.text,
    note: clamped.truncated ? `truncated to ${MAX_CHARS.toLocaleString()} characters` : note,
  };
}

/** U+FFFD and control bytes are what a mis-decoded binary looks like. */
function looksBinary(s: string): boolean {
  const sample = s.slice(0, 4000);
  let bad = 0;
  for (const c of sample) {
    const code = c.codePointAt(0)!;
    if (code === 0xfffd || (code < 9) || (code > 13 && code < 32)) bad++;
  }
  return bad / Math.max(1, sample.length) > 0.05;
}

/**
 * PDF text extraction via pdf.js, imported only when a PDF arrives.
 *
 * Scanned PDFs have no text layer at all. That is detected and reported, since
 * silently indexing an empty document is worse than refusing it.
 */
async function extractPdf(file: File): Promise<{ text: string; note?: string }> {
  const pdfjs = await import("pdfjs-dist");
  // Bundled through Vite (`?url`) rather than fetched from a CDN: the app must
  // work offline once loaded, and a CDN is another origin to trust.
  const workerUrl = (await import("pdfjs-dist/build/pdf.worker.min.mjs?url")).default;
  pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

  const task = pdfjs.getDocument({
    data: new Uint8Array(await file.arrayBuffer()),
    useSystemFonts: false,
  });
  const doc = await task.promise;

  const parts: string[] = [];
  const pages = Math.min(doc.numPages, 400);
  for (let p = 1; p <= pages; p++) {
    const page = await doc.getPage(p);
    const content = await page.getTextContent();
    let line = "";
    for (const item of content.items) {
      if (!("str" in item)) continue;
      line += item.str;
      // pdf.js marks the end of a visual line. Without this every page is one
      // run-on string and sentence splitting fails entirely.
      if (item.hasEOL) { parts.push(line); line = ""; } else { line += " "; }
    }
    if (line.trim()) parts.push(line);
    parts.push("");
    page.cleanup();
  }
  await task.destroy();

  const text = parts.join("\n");
  const note = doc.numPages > pages
    ? `first ${pages} of ${doc.numPages} pages`
    : undefined;
  if (text.replace(/\s/g, "").length < 40) {
    throw new IngestError(`${file.name} has no text layer.`,
      "It's probably a scan — run OCR on it first, then add the result.");
  }
  return { text, note };
}

/**
 * DOCX extraction without a Word library.
 *
 * A .docx is a zip whose `word/document.xml` holds the text in `<w:t>` elements.
 * Unzipping with fflate (~8 KB) and pulling those out costs a fraction of a full
 * OOXML parser, and only the prose is needed here.
 */
async function extractDocx(file: File): Promise<string> {
  const { unzipSync } = await import("fflate");
  const zip = unzipSync(new Uint8Array(await file.arrayBuffer()));
  const entry = zip["word/document.xml"];
  if (!entry) throw new IngestError(`${file.name} isn't a readable .docx.`);
  const xml = new TextDecoder().decode(entry);
  return xml
    // Paragraph and line breaks must become real breaks before tags are stripped,
    // or the whole document collapses into one sentence.
    .replace(/<\/w:p>/g, "\n")
    .replace(/<w:br\b[^>]*\/?>/g, "\n")
    .replace(/<w:tab\b[^>]*\/?>/g, " ")
    .replace(/<[^>]+>/g, "")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");
}

/**
 * HTML to text.
 *
 * Parsed with DOMParser into an inert document: no script execution, no
 * subresource loading, no contact with the live DOM. Source HTML is untrusted.
 */
export function extractHtml(html: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.querySelectorAll("script,style,noscript,svg,nav,footer,header,aside,form,iframe")
    .forEach((el) => el.remove());
  // Block elements need explicit breaks. innerText would supply them but only
  // works on rendered nodes, and this document is never attached.
  doc.querySelectorAll("p,div,li,tr,br,h1,h2,h3,h4,h5,h6,section,article")
    .forEach((el) => el.append("\n"));
  const main = doc.querySelector("article,main,[role=main]") ?? doc.body;
  return main?.textContent ?? "";
}

/**
 * JSON/JSONL: flatten to labelled leaves.
 *
 * The key is what makes a scalar retrievable. A bare `400` is unfindable, since
 * no question embeds near a lone number, where `battery_hours: 400` answers "how
 * many battery hours does it have". Keys are carried for every value type,
 * numeric ones included.
 *
 * Nested paths stay dotted for the same reason: `specs.weight_g: 99` says what
 * the 99 is, and the dotted form tokenises into separate words.
 */
function extractJson(raw: string): string {
  const out: string[] = [];

  const walk = (v: unknown, path: string, depth: number): void => {
    if (depth > 12) return;
    if (v === null || v === undefined) return;

    if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
      const val = String(v);
      if (!path && val.length <= 2) return;      // a bare scalar with no label
      out.push(path ? `${path}: ${val}` : val);
      return;
    }
    if (Array.isArray(v)) {
      // Array indices are not worth embedding, so every element inherits the
      // array's own label instead.
      for (const x of v) walk(x, path, depth + 1);
      return;
    }
    if (typeof v === "object") {
      for (const [k, x] of Object.entries(v)) {
        walk(x, path ? `${path}.${k}` : k, depth + 1);
      }
    }
  };

  try {
    walk(JSON.parse(raw), "", 0);
  } catch {
    // JSONL: one object per line, and one bad line should not lose the file.
    for (const line of raw.split("\n")) {
      if (!line.trim()) continue;
      try { walk(JSON.parse(line), "", 0); } catch { /* skip */ }
    }
  }
  return out.join("\n");
}

// -- urls -------------------------------------------------------------------

/**
 * URL ingestion goes through the Worker, because the browser cannot fetch an
 * arbitrary cross-origin page and read the body. The Worker fetches, strips the
 * markup and returns text, enforcing scheme, size and redirect limits along the
 * way; see `worker/src`.
 */
export async function ingestUrl(url: string, workerBase: string): Promise<Extracted> {
  const trimmed = url.trim();
  let parsed: URL;
  try {
    parsed = new URL(/^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`);
  } catch {
    throw new IngestError("That doesn't look like a web address.");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new IngestError("Only http and https links can be added.");
  }
  if (!workerBase) {
    throw new IngestError("Adding links needs the Worker.",
      "Set VITE_WORKER_BASE, or paste the page text instead.");
  }

  const res = await fetch(`${workerBase}/fetch-url`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ url: parsed.toString() }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new IngestError(
      urlErrorMessage((detail as { error?: string }).error, res.status),
      parsed.hostname,
    );
  }
  const data = await res.json() as { title?: string; text?: string; contentType?: string };
  const norm = normalise(data.text ?? "");
  if (norm.length < 20) {
    throw new IngestError(`Nothing readable at ${parsed.hostname}.`,
      "The page may render its text with JavaScript. Paste it instead.");
  }
  const { text, truncated } = clamp(norm);
  return {
    title: data.title?.trim() || parsed.hostname + parsed.pathname,
    text,
    note: truncated ? `truncated to ${MAX_CHARS.toLocaleString()} characters` : parsed.hostname,
  };
}

function urlErrorMessage(code: string | undefined, status: number): string {
  switch (code) {
    case "blocked_host":    return "That address points at a private network — refused.";
    case "too_large":       return "That page is too large to add.";
    case "unsupported_type":return "That link isn't a web page or text file.";
    case "upstream_status": return "The site refused the request.";
    case "timeout":         return "The site took too long to respond.";
    default:                return `Couldn't fetch that link (${status}).`;
  }
}
