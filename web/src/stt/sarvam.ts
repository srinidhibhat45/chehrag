/**
 * Sarvam speech-to-text client.
 *
 * Streaming is the primary path, because partial transcripts let retrieval start
 * while the user is still speaking: by the time a sentence ends, the answer for
 * its prefix is already computed.
 *
 * The API key never reaches the browser. The Worker terminates this socket and
 * opens an authenticated one upstream - browsers cannot set custom headers on a
 * WebSocket handshake, so a proxy is required regardless of secrecy.
 *
 * Falls back to batch REST when the socket cannot be established: corporate
 * proxies and some mobile networks kill long-lived WebSockets.
 */

export type SttEvent =
  | { type: "partial"; text: string }
  | { type: "final"; text: string; language?: string }
  | { type: "speech_start" }
  | { type: "speech_end" }
  /** Something changed and the engine is still working. Never a teardown. */
  | { type: "notice"; message: string }
  /** The engine cannot continue. The listener is expected to stop it. */
  | { type: "error"; message: string }
  | { type: "closed" };

export interface SttOptions {
  workerBase: string;
  languageCode?: string;   // "auto" or e.g. "hi-IN"
  onEvent: (e: SttEvent) => void;
}

const TARGET_SAMPLE_RATE = 16000;   // Sarvam accepts 8k or 16k only

/** Float32 [-1,1] -> 16-bit PCM little-endian, which is `linear16`. */
function floatToPcm16(input: Float32Array): ArrayBuffer {
  const out = new DataView(new ArrayBuffer(input.length * 2));
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return out.buffer;
}

function toBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let bin = "";
  // Chunked to avoid blowing the argument limit on large buffers.
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CH));
  }
  return btoa(bin);
}

export class SarvamStt {
  private ws: WebSocket | null = null;
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | ScriptProcessorNode | null = null;
  private stream: MediaStream | null = null;
  private pingTimer: number | null = null;
  private recorder: MediaRecorder | null = null;
  private batchChunks: Blob[] = [];
  private mode: "stream" | "batch" = "stream";

  constructor(private readonly opts: SttOptions) {}

  /** The live microphone stream, once `start()` has resolved. Exposed so the
   *  lamp can pulse with the voice; transcription does not depend on it. */
  get micStream(): MediaStream | null { return this.stream; }

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });

    try {
      await this.startStreaming();
      this.mode = "stream";
    } catch (err) {
      // WebSocket blocked or upstream refused - degrade rather than fail.
      //
      // A `notice`, not an `error`: the microphone is open, batch capture is
      // about to start, and only partial transcripts are lost. An `error` here
      // makes the UI tear the engine down mid-`start()`, destroying the
      // fallback this branch has just set up.
      this.opts.onEvent({
        type: "notice",
        message: `Live transcription unavailable (${err instanceof Error ? err.message : err}) - recording, and transcribing when you stop.`,
      });
      this.startBatch();
      this.mode = "batch";
    }
  }

  private async startStreaming(): Promise<void> {
    const url = new URL(this.opts.workerBase.replace(/^http/, "ws") + "/stt/stream");
    url.searchParams.set("model", "saaras:v3-realtime");
    url.searchParams.set("language_code", this.opts.languageCode ?? "auto");
    url.searchParams.set("sample_rate", String(TARGET_SAMPLE_RATE));
    url.searchParams.set("input_audio_codec", "linear16");

    const ws = new WebSocket(url.toString());
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("socket open timeout")), 6000);
      ws.onopen = () => { clearTimeout(timer); resolve(); };
      ws.onerror = () => { clearTimeout(timer); reject(new Error("socket error")); };
    });

    ws.onmessage = (ev) => {
      let msg: { event?: string; text?: string; language?: string; message?: string; code?: string };
      try { msg = JSON.parse(typeof ev.data === "string" ? ev.data : ""); } catch { return; }
      switch (msg.event) {
        case "transcript.partial":
          if (msg.text) this.opts.onEvent({ type: "partial", text: msg.text });
          break;
        case "transcript.final":
          if (msg.text) this.opts.onEvent({ type: "final", text: msg.text, language: msg.language });
          break;
        case "vad.speech_start": this.opts.onEvent({ type: "speech_start" }); break;
        case "vad.speech_end":   this.opts.onEvent({ type: "speech_end" }); break;
        // Sarvam's own fatal errors (quota exhausted, bad model, rate limit, ...)
        // arrive as a message on the open socket, not as a close code - without
        // this case they were dropped silently and the UI hung on "Listening..."
        // forever once the server closed the connection underneath it.
        case "error":
          this.opts.onEvent({
            type: "error",
            message: msg.message ?? (msg.code ? `Sarvam error: ${msg.code}` : "Sarvam speech-to-text error"),
          });
          break;
      }
    };
    ws.onclose = () => {
      if (this.pingTimer != null) { clearInterval(this.pingTimer); this.pingTimer = null; }
      this.opts.onEvent({ type: "closed" });
    };

    // Sarvam closes idle sockets with 1008; a periodic ping keeps it alive.
    this.pingTimer = window.setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ event: "ping" }));
    }, 15000);

    // AudioContext resamples to 16 kHz. The browser's resampler beats anything
    // worth hand-rolling here.
    const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    this.ctx = ctx;
    const src = ctx.createMediaStreamSource(this.stream!);

    // ScriptProcessor is deprecated but universally available and adequate for
    // capture-only work, where a dropped frame costs a syllable.
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    this.node = proc;
    proc.onaudioprocess = (e) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const pcm = floatToPcm16(e.inputBuffer.getChannelData(0));
      ws.send(JSON.stringify({ event: "audio_input", audio: toBase64(pcm) }));
    };
    src.connect(proc);
    proc.connect(ctx.destination);
  }

  private startBatch(): void {
    // `stop()` may already have run and released the tracks: a listener can
    // call it from inside the notice above. Without this the constructor throws
    // on a null stream and takes `start()` down with it.
    if (!this.stream) return;
    const rec = new MediaRecorder(this.stream, { mimeType: "audio/webm" });
    this.recorder = rec;
    this.batchChunks = [];
    rec.ondataavailable = (e) => { if (e.data.size) this.batchChunks.push(e.data); };
    rec.start();
  }

  /** Stops capture. In batch mode this is when transcription actually happens. */
  async stop(): Promise<void> {
    if (this.pingTimer != null) { clearInterval(this.pingTimer); this.pingTimer = null; }

    if (this.mode === "stream") {
      try { this.ws?.send(JSON.stringify({ event: "speech_end" })); } catch { /* closing */ }
      try { this.ws?.send(JSON.stringify({ event: "end" })); } catch { /* closing */ }
    } else if (this.recorder) {
      const rec = this.recorder;
      const done = new Promise<void>((r) => { rec.onstop = () => r(); });
      rec.stop();
      await done;
      await this.transcribeBatch();
    }

    this.node?.disconnect();
    await this.ctx?.close().catch(() => {});
    this.stream?.getTracks().forEach((t) => t.stop());
    this.node = null; this.ctx = null; this.stream = null;
    setTimeout(() => { try { this.ws?.close(1000); } catch { /* already closed */ } }, 400);
  }

  private async transcribeBatch(): Promise<void> {
    const blob = new Blob(this.batchChunks, { type: "audio/webm" });
    const form = new FormData();
    form.append("file", blob, "audio.webm");
    form.append("model", "saaras:v3");
    if (this.opts.languageCode) form.append("language_code", this.opts.languageCode);
    try {
      const res = await fetch(`${this.opts.workerBase}/stt/batch`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`batch stt ${res.status}`);
      const data = await res.json() as { transcript?: string; language?: string };
      this.opts.onEvent({ type: "final", text: data.transcript ?? "", language: data.language ?? undefined });
    } catch (err) {
      this.opts.onEvent({ type: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }
}
