/**
 * Renders the fireball, off the main thread.
 *
 * Not for throughput: every frame the fire draws is a frame the thread running
 * queries does not spend. The 200 ms ceiling is measured on the main thread, so
 * an idle animation sharing it would land inside the number it advertises.
 *
 * `requestAnimationFrame` is available in a dedicated worker wherever
 * `OffscreenCanvas` is, so the loop stays vsync-locked rather than being a
 * `setTimeout` approximation of one.
 */

import { FireRenderer, type FireInput } from "./fire";

export type FireMessage =
  | { type: "init"; canvas: OffscreenCanvas; w: number; h: number; dpr: number; light: boolean }
  | { type: "resize"; w: number; h: number; dpr: number }
  | { type: "state"; input: Partial<FireInput> }
  | { type: "light"; on: boolean }
  | { type: "visibility"; visible: boolean }
  | { type: "freeze"; on: boolean };

let fire: FireRenderer | null = null;

self.onmessage = (e: MessageEvent<FireMessage>) => {
  const m = e.data;
  try {
    switch (m.type) {
      case "init": {
        fire = new FireRenderer(m.canvas);
        fire.setLight(m.light);
        fire.resize(m.w, m.h, m.dpr);
        fire.start();
        self.postMessage({ type: "ready" });
        break;
      }
      case "resize":     fire?.resize(m.w, m.h, m.dpr); break;
      case "state":      fire?.set(m.input); break;
      case "light":      fire?.setLight(m.on); break;
      // A hidden tab still gets rAF in some configurations, and a fireball
      // nobody is looking at is pure battery.
      case "visibility": m.visible ? fire?.start() : fire?.stop(); break;
      // Held for the duration of a measured query. Off the main thread the
      // fire cannot take time from the pipeline directly, but it still competes
      // for a core and the GPU, which on a two-core machine lands inside the
      // budget. Kept separate from `visibility` so the two cannot cancel out.
      case "freeze":     m.on ? fire?.stop() : fire?.start(); break;
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
