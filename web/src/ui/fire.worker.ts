/**
 * Renders the fireball, off the main thread.
 *
 * This worker exists for one reason and it is not throughput: every frame the
 * fire draws is a frame the thread running queries does not have to spend. The
 * whole project's claim is a 200 ms ceiling measured on the main thread, so an
 * idle animation sharing that thread would be putting its own frames inside the
 * number it exists to advertise.
 *
 * `requestAnimationFrame` is available in a dedicated worker whenever
 * `OffscreenCanvas` is, so the loop is still vsync-locked — this is not a
 * `setTimeout` approximation of one.
 */

import { FireRenderer, type FireInput } from "./fire";

export type FireMessage =
  | { type: "init"; canvas: OffscreenCanvas; w: number; h: number; dpr: number; light: boolean }
  | { type: "resize"; w: number; h: number; dpr: number }
  | { type: "state"; input: Partial<FireInput> }
  | { type: "light"; on: boolean }
  | { type: "visibility"; visible: boolean };

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
      // nobody is looking at is pure battery. Stop outright.
      case "visibility": m.visible ? fire?.start() : fire?.stop(); break;
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
