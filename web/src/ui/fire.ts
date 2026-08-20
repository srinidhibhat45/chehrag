/**
 * The fireball.
 *
 * A fragment shader rather than a radial gradient with a blur, because what
 * makes fire legible is structure — turbulent filaments that curl, rise and burn
 * out — and a gradient has no structure at any scale. Three things do the work:
 *
 *   1. Turbulence, not smooth noise. `sum |noise|` over successive octaves
 *      leaves creases where the noise crosses zero, and those creases read as
 *      flame filaments. Ordinary fbm reads as cloud.
 *
 *   2. Domain warping. The turbulence is sampled at a position itself displaced
 *      by noise, which makes filaments curl and shear rather than scroll past.
 *
 *   3. A blackbody-ish ramp. Real flame runs deep red -> orange -> amber ->
 *      white over a narrow temperature range, so brightness and hue move
 *      together; a fixed hue with varying alpha reads as cellophane.
 *
 * The sphere is a density field rather than lit geometry. Inside the radius the
 * field is sampled on the surface of an implicit sphere, so convection cells
 * wrap around it and rotate with it; outside, the same field continues on the
 * equator plane, so the corona is the same fire escaping rather than a glow
 * sprite pasted around the edge.
 *
 * This module carries no DOM or worker specifics, so the same code runs on both
 * sides of the `OffscreenCanvas` boundary: in a worker where the browser allows
 * it (see `fire.worker.ts`), in-thread and paused during measured queries where
 * it does not.
 */

export type FireState =
  | "dormant"     // index still loading
  | "idle"        // ready, waiting
  | "listening"   // microphone open
  | "thinking"    // query in flight
  | "speaking"    // TTS playing back
  | "answered"    // flare
  | "refused";    // cool to embers

/** Everything the renderer needs to know, in one flat message-safe object. */
export interface FireInput {
  state: FireState;
  /** 0..1 — index load progress; drives the charge ring while dormant. */
  charge: number;
  /** 0..1 — microphone amplitude while listening, playback envelope while speaking. */
  amp: number;
}

const VERT = `#version 300 es
in vec2 p;
void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;

/**
 * One full-screen pass. Everything is procedural — no textures, so there is
 * nothing to upload and the first frame is as cheap as the thousandth.
 */
const FRAG = `#version 300 es
precision highp float;
out vec4 fragColor;

uniform vec2  uRes;
uniform float uTime;
uniform float uAmp;      // 0..1 voice / playback envelope
uniform float uCharge;   // 0..1 load progress (ring)
uniform float uHeat;     // 0..1 how hard it burns
uniform float uCool;     // 0..1 embers (refusal)
uniform float uFlare;    // 0..1 momentary burst (answered)
uniform float uSpin;     // accumulated rotation, radians
uniform float uRing;     // 0..1 how visible the charge ring is
uniform float uLight;    // 1.0 when the page is in light mode

const float PI = 3.14159265;

/* ---- gradient noise ---------------------------------------------------- */
/* sin-hash: not the highest quality, but it needs no texture upload and the
   turbulence below hides its lattice artefacts. */
vec3 hash33(vec3 p){
  p = vec3(dot(p, vec3(127.1, 311.7,  74.7)),
           dot(p, vec3(269.5, 183.3, 246.1)),
           dot(p, vec3(113.5, 271.9, 124.6)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float gnoise(vec3 p){
  vec3 i = floor(p), f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(dot(hash33(i + vec3(0,0,0)), f - vec3(0,0,0)),
                     dot(hash33(i + vec3(1,0,0)), f - vec3(1,0,0)), u.x),
                 mix(dot(hash33(i + vec3(0,1,0)), f - vec3(0,1,0)),
                     dot(hash33(i + vec3(1,1,0)), f - vec3(1,1,0)), u.x), u.y),
             mix(mix(dot(hash33(i + vec3(0,0,1)), f - vec3(0,0,1)),
                     dot(hash33(i + vec3(1,0,1)), f - vec3(1,0,1)), u.x),
                 mix(dot(hash33(i + vec3(0,1,1)), f - vec3(0,1,1)),
                     dot(hash33(i + vec3(1,1,1)), f - vec3(1,1,1)), u.x), u.y), u.z);
}

/* Smooth fbm — used only for the warp field, where creases would be wrong. */
float fbm(vec3 p){
  float a = 0.5, s = 0.0;
  for(int i = 0; i < 4; i++){ s += a * gnoise(p); p *= 2.03; a *= 0.5; }
  return s;
}

/* |noise| leaves a crease at every zero crossing, and the creases are the
   flame filaments. */
float turb(vec3 p){
  float a = 0.5, s = 0.0;
  for(int i = 0; i < 5; i++){ s += a * abs(gnoise(p)); p *= 2.07; a *= 0.5; }
  return s;
}

/* ---- blackbody-ish ramp ------------------------------------------------
   Four overlapping smoothsteps added together rather than mixed between stops.
   The overlaps give the continuous red->orange->amber->white climb, with
   brightness rising alongside hue as it does in a real flame. */
vec3 fire(float t){
  t = clamp(t, 0.0, 1.0);
  vec3 c = vec3(0.0);
  c += vec3(0.42, 0.045, 0.010) * smoothstep(0.00, 0.30, t);   // dull ember
  c += vec3(0.62, 0.190, 0.015) * smoothstep(0.20, 0.52, t);   // red-orange
  c += vec3(0.48, 0.360, 0.060) * smoothstep(0.42, 0.74, t);   // amber
  c += vec3(0.34, 0.340, 0.330) * smoothstep(0.68, 1.00, t);   // white core
  return c;
}

void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / min(uRes.x, uRes.y);
  float r  = length(uv);

  /* Breathes with the voice and swells slightly on a flare. Kept small: large
     size changes read as a balloon. */
  float R = 0.245 * (1.0 + 0.070 * uAmp + 0.045 * uFlare + 0.020 * sin(uTime * 0.7));

  float t  = uTime;
  float rn = r / R;

  /* Implicit sphere. Inside, z lifts the sample point onto the surface so the
     convection cells wrap; outside, z is 0 and the same field simply continues
     outward, which is why the corona looks like this fire and not a decal. */
  float z  = sqrt(max(1.0 - rn * rn, 0.0));
  vec3  sp = vec3(uv / R, z);

  /* Spin about the vertical axis, tilted so the poles never sit dead centre;
     a sphere rotating exactly about the view axis looks flat. */
  float cs = cos(uSpin), sn = sin(uSpin);
  vec3  q  = vec3(sp.x * cs + sp.z * sn, sp.y, -sp.x * sn + sp.z * cs);
  q.yz = vec2(q.y * 0.966 - q.z * 0.259, q.y * 0.259 + q.z * 0.966);

  /* Buoyancy: the field drifts downward through the sample point, so features
     travel *up* the ball the way hot gas does. */
  vec3 base = q * 2.25 + vec3(0.0, -t * 0.34, t * 0.06);

  /* Sampled coarser than the detail it displaces; at the same scale it adds
     noise instead of shearing the filaments. */
  vec3 w = vec3(fbm(base * 0.55 + 13.1),
                fbm(base * 0.55 + 41.7),
                fbm(base * 0.55 + 77.3));

  float d  = turb(base + w * 1.45);
  float d2 = turb(base * 2.9 + w * 0.9 + vec3(0.0, -t * 0.9, 0.0));  // fine wisps

  /* Radial energy: a hot body falling off fast, plus a corona that reaches
     further and carries most of the visible movement.

     Written as 1.0 - smoothstep(lo, hi, x) rather than smoothstep(hi, lo, x).
     The reversed-edge form works on most drivers, but GLSL leaves it undefined
     when edge0 >= edge1. */
  float body  = 1.0 - smoothstep(0.30, 1.12, rn);
  float halo  = exp(-max(rn - 0.90, 0.0) * 4.2);
  float lick  = exp(-max(rn - 0.95, 0.0) * 2.1) * (0.30 + 0.55 * uAmp);

  /* Where fire is allowed to exist at all.

     Load-bearing rather than tidying. The turbulence term is a signed field with
     a positive mean, so ungated it lifts every pixel on the canvas slightly
     above zero — a faint haze to the corners. The haze is invisible alone, but
     the alpha feather cuts it at a fixed radius and the cut reads as a hard
     circular outline. This makes the fire absent rather than clipped. */
  float env = clamp(body + halo * 0.55 + lick * 0.6, 0.0, 1.0);

  /* Temperature. The subtraction carves the flame out of the noise; without a
     threshold every pixel glows a little and the result is fog. */
  float temp = body * (1.05 + 0.45 * uHeat)
             + halo * 0.42
             + (d - 0.62) * (1.15 + 0.35 * uHeat) * env
             + (d2 - 0.55) * 0.42 * lick
             + uFlare * 0.55 * body;

  /* Pull the top of the ramp down so a refusal cools to red without the shape
     changing: the same fire, banked. */
  temp *= (1.0 - 0.55 * uCool);
  temp  = temp * (0.55 + 0.65 * uHeat);

  /* Light mode raises the floor before anything is coloured.

     On a dark ground the faint outskirts of the field read as glow. On a pale
     one they are darker than the paper, which reads as smoke: the sign of the
     contrast is wrong, and no choice of hue fixes it. A semi-transparent dark
     pixel over cream also converges on grey, since the background contributes
     to all three channels while the flame only adds red.

     So dim material is not drawn at all in light mode and what survives is
     scaled back up, leaving the flame smaller and denser on a pale ground. */
  temp = max(temp - uLight * 0.34, 0.0) * mix(1.0, 1.55, uLight);

  float density = clamp(temp, 0.0, 1.4);
  vec3  col     = fire(density * (1.0 - 0.30 * uCool));

  /* White-hot centre. Squared falloff, so it stays a point rather than a wash. */
  float core = (1.0 - smoothstep(0.0, 0.62, rn)) * (0.55 + 0.9 * uHeat) * (1.0 - 0.7 * uCool);
  col += vec3(0.55, 0.40, 0.26) * core * core;

  /* With env gating the turbulence there is nothing to clip at the rim, so this
     is a dissolve rather than a cutoff. */
  float alpha = clamp(density * 1.35, 0.0, 1.0);
  alpha *= 1.0 - smoothstep(0.9, 2.4, rn);

  /* ---- charge ring ------------------------------------------------------
     Drawn here rather than in CSS, so the loading state costs the main thread
     nothing either. Sweeps clockwise from 12 o'clock. */
  if (uRing > 0.005) {
    float rr  = R * 1.72;
    float band = 1.0 - smoothstep(0.0015, 0.0055, abs(r - rr));
    float a01  = fract((atan(uv.x, uv.y)) / (2.0 * PI) + 1.0);
    float trackA = band * 0.10 * uRing;
    float fillA  = band * step(a01, uCharge) * 0.95 * uRing;
    col   += vec3(0.55, 0.34, 0.12) * (trackA + fillA * 1.6);
    alpha  = max(alpha, trackA + fillA);
  }

  /* In light mode the faint corona has to thin rather than darken. Scaling
     colour down at low density puts a dim warm grey over cream, which reads as
     smoke. Dropping alpha lets the weak edges dissolve into the paper instead,
     and the colour is pushed toward red to keep contrast on a bright ground. */
  float thin = 1.0 - 0.94 * (1.0 - smoothstep(0.04, 0.50, density));
  alpha *= mix(1.0, thin, uLight);
  col   *= mix(vec3(1.0), vec3(1.06, 0.84, 0.62), uLight);

  /* Premultiplied. With blendFunc(ONE, ONE_MINUS_SRC_ALPHA) this glows over a
     dark ground without blowing out over a light one. */
  fragColor = vec4(col * alpha, alpha);
}`;

interface Uniforms {
  uRes: WebGLUniformLocation | null;
  uTime: WebGLUniformLocation | null;
  uAmp: WebGLUniformLocation | null;
  uCharge: WebGLUniformLocation | null;
  uHeat: WebGLUniformLocation | null;
  uCool: WebGLUniformLocation | null;
  uFlare: WebGLUniformLocation | null;
  uSpin: WebGLUniformLocation | null;
  uRing: WebGLUniformLocation | null;
  uLight: WebGLUniformLocation | null;
}

/** Per-state targets the renderer eases toward. Nothing here snaps. */
const TARGETS: Record<FireState, { heat: number; cool: number; spin: number }> = {
  dormant:   { heat: 0.16, cool: 0.55, spin: 0.10 },
  idle:      { heat: 0.44, cool: 0.00, spin: 0.16 },
  listening: { heat: 0.78, cool: 0.00, spin: 0.30 },
  thinking:  { heat: 0.92, cool: 0.00, spin: 0.85 },
  speaking:  { heat: 0.72, cool: 0.00, spin: 0.26 },
  answered:  { heat: 1.00, cool: 0.00, spin: 0.40 },
  refused:   { heat: 0.30, cool: 0.85, spin: 0.08 },
};

export class FireRenderer {
  private gl: WebGL2RenderingContext;
  private u: Uniforms;
  private raf = 0;
  private running = false;

  private t0 = 0;
  private last = 0;
  private spin = 0;

  // Eased values. Targets arrive as steps; these are what the shader sees.
  private heat = 0.16;
  private cool = 0.55;
  private spinRate = 0.1;
  private amp = 0;
  private charge = 0;
  private ring = 1;
  private flare = 0;

  private state: FireState = "dormant";
  private light = 0;
  private dpr = 1;
  private cssW = 0;
  private cssH = 0;

  constructor(private readonly canvas: HTMLCanvasElement | OffscreenCanvas) {
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,          // full-screen procedural pass — MSAA buys nothing
      depth: false,
      stencil: false,
      powerPreference: "low-power",
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("webgl2 unavailable");
    this.gl = gl;

    const prog = link(gl, VERT, FRAG);
    gl.useProgram(prog);

    // One triangle covering the viewport. A quad needs two, and clips the same
    // pixels twice along the diagonal.
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0, 0, 0, 0);

    const g = (n: string) => gl.getUniformLocation(prog, n);
    this.u = {
      uRes: g("uRes"), uTime: g("uTime"), uAmp: g("uAmp"), uCharge: g("uCharge"),
      uHeat: g("uHeat"), uCool: g("uCool"), uFlare: g("uFlare"), uSpin: g("uSpin"),
      uRing: g("uRing"), uLight: g("uLight"),
    };
  }

  resize(cssW: number, cssH: number, dpr: number): void {
    // Capped at 2x. The field is smooth enough that the difference is not
    // visible, and this is the cheapest lever on a weak GPU.
    this.dpr = Math.min(dpr, 2);
    this.cssW = cssW;
    this.cssH = cssH;
    const w = Math.max(1, Math.round(cssW * this.dpr));
    const h = Math.max(1, Math.round(cssH * this.dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.gl.viewport(0, 0, w, h);
  }

  setLight(on: boolean): void { this.light = on ? 1 : 0; }

  set(input: Partial<FireInput>): void {
    if (input.charge !== undefined) this.charge = clamp01(input.charge);
    if (input.amp !== undefined) this.amp = clamp01(input.amp);
    if (input.state !== undefined && input.state !== this.state) {
      // One-shot: it decays on its own rather than being a state the app has
      // to remember to leave.
      if (input.state === "answered") this.flare = 1;
      this.state = input.state;
    }
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.t0 = now();
    this.last = this.t0;
    const frame = () => {
      if (!this.running) return;
      this.draw();
      this.raf = requestAnimationFrame(frame);
    };
    this.raf = requestAnimationFrame(frame);
  }

  stop(): void {
    this.running = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  /**
   * Advance and draw one frame.
   *
   * Every time-dependent quantity integrates against measured elapsed time
   * rather than a frame count, so irregular frame delivery changes how often the
   * fire is sampled and not where it has got to.
   */
  private draw(): void {
    const t = now();
    const dt = Math.min((t - this.last) / 1000, 0.05);   // clamp: tab wake-ups
    this.last = t;

    const target = TARGETS[this.state];
    this.heat     += (target.heat - this.heat) * (1 - Math.exp(-dt * 6));
    this.cool     += (target.cool - this.cool) * (1 - Math.exp(-dt * 4));
    this.spinRate += (target.spin - this.spinRate) * (1 - Math.exp(-dt * 3));
    this.flare    += (0 - this.flare) * (1 - Math.exp(-dt * 3.2));
    this.spin     += this.spinRate * dt;

    // Only meaningful while loading, so it fades out once charged.
    //
    // Snapped to zero at the bottom of the fade rather than left to asymptote.
    // An exponential approach never reaches its target, and the loop stops
    // outright when the tab is hidden — so a tab switched away mid-fade would
    // freeze on a faint stale ring around a fire that finished loading.
    const ringTarget = this.state === "dormant" && this.charge < 0.999 ? 1 : 0;
    this.ring += (ringTarget - this.ring) * (1 - Math.exp(-dt * 8));
    if (this.ring < 0.02) this.ring = 0;

    const gl = this.gl;
    const { u } = this;
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(u.uRes, this.canvas.width, this.canvas.height);
    gl.uniform1f(u.uTime, (t - this.t0) / 1000);
    gl.uniform1f(u.uAmp, this.amp);
    gl.uniform1f(u.uCharge, this.charge);
    gl.uniform1f(u.uHeat, this.heat);
    gl.uniform1f(u.uCool, this.cool);
    gl.uniform1f(u.uFlare, this.flare);
    gl.uniform1f(u.uSpin, this.spin);
    gl.uniform1f(u.uRing, this.ring);
    gl.uniform1f(u.uLight, this.light);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const compile = (type: number, src: string): WebGLShader => {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(`shader: ${gl.getShaderInfoLog(sh)}`);
    }
    return sh;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`link: ${gl.getProgramInfoLog(p)}`);
  }
  return p;
}

const now = (): number =>
  typeof performance !== "undefined" ? performance.now() : Date.now();

function clamp01(v: number): number { return v < 0 ? 0 : v > 1 ? 1 : v; }
