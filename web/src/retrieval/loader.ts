/**
 * Index loading + IndexedDB caching.
 *
 * The index is ~40 MB of binary blobs. Downloading that on every visit would be
 * rude; re-parsing it on every visit would be slow. So blobs are cached as raw
 * ArrayBuffers in IndexedDB, keyed by the manifest's build timestamp, and typed
 * arrays are created as *views* over those buffers — no copy, no parse.
 *
 * None of this is in the 200ms path. Load happens once at startup; the budget is
 * measured per query against a warm index. That distinction is stated explicitly
 * in the latency report rather than glossed over.
 */

import { IvfIndex, type IvfIndexData, PassageRescorer } from "./ivf";

export interface Manifest {
  lang: string;
  dim: number;
  codeWords: number;
  numPassages: number;
  strategies: Record<string, { count: number; nlist: number; avgListSize: number }>;
  /** Passage text is sharded: Cloudflare Pages rejects files over 25 MiB. */
  passageShards: string[];
  model: string;
  builtAt: string;
}

const DB_NAME = "voicerag-index";
const STORE = "blobs";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet(db: IDBDatabase, key: string): Promise<ArrayBuffer | undefined> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly").objectStore(STORE).get(key);
    tx.onsuccess = () => resolve(tx.result);
    tx.onerror = () => reject(tx.error);
  });
}

async function idbPut(db: IDBDatabase, key: string, val: ArrayBuffer): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite").objectStore(STORE).put(val, key);
    tx.onsuccess = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export type ProgressFn = (loaded: number, total: number, label: string) => void;

/**
 * Exact byte size of the binary blobs, derived from the manifest.
 *
 * Worth computing rather than guessing, because this is what the loading lamp
 * fills against. Every array's length is fully determined by counts already in
 * the manifest, so the only estimate in the whole figure is the passage text —
 * and that is called out as an estimate rather than folded in silently.
 */
export function expectedBytes(manifest: Manifest): { blobs: number; text: number } {
  const { dim, codeWords, numPassages } = manifest;
  const rawDim = 384;                     // e5-small, the model the PCA was fitted on
  let blobs =
    rawDim * 4 +                          // pca_mean.f32
    dim * rawDim * 4 +                    // pca_comp.f32
    numPassages * dim +                   // passage_int8
    numPassages * 4;                      // passage_scale.f32
  for (const s of Object.values(manifest.strategies)) {
    blobs +=
      s.nlist * dim * 4 +                 // centroids.f32
      (s.nlist + 1) * 4 +                 // offsets.u32
      s.count * codeWords * 4 +           // codes.u32
      s.count * 4;                        // parents.i32
  }
  // Passage text is JSON and its size is not derivable from counts. ~790 bytes
  // per passage is what this corpus measures; it only affects the progress bar.
  return { blobs, text: numPassages * 790 };
}

async function fetchCached(
  db: IDBDatabase | null,
  base: string,
  path: string,
  version: string,
  onProgress?: ProgressFn,
): Promise<ArrayBuffer> {
  const key = `${version}:${path}`;
  if (db) {
    try {
      const hit = await idbGet(db, key);
      if (hit) { onProgress?.(hit.byteLength, hit.byteLength, `${path} (cached)`); return hit; }
    } catch { /* cache miss is not an error */ }
  }
  // `?v=<builtAt>` is what makes `Cache-Control: immutable` safe on these files.
  // Their paths are not content-hashed, so without the query a rebuilt index
  // would keep serving year-old bytes from the HTTP cache against a new
  // manifest — vectors and text silently out of step, no error anywhere.
  const res = await fetch(`${base}/${path}?v=${encodeURIComponent(version)}`);
  if (!res.ok) throw new Error(`failed to fetch ${path}: ${res.status}`);
  const buf = await res.arrayBuffer();
  onProgress?.(buf.byteLength, buf.byteLength, path);
  if (db) { try { await idbPut(db, key, buf); } catch { /* quota — proceed uncached */ } }
  return buf;
}

export interface LoadedIndex {
  manifest: Manifest;
  indices: Map<string, IvfIndex>;
  rescorer: PassageRescorer;
  passages: string[];
  pcaMean: Float32Array;
  pcaComp: Float32Array;   // dim * 384, row-major
  /** Project a raw 384-d embedding into the index's reduced space. */
  projectQuery: (raw: Float32Array, out: Float32Array) => void;
}

/**
 * Assemble an index from already-fetched buffers.
 *
 * Transport-agnostic on purpose: the browser reaches these bytes over HTTP with
 * an IndexedDB cache in front, and the benchmark reads them from disk. Both then
 * run THIS function, so the benchmark exercises the deployed code path rather
 * than a re-implementation of it that could quietly diverge.
 */
export function assembleIndex(
  manifest: Manifest,
  buf: Record<string, ArrayBuffer>,
  passages: string[],
): LoadedIndex {
  const pcaMean = new Float32Array(buf["pca_mean.f32.bin"]);
  const pcaComp = new Float32Array(buf["pca_comp.f32.bin"]);
  const dim = manifest.dim;
  const rawDim = pcaMean.length;

  const rescorer = new PassageRescorer(
    new Int8Array(buf["passage_int8.bin"]),
    new Float32Array(buf["passage_scale.f32.bin"]),
    dim,
  );

  const indices = new Map<string, IvfIndex>();
  for (const [name, info] of Object.entries(manifest.strategies)) {
    const data: IvfIndexData = {
      dim,
      codeWords: manifest.codeWords,
      nlist: info.nlist,
      count: info.count,
      centroids: new Float32Array(buf[`${name}.centroids.f32.bin`]),
      listOffsets: new Uint32Array(buf[`${name}.offsets.u32.bin`]),
      codes: new Uint32Array(buf[`${name}.codes.u32.bin`]),
      slotParent: new Int32Array(buf[`${name}.parents.i32.bin`]),
    };
    indices.set(name, new IvfIndex(data, manifest.numPassages));
  }

  // Must mirror pipeline/src/build_index.py::project exactly — centre, project,
  // re-L2-normalise. A mismatch is silent: retrieval just gets quietly worse.
  const projectQuery = (raw: Float32Array, out: Float32Array): void => {
    let norm = 0;
    for (let i = 0; i < dim; i++) {
      let s = 0;
      const off = i * rawDim;
      for (let j = 0; j < rawDim; j++) s += pcaComp[off + j] * (raw[j] - pcaMean[j]);
      out[i] = s;
      norm += s * s;
    }
    norm = Math.sqrt(norm) || 1e-9;
    for (let i = 0; i < dim; i++) out[i] /= norm;
  };

  return { manifest, indices, rescorer, passages, pcaMean, pcaComp, projectQuery };
}

/** Every binary blob the index needs, given a manifest. */
export function indexBlobNames(manifest: Manifest): string[] {
  const names = ["pca_mean.f32.bin", "pca_comp.f32.bin",
                 "passage_int8.bin", "passage_scale.f32.bin"];
  for (const s of Object.keys(manifest.strategies)) {
    names.push(`${s}.centroids.f32.bin`, `${s}.offsets.u32.bin`,
               `${s}.codes.u32.bin`, `${s}.parents.i32.bin`);
  }
  return names;
}

export async function loadIndex(base: string, onProgress?: ProgressFn): Promise<LoadedIndex> {
  // The one file that must always be fresh: it carries the version every other
  // URL is busted with, so a stale manifest pins a stale index forever.
  const manifest: Manifest = await (await fetch(`${base}/manifest.json`, {
    cache: "no-cache",
  })).json();
  const v = manifest.builtAt;

  let db: IDBDatabase | null = null;
  try { db = await openDb(); } catch { /* private mode — run uncached */ }

  const names = indexBlobNames(manifest);
  const shards = manifest.passageShards ?? ["passages.json"];

  // Cumulative progress against the real expected total, so the lamp fills at a
  // rate that means something instead of jumping per file.
  const expect = expectedBytes(manifest);
  const total = expect.blobs + expect.text;
  let loaded = 0;
  const step = (bytes: number, label: string) => {
    loaded += bytes;
    onProgress?.(Math.min(loaded, total), total, label);
  };

  const [buffers, shardArrays] = await Promise.all([
    Promise.all(names.map((n) =>
      fetchCached(db, base, n, v, (b, _t, label) => step(b, label)))),
    // Shards are fetched in parallel and concatenated in order. Ordinals are
    // assigned by position, so order is load-bearing — do not sort or race.
    //
    // `r.json()` rather than `r.text()` + `JSON.parse`: the text form would hold
    // a 74 MB string alongside the parsed array at peak, and on a 4 GB phone
    // that headroom is not free. The cost is that the exact transferred size is
    // no longer observable, so progress is charged per shard against the
    // estimate — which is what a progress bar needs anyway.
    Promise.all(shards.map(async (s) => {
      const r = await fetch(`${base}/${s}?v=${encodeURIComponent(v)}`);
      if (!r.ok) throw new Error(`failed to fetch ${s}: ${r.status}`);
      const arr = await r.json() as string[];
      step(expect.text / shards.length, s);
      return arr;
    })),
  ]);

  const passages: string[] = [];
  for (const arr of shardArrays) for (const s of arr) passages.push(s);
  if (passages.length !== manifest.numPassages) {
    throw new Error(`passage count mismatch: got ${passages.length}, manifest says ${manifest.numPassages}`);
  }

  const buf: Record<string, ArrayBuffer> = {};
  names.forEach((n, i) => { buf[n] = buffers[i]; });
  return assembleIndex(manifest, buf, passages);
}
