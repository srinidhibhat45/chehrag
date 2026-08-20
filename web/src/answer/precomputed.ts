/**
 * Answers generated ahead of time for the shipped corpus.
 *
 * Retrieval runs in the browser and finishes in about 4 ms. Writing the answer
 * is a network round trip to a model and takes about 540 ms, so on the shipped
 * corpus — which never changes — that hop is being paid over and over for a
 * question that was already answered once. `bench/precompute.ts` answers the
 * corpus offline and this reads the result back.
 *
 * WHAT THIS DOES NOT CHANGE. Retrieval still runs in full, and the latency
 * figure under each answer still measures it. A hit here replaces the
 * *generation* hop only, so the 200 ms guarantee describes exactly what it did
 * before — and because it is generation that is skipped, the message carries no
 * generation figure at all. It says "written ahead" where a live answer reports
 * its milliseconds, so a stored answer is never mistaken for a fast model.
 *
 * TWO CONDITIONS, BOTH REQUIRED. A stored answer is used only when
 *
 *   1. the query vector is within `MIN_SIMILARITY` of a stored one, and
 *   2. the passage retrieval just ranked first is one the stored answer was
 *      actually written from.
 *
 * The second is what makes this safe. Cosine alone cannot tell "what is a
 * corporation" from "what is a corporation tax", and serving a confident answer
 * to the wrong question is worse than paying for the round trip. Requiring the
 * evidence to agree means a near-miss on the vector still has to land on the
 * same passage, and it also handles the case that breaks every naive answer
 * cache: once the user adds their own sources, retrieval returns different
 * passages and the stored answer is stale by definition.
 */

import { PassageRescorer } from "../retrieval/ivf";

export interface PrecomputedManifest {
  /** `builtAt` of the index these answers were generated against. */
  indexBuiltAt: string;
  /** Reduced dimension, which must match the index's. */
  dim: number;
  count: number;
  model: string;
  builtAt: string;
}

interface Entry {
  /** The question this was written for. Carried so the store is inspectable and
   *  `bench/precomputed.ts` can check it without a separate manifest; the
   *  lookup itself matches on vectors and never reads this. */
  q: string;
  /** The answer text, already verified against its passages at build time. */
  a: string;
  /** Passage ordinals it was written from, best first. */
  p: number[];
}

/**
 * Cosine floor for a match.
 *
 * Deliberately high. The passage guard below carries most of the safety, so
 * this only has to exclude questions that are merely on the same topic — and
 * on this corpus two unrelated questions about corporations sit around 0.85,
 * which is well inside the range a looser floor would admit.
 */
const MIN_SIMILARITY = 0.94;

export interface PrecomputedHit {
  answer: string;
  similarity: number;
}

export class PrecomputedAnswers {
  private rescorer: PassageRescorer;
  private entries: Entry[];
  /** Passage ordinal -> indices of every entry written from it. */
  private byPassage = new Map<number, number[]>();

  constructor(
    private readonly manifest: PrecomputedManifest,
    int8: Int8Array,
    scale: Float32Array,
    entries: Entry[],
  ) {
    this.rescorer = new PassageRescorer(int8, scale, manifest.dim);
    this.entries = entries;
    entries.forEach((e, i) => {
      for (const pid of e.p) {
        const list = this.byPassage.get(pid);
        if (list) list.push(i);
        else this.byPassage.set(pid, [i]);
      }
    });
  }

  get size(): number { return this.entries.length; }
  get model(): string { return this.manifest.model; }
  /** Every stored (question, answer) pair, for inspection and for the bench. */
  get pairs(): ReadonlyArray<{ q: string; a: string }> { return this.entries; }

  /**
   * Look for an answer already written for this question.
   *
   * `qv` is the projected, L2-normalised query vector the pipeline just
   * computed, so a lookup costs one int8 quantisation plus a dot product per
   * candidate. Candidates come from `topPassage` rather than from a scan of
   * every stored vector: the passage guard has to hold anyway, so there is no
   * reason to score entries that cannot pass it.
   */
  lookup(qv: Float32Array, topPassage: number): PrecomputedHit | null {
    const candidates = this.byPassage.get(topPassage);
    if (!candidates?.length) return null;

    this.rescorer.prepare(qv);
    let best = -1;
    let bestSim = -Infinity;
    for (const i of candidates) {
      const sim = this.rescorer.score(i);
      if (sim > bestSim) { bestSim = sim; best = i; }
    }
    if (best < 0 || bestSim < MIN_SIMILARITY) return null;
    return { answer: this.entries[best].a, similarity: bestSim };
  }
}

/**
 * Load the store, or return null if this build ships without one.
 *
 * Absence is a supported state rather than an error: the app falls back to
 * generating live, which is what it did before any of this existed. A store
 * built against a different index is also treated as absent — passage ordinals
 * are positional, so replaying them against a rebuilt index would attach
 * answers to whatever now sits at that offset.
 */
export async function loadPrecomputed(
  base: string,
  indexBuiltAt: string,
  dim: number,
): Promise<PrecomputedAnswers | null> {
  let manifest: PrecomputedManifest;
  try {
    const res = await fetch(`${base}/manifest.json`, { cache: "no-cache" });
    if (!res.ok) return null;
    manifest = await res.json() as PrecomputedManifest;
  } catch {
    return null;
  }

  if (manifest.indexBuiltAt !== indexBuiltAt || manifest.dim !== dim) {
    console.warn(
      `[chehrag] precomputed answers were built against a different index ` +
      `(${manifest.indexBuiltAt} vs ${indexBuiltAt}) — ignoring them and ` +
      `generating live. Re-run bench/precompute.ts to rebuild.`,
    );
    return null;
  }

  try {
    const v = encodeURIComponent(manifest.builtAt);
    const [int8, scale, entries] = await Promise.all([
      fetch(`${base}/answers.q8.bin?v=${v}`).then((r) => r.arrayBuffer()),
      fetch(`${base}/answers.scale.f32.bin?v=${v}`).then((r) => r.arrayBuffer()),
      fetch(`${base}/answers.json?v=${v}`).then((r) => r.json() as Promise<Entry[]>),
    ]);
    if (entries.length !== manifest.count) return null;
    return new PrecomputedAnswers(
      manifest, new Int8Array(int8), new Float32Array(scale), entries);
  } catch {
    return null;
  }
}
