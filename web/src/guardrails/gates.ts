/**
 * Three guardrail gates:
 *
 *   Gate 1 (input)      should this be processed at all?
 *   Gate 2 (retrieval)  was anything good enough found?
 *   Gate 3 (grounding)  is every claim in the answer traceable to a source?
 *
 * All three run inside the latency budget, so all three are lexical or
 * statistical rather than model-based. Gate 2's threshold is fitted on the
 * corpus's 3,012 labelled unanswerable queries; see `bench/calibrate.ts`.
 */

import { contentTokens, dominantScript, normaliseDigits } from "../retrieval/tokens";

export type Refusal =
  | "EMPTY"
  | "TOO_SHORT"
  | "GIBBERISH"
  | "UNSAFE"
  | "INJECTION"
  | "NOT_A_QUESTION"
  | "LOW_CONFIDENCE"
  | "NO_AGREEMENT"
  | "UNGROUNDED"
  /**
   * The answer cited an excerpt that was never supplied.
   *
   * Distinct from UNGROUNDED, and worth distinguishing: an ungrounded answer
   * asserts something the sources do not support, while this one asserts
   * where it came from and is wrong about that. A model that invents excerpt
   * 7 out of five has stopped reading and started composing, and its prose can
   * still overlap the real passages well enough to pass a coverage test.
   */
  | "FABRICATED_CITATION"
  // A precondition rather than a judgement about the query. Kept distinct from
  // LOW_CONFIDENCE: "nothing matched" and "nothing was searched" are different
  // facts and need different messages.
  | "NO_SOURCES";

export interface GateResult {
  pass: boolean;
  reason?: Refusal;
  /** Shown to the user. Explains the refusal in their language, not ours. */
  message?: string;
  detail?: Record<string, number | string>;
}

const PASS: GateResult = { pass: true };

// -- Gate 1 - input validation ----------------------------------------------

/**
 * Instruction-override attempts, EN + HI.
 *
 * Patterns target semantic shapes rather than exact strings: input arrives via
 * speech-to-text, which mangles unusual phrasing.
 */
const INJECTION_PATTERNS: RegExp[] = [
  /\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\b/i,
  /\bdisregard\s+(all\s+|the\s+)?(previous|prior|above|instructions?)\b/i,
  /\b(system|developer)\s+(prompt|message|instruction)/i,
  /\byou\s+are\s+now\b|\bfrom\s+now\s+on\s+you\b|\bact\s+as\s+(a|an|if)\b/i,
  /\b(reveal|show|print|repeat|output)\s+(me\s+)?(your|the)\s+(prompt|instructions?|rules?)\b/i,
  /\bdeveloper\s+mode\b|\bjailbreak\b|\bDAN\s+mode\b/i,
  /पिछले?\s+निर्देश|पहले\s+के\s+निर्देश/,          // "previous instructions"
  /निर्देश\s*(को)?\s*(अनदेखा|नज़रअंदाज़|भूल)/,      // "ignore/forget instructions"
  /तुम\s+अब\s+से|आप\s+अब\s+से/,                  // "you from now on"
  /अपने\s+(निर्देश|नियम)\s*(दिखाओ|बताओ)/,        // "show your instructions"
];

/**
 * Deliberately narrow. This is a general-knowledge QA corpus, so broad topic
 * filtering would refuse legitimate questions about medicine, history and law.
 * Blocks requests for operational harm instructions, not subject matter.
 */
const UNSAFE_PATTERNS: RegExp[] = [
  /\bhow\s+(to|do\s+i|can\s+i)\s+(make|build|synthesi[sz]e|obtain)\s+.{0,30}\b(bomb|explosive|nerve\s+agent|ricin|meth(amphetamine)?|napalm)\b/i,
  /\bhow\s+(to|do\s+i|can\s+i)\s+.{0,25}\b(kill|murder|poison)\s+(a\s+)?(person|someone|him|her|them|my)\b/i,
  /\b(kill|hang|cut)\s+(myself|my\s?self)\b|\bcommit\s+suicide\b|\bend\s+my\s+life\b/i,
  /\bchild\s+(porn|sexual|abuse\s+material)\b|\bCSAM\b/i,
  /\bhow\s+to\s+(hack|breach|ddos)\s+.{0,20}\b(bank|hospital|government|grid)\b/i,
  /आत्महत्या\s*(कैसे|करने)/,                        // "how to commit suicide"
  /(बम|विस्फोटक)\s*(कैसे\s*)?(बनाएं|बनाना|बनाऊं)/,   // "how to make a bomb"
];

/** Self-harm gets a support response, never a generic refusal. */
const SELF_HARM = /\b(kill\s+myself|commit\s+suicide|end\s+my\s+life|want\s+to\s+die)\b|आत्महत्या/i;

const QUESTION_HINTS =
  /\?|^(what|who|when|where|why|how|which|is|are|was|were|does|do|did|can|could|should|will|would|list|name|define|tell)\b/i;

/**
 * Interrogative markers, one set per script.
 *
 * Keyed by script rather than by language, because script is what is detectable
 * from the string. Devanagari therefore serves Hindi, Marathi, Nepali and
 * Sanskrit at once and carries all four sets of question words (काय and कुठे
 * are Marathi, not Hindi).
 *
 * A script with no entry here is not judged - see `gateInput`.
 */
const SCRIPT_QUESTION_HINTS: Array<{ script: RegExp; hints: RegExp }> = [
  // Devanagari - Hindi, Marathi, Nepali, Sanskrit
  { script: /[ऀ-ॿ]/,
    hints: /क्या|काय|कौन|कोण|कब|कधी|केव्हा|कहाँ|कहां|कुठे|क्यों|का\b|कैसे|कसे|कितन|किती|किस|कोणत|बताओ|बताएं|सांगा|नाम|कस्तो|कहिले|कहाँबाट/ },
  // Bengali / Assamese
  { script: /[ঀ-৿]/,
    hints: /কি|কী|কে|কোন|কখন|কেতিয়া|কোথায়|ক'ত|কেন|কিয়|কীভাবে|কিভাবে|কেনেকৈ|কত|কার/ },
  // Gurmukhi - Punjabi
  { script: /[਀-੿]/,
    hints: /ਕੀ|ਕੌਣ|ਕਦੋਂ|ਕਿੱਥੇ|ਕਿਥੇ|ਕਿਉਂ|ਕਿਵੇਂ|ਕਿਹੜ|ਕਿੰਨ|ਦੱਸੋ/ },
  // Gujarati
  { script: /[઀-૿]/,
    hints: /શું|કોણ|ક્યારે|ક્યાં|કેમ|શા\s*માટે|કેવી\s*રીતે|કયો|કઈ|કેટલ|જણાવો/ },
  // Odia
  { script: /[଀-୿]/,
    hints: /କ'?ଣ|କିଏ|କେବେ|କେଉଁ|କାହିଁକି|କିପରି|କେତେ|କହନ୍ତୁ/ },
  // Tamil
  { script: /[஀-௿]/,
    hints: /என்ன|யார்|எப்போது|எங்கே|எங்கு|ஏன்|எப்படி|எந்த|எத்தனை|எவ்வளவு|கூறு/ },
  // Telugu
  { script: /[ఀ-౿]/,
    hints: /ఏమిటి|ఏమి|ఎవరు|ఎప్పుడు|ఎక్కడ|ఎందుకు|ఎలా|ఏది|ఏ\s|ఎంత|చెప్ప/ },
  // Kannada
  { script: /[ಀ-೿]/,
    hints: /ಏನು|ಯಾರು|ಯಾವಾಗ|ಎಲ್ಲಿ|ಏಕೆ|ಹೇಗೆ|ಯಾವ|ಎಷ್ಟು|ಹೇಳಿ/ },
  // Malayalam
  { script: /[ഀ-ൿ]/,
    hints: /എന്ത|ആര|എപ്പോൾ|എവിടെ|എന്തുകൊണ്ട|എങ്ങനെ|ഏത|എത്ര|പറയ/ },
  // Arabic script - Urdu
  { script: /[؀-ۿ]/,
    hints: /کیا|کون|کب|کہاں|کیوں|کیسے|کونسا|کتنا|بتائیں|بتاؤ/ },
];

/** Shannon entropy over characters - catches mashed keys and STT noise. */
function charEntropy(s: string): number {
  const f = new Map<string, number>();
  for (const c of s) f.set(c, (f.get(c) ?? 0) + 1);
  let h = 0;
  for (const n of f.values()) {
    const p = n / s.length;
    h -= p * Math.log2(p);
  }
  return h;
}

export function gateInput(raw: string): GateResult {
  const q = raw.trim();

  if (!q) {
    return { pass: false, reason: "EMPTY", message: "I didn't catch anything - could you say that again?" };
  }
  if (q.length < 3) {
    return { pass: false, reason: "TOO_SHORT", message: "That was too short for me to work with. Could you ask the full question?" };
  }

  if (SELF_HARM.test(q)) {
    return {
      pass: false,
      reason: "UNSAFE",
      message:
        "I can't help with this, but you don't have to handle it alone. " +
        "In India you can reach Tele-MANAS free, 24/7, on 14416. " +
        "If you're somewhere else, your local emergency number can connect you to someone right now.",
    };
  }
  for (const re of UNSAFE_PATTERNS) {
    if (re.test(q)) {
      return { pass: false, reason: "UNSAFE", message: "I can't help with that one. Ask me something else and I'll do my best." };
    }
  }
  for (const re of INJECTION_PATTERNS) {
    if (re.test(q)) {
      return {
        pass: false,
        reason: "INJECTION",
        message: "That reads like an attempt to change my instructions rather than a question. I only answer from the documents I've been given.",
      };
    }
  }

  // Gibberish: low entropy (repeated characters) or no letters at all.
  const compact = q.replace(/\s+/g, "");
  if (compact.length >= 6 && charEntropy(compact) < 1.8) {
    return { pass: false, reason: "GIBBERISH", message: "That didn't come through clearly. Could you try again?" };
  }
  // Any Unicode letter, in any script. `\p{L}` rather than a list of ranges, so
  // adding a script to the corpus does not require editing this line.
  if (!/\p{L}/u.test(q)) {
    return { pass: false, reason: "GIBBERISH", message: "I couldn't find any words in that. Could you try again?" };
  }

  // Commands aimed at the system, as opposed to requests for information.
  //
  // Tested positively - an imperative verb must be present - rather than by the
  // absence of a question word. Information-seeking imperatives ("explain what a
  // bone scan is") carry no interrogative but are ordinary questions. The
  // interrogative sets above keep "how do I delete a file" out: a directive that
  // also carries a question marker asks about the action, it does not demand it.
  if (ACTION_IMPERATIVE.test(q) && q.split(/\s+/).length > 6 && !hasInterrogative(q)) {
    return {
      pass: false,
      reason: "NOT_A_QUESTION",
      message: "That reads like an instruction to carry something out rather than a question. What would you like to know?",
    };
  }

  return PASS;
}

/**
 * Verbs that ask for an action on the world, not for information.
 *
 * Anchored to the start of the query (after an optional politeness or
 * "go ahead and" preamble) because that is where an imperative's verb goes.
 * Mid-sentence occurrences are usually the subject matter - "what happens when
 * you delete a partition" is a question about deleting, not a deletion.
 */
const ACTION_IMPERATIVE =
  /^\s*(please\s+|kindly\s+|now\s+|just\s+)*(go\s+ahead\s+and\s+|make\s+sure\s+(you|to)\s+)?(delete|remove|drop|erase|wipe|destroy|purge|truncate|shut\s*down|reboot|restart|execute|run|install|uninstall|deploy|send|email|post|publish|tweet|transfer|wire|pay|buy|order|disable|enable|grant|revoke|overwrite|format|kill|terminate)\b/i;

function hasInterrogative(q: string): boolean {
  if (QUESTION_HINTS.test(q)) return true;
  const script = SCRIPT_QUESTION_HINTS.find((s) => s.script.test(q));
  return !!script && script.hints.test(q);
}

// -- Gate 2 - retrieval confidence ------------------------------------------

export interface RetrievalSignals {
  /** Best fused score. */
  topScore: number;
  /** Gap between best and runner-up. A flat distribution means "nothing stood out". */
  margin: number;
  /** How many distinct chunking strategies surfaced the winning passage. */
  strategyAgreement: number;
  /** Lexical overlap between query and winning passage, 0..1. */
  lexicalOverlap: number;
}

export interface ConfidenceThresholds {
  minTopScore: number;
  minAgreement: number;
  minLexicalOverlap: number;
  /**
   * Mid-band rescue: a hit below `minTopScore` but at or above
   * `rescueMinScore`, sharing at least `rescueMinOverlap` of the query's
   * content words, is answered rather than refused.
   *
   * `minTopScore` is an absolute cosine fitted on MS MARCO, whose passages are
   * dense on-topic search results. A user's prose scores lower, so its correct
   * passages land in the mid-band. Score does not separate that band; overlap
   * does. False positives there match the subject and not the question ("how
   * much does a kettle cost" against a history of kettles) at 0.0-0.4, true
   * positives at 0.5-1.0.
   *
   * User sources only. `rag.ts` switches it off for corpus hits, which is the
   * split `minTopScore` was fitted on. Infinity disables it.
   */
  rescueMinScore: number;
  rescueMinOverlap: number;
}

/**
 * Conservative placeholders for the case where calibration has never been run.
 * `bench/calibrate.ts` fits these against the 3,012 unanswerable queries and
 * writes the result to `public/thresholds.json`.
 */
export const DEFAULT_THRESHOLDS: ConfidenceThresholds = {
  minTopScore: 0.80,
  minAgreement: 2,
  minLexicalOverlap: 0.0,
  /**
   * Rescue floor, fitted by `bench/usersource.ts` over 39 labelled questions
   * against a field-structured document and a prose one.
   *
   * Coverage is flat from 0 to 0.20 and falls off above it, so 0.20 is the
   * highest value that costs nothing. It still excludes the genuinely unrelated
   * band, where an off-topic question scores around 0.12.
   *
   * `strategyAgreement` is deliberately not part of the rescue: on a small
   * document every chunking strategy covers everything, so it reads 4 for
   * answerable and unanswerable questions alike.
   */
  rescueMinScore: 0.20,
  rescueMinOverlap: 0.5,
};

export function gateRetrieval(
  s: RetrievalSignals,
  t: ConfidenceThresholds = DEFAULT_THRESHOLDS,
): GateResult {
  // See `rescueMinScore`: two weak signals agreeing beat one bar calibrated on
  // a different corpus.
  const rescued =
    s.topScore >= t.rescueMinScore && s.lexicalOverlap >= t.rescueMinOverlap;

  if (s.topScore < t.minTopScore && !rescued) {
    return {
      pass: false,
      reason: "LOW_CONFIDENCE",
      message: "I couldn't find anything in my sources that answers that. I'd rather say so than guess.",
      detail: { topScore: +s.topScore.toFixed(4), threshold: t.minTopScore },
    };
  }
  // A passage found by only one strategy, with no lexical support, is the
  // classic shape of a spurious dense-retrieval hit.
  if (s.strategyAgreement < t.minAgreement && s.lexicalOverlap < 0.15) {
    return {
      pass: false,
      reason: "NO_AGREEMENT",
      message: "I found something loosely related but nothing I'd stand behind as an answer.",
      detail: { agreement: s.strategyAgreement, lexicalOverlap: +s.lexicalOverlap.toFixed(3) },
    };
  }
  return PASS;
}

// -- Gate 3 - grounding -----------------------------------------------------

/**
 * Fraction of the answer's content tokens that appear in the retrieved context.
 *
 * The extractive path is grounded by construction (it returns spans), so this
 * exists for the generated path, where a fluent model can add a fact that was
 * never in the sources.
 *
 * Content tokens only. Counting function words would penalise the grammar a
 * generated sentence has to add over the passage fragment it came from; see
 * `retrieval/tokens.ts`.
 */
export function groundingScore(answer: string, contexts: string[]): number {
  const a = contentTokens(answer);
  if (!a.size) return 0;
  const ctx = new Set<string>();
  for (const c of contexts) for (const w of contentTokens(c)) ctx.add(w);
  let hit = 0;
  for (const w of a) if (ctx.has(w)) hit++;
  return hit / a.size;
}

/**
 * The specifics in a piece of text: tokens a model would have to invent rather
 * than merely rephrase.
 *
 * A bag-of-words score cannot separate "You are 45" from "You are 46", one
 * token in three, yet that token is the whole difference between correct and
 * fabricated. Numbers and names find it without a threshold:
 *
 *   digits    an age, a date, an amount, a duration, a version
 *   capitals  a person, a company, a place, a product
 *
 * Sentence-initial capitals are excluded, since "Your" in "Your name is..." is
 * capitalised by grammar. Scripts without case contribute numbers only.
 */
function specifics(text: string): Set<string> {
  const out = new Set<string>();
  for (const sentence of text.split(/(?<=[।.!?])\s+/)) {
    const words = sentence.trim().split(/\s+/).filter(Boolean);
    words.forEach((w, i) => {
      // `\p{M}` keeps Indic matras attached to their base letter; see tokens.ts.
      const bare = w.replace(/[^\p{L}\p{N}\p{M}]/gu, "");
      if (bare.length < 2) return;
      if (/\p{N}/u.test(bare)) { out.add(bare.toLowerCase()); return; }
      // Not the first word of the sentence, and starts with a capital.
      if (i > 0 && bare[0] !== bare[0].toLowerCase()) out.add(bare.toLowerCase());
    });
  }
  return out;
}

/**
 * Gate 3 - is the answer traceable to the passages it was written from?
 *
 * Two tests doing two different jobs. The hard test is on specifics: every
 * number and every name in the answer must appear in the retrieved text, with
 * no threshold, because a date the document does not contain is not 90%
 * grounded. That is the test that catches hallucination.
 *
 * The soft test is a coverage floor on content words, a backstop against
 * wholesale topic drift - an answer written from the model's own knowledge in
 * generic vocabulary. It is set low deliberately: a generated sentence
 * legitimately adds grammar that no source passage contains.
 */
export function gateGrounding(
  answer: string,
  contexts: string[],
  minCoverage = 0.34,
): GateResult {
  const joined = contexts.join(" ");

  /**
   * A cross-script answer cannot be word-checked, so it is not.
   *
   * The corpus is Hindi and questions arrive in fourteen languages, so an
   * English answer written from a Hindi passage is the normal case, not an edge
   * one - and it shares no tokens with its source at all. Word overlap between
   * two scripts measures translation rather than grounding.
   *
   * Numbers survive translation, so the digit half of the specifics check still
   * runs across numeral systems. Names transliterate rather than translate and
   * are not checked here.
   */
  const crossScript =
    dominantScript(answer) !== dominantScript(joined) &&
    dominantScript(answer) !== "unknown" &&
    dominantScript(joined) !== "unknown";

  const ctxRaw = new Set<string>();
  const flat = normaliseDigits(joined).toLowerCase().replace(/[^\p{L}\p{N}\p{M}\s]/gu, " ");
  for (const w of flat.split(/\s+/)) {
    if (w) ctxRaw.add(w);
  }

  const found = specifics(normaliseDigits(answer));
  const checked = crossScript
    ? [...found].filter((w) => /\d/.test(w))     // numbers only - they translate
    : [...found];
  const invented = checked.filter((w) => !ctxRaw.has(w));

  if (invented.length) {
    return {
      pass: false,
      reason: "UNGROUNDED",
      message: "I drafted an answer but couldn't trace all of it back to my sources, so I'm not going to give it to you.",
      detail: { invented: invented.slice(0, 4).join(", ") },
    };
  }

  if (crossScript) {
    // Passed on what is checkable, and flagged: this answer carries less
    // verification than a same-script one.
    return { pass: true, detail: { crossScript: 1 } };
  }

  const score = groundingScore(answer, contexts);
  if (score < minCoverage) {
    return {
      pass: false,
      reason: "UNGROUNDED",
      message: "I drafted an answer but couldn't trace all of it back to my sources, so I'm not going to give it to you.",
      detail: { groundingScore: +score.toFixed(3), threshold: minCoverage },
    };
  }
  return PASS;
}

/**
 * Gate 3a - do the answer's citations refer to excerpts that exist?
 *
 * The model answers by calling a tool and naming the excerpts it used, so this
 * is checkable rather than inferred. It runs before the grounding check because
 * it decides which passages grounding is measured against: an answer that
 * cites excerpt 7 of 5 cannot be checked at all, and passing it on the strength
 * of the excerpts it did not claim to use would be checking the wrong thing.
 *
 * An empty citation list is not a failure. Some providers ignore the field, and
 * the caller then falls back to checking against every excerpt supplied - a
 * weaker test, but the honest one when the model has not said.
 */
export function gateCitations(cited: number[], available: number): GateResult {
  const bad = cited.filter((n) => !Number.isInteger(n) || n < 1 || n > available);
  if (bad.length) {
    return {
      pass: false,
      reason: "FABRICATED_CITATION",
      message: "I drafted an answer but it pointed at a source that doesn't exist, so I'm not going to give it to you.",
      detail: { cited: bad.slice(0, 4).join(", "), available },
    };
  }
  return PASS;
}

/** Per-sentence grounding, so a partly-supported answer can be trimmed rather than dropped. */
export function filterUngroundedSentences(
  answer: string,
  contexts: string[],
  minScore = 0.55,
): { kept: string; dropped: string[] } {
  const sents = answer.split(/(?<=[।.!?])\s+/).filter(Boolean);
  const kept: string[] = [];
  const dropped: string[] = [];
  for (const s of sents) {
    (groundingScore(s, contexts) >= minScore ? kept : dropped).push(s);
  }
  return { kept: kept.join(" "), dropped };
}
