/**
 * Three guardrail gates.
 *
 * Requirement 6 asks the system to know when *not* to answer. That is three
 * separate questions, so it is three separate gates:
 *
 *   Gate 1 (input)      should we process this at all?
 *   Gate 2 (retrieval)  did we actually find anything good enough?
 *   Gate 3 (grounding)  is every claim in the answer traceable to a source?
 *
 * All three run inside the latency budget, so all three are lexical/statistical
 * rather than model-based. An LLM-based safety check would cost more than the
 * entire budget and would itself need a fallback for when it times out.
 *
 * Gate 2 is the important one. It is also the only one we can calibrate
 * honestly, because the corpus ships 3,012 queries with no relevant passage --
 * real cases where the correct action is refusal. The threshold is fitted on
 * that split rather than guessed. See `bench/calibrate.ts`.
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
  // Not a guardrail decision about the query — a precondition. Distinct from
  // LOW_CONFIDENCE on purpose: "I searched and found nothing good enough" and
  // "there was nothing to search" are different facts, and collapsing them
  // tells a new user their question was bad when the app is simply empty.
  | "NO_SOURCES";

export interface GateResult {
  pass: boolean;
  reason?: Refusal;
  /** Shown to the user. Explains the refusal in their language, not ours. */
  message?: string;
  detail?: Record<string, number | string>;
}

const PASS: GateResult = { pass: true };

// ---------------------------------------------------------------------------
// Gate 1 — input validation
// ---------------------------------------------------------------------------

/**
 * Instruction-override attempts, EN + HI.
 *
 * Note this is a *voice* system: an attacker has to say these out loud and get
 * them through speech-to-text, which mangles unusual phrasing. So the patterns
 * target semantic shapes that survive transcription, not exact strings.
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
 * We block requests for *operational harm instructions*, not subject matter.
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
 * This exists because the not-a-question rule below is only meaningful in a
 * language whose question words we can actually recognise. The first version of
 * this file had Latin and Devanagari and nothing else, which meant a Tamil
 * question of more than eight words was refused as "not a question" — a
 * confident judgement about a sentence the gate could not read at all.
 *
 * Keyed by script rather than by language because that is what is detectable
 * from the string. Devanagari therefore has to serve Hindi, Marathi, Nepali and
 * Sanskrit at once, so it carries all four sets of question words (काय and
 * कुठे are Marathi, not Hindi, and a Hindi-only list refuses Marathi).
 *
 * A script with no entry here is not judged — see `gateInput`.
 */
const SCRIPT_QUESTION_HINTS: Array<{ script: RegExp; hints: RegExp }> = [
  // Devanagari — Hindi, Marathi, Nepali, Sanskrit
  { script: /[ऀ-ॿ]/,
    hints: /क्या|काय|कौन|कोण|कब|कधी|केव्हा|कहाँ|कहां|कुठे|क्यों|का\b|कैसे|कसे|कितन|किती|किस|कोणत|बताओ|बताएं|सांगा|नाम|कस्तो|कहिले|कहाँबाट/ },
  // Bengali / Assamese
  { script: /[ঀ-৿]/,
    hints: /কি|কী|কে|কোন|কখন|কেতিয়া|কোথায়|ক'ত|কেন|কিয়|কীভাবে|কিভাবে|কেনেকৈ|কত|কার/ },
  // Gurmukhi — Punjabi
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
  // Arabic script — Urdu
  { script: /[؀-ۿ]/,
    hints: /کیا|کون|کب|کہاں|کیوں|کیسے|کونسا|کتنا|بتائیں|بتاؤ/ },
];

/** Shannon entropy over characters — catches mashed keys and STT noise. */
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
    return { pass: false, reason: "EMPTY", message: "I didn't catch anything — could you say that again?" };
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
  // Any Unicode letter, in any script.
  //
  // This was `[a-zA-Z ऀ-ॿ]` — Latin and Devanagari only — which meant that in a
  // system advertising fourteen Indian languages, nine of them were refused as
  // "gibberish" here, in 0.1 ms, before a single vector was computed. Bengali,
  // Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia and Urdu all
  // failed identically and silently, and the failure was invisible because a
  // fast confident refusal looks exactly like a working guardrail.
  //
  // `\p{L}` is the general fix rather than a longer list of ranges: the next
  // script added to the corpus should not require editing this line.
  if (!/\p{L}/u.test(q)) {
    return { pass: false, reason: "GIBBERISH", message: "I couldn't find any words in that. Could you try again?" };
  }

  // Commands aimed at the system, as opposed to requests for information.
  //
  // This used to be the inverse test — "more than eight words and no
  // interrogative marker" — and measurement killed it. Across 3,000 parallel
  // queries in fifteen languages it refused 98, and those 98 retrieved the gold
  // passage at hit@5 35.7%, which is the *average* for this corpus: they were
  // ordinary questions. The sample says why in one line:
  //
  //     "Explain what a bone scan is and what it is used for."  ->  NOT_A_QUESTION
  //
  // The information-seeking imperative — explain, list, define, tell me — is
  // one of the most common shapes a real query takes, and absence-of-a-question-
  // word cannot tell it apart from a command. So the test is now positive: look
  // for a verb that asks the system to *act on something*, which is the thing
  // actually worth refusing.
  //
  // The interrogative sets above are what keep "how do I delete a file" out of
  // this: a directive that also carries a question marker is a question about
  // the action, not a demand to perform it.
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
 * Mid-sentence occurrences are usually the subject matter — "what happens when
 * you delete a partition" is a question about deleting, not a deletion.
 */
const ACTION_IMPERATIVE =
  /^\s*(please\s+|kindly\s+|now\s+|just\s+)*(go\s+ahead\s+and\s+|make\s+sure\s+(you|to)\s+)?(delete|remove|drop|erase|wipe|destroy|purge|truncate|shut\s*down|reboot|restart|execute|run|install|uninstall|deploy|send|email|post|publish|tweet|transfer|wire|pay|buy|order|disable|enable|grant|revoke|overwrite|format|kill|terminate)\b/i;

function hasInterrogative(q: string): boolean {
  if (QUESTION_HINTS.test(q)) return true;
  const script = SCRIPT_QUESTION_HINTS.find((s) => s.script.test(q));
  return !!script && script.hints.test(q);
}

// ---------------------------------------------------------------------------
// Gate 2 — retrieval confidence
// ---------------------------------------------------------------------------

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
   * Mid-band rescue. A hit scoring below `minTopScore` but at or above
   * `rescueMinScore`, whose winning passage shares at least `rescueMinOverlap`
   * of the query's content words, is answered rather than refused.
   *
   * This exists because `minTopScore` is an *absolute* cosine, and absolute
   * cosine does not transfer between corpora. It was fitted on MS MARCO, whose
   * passages are search results — written densely, on topic, and selected
   * because they answered a query. A user's policy document or a page of prose
   * is not built that way, and its genuinely correct passage lands where MS
   * MARCO's mediocre ones do. Measured on 28 answerable and 15 unanswerable
   * questions over two English documents, the single threshold refused 25% of
   * answerable questions — including one the document answered word for word.
   *
   * The mid-band is separable, just not by score. Every false positive in it
   * had weak word overlap with the passage that won (0.0–0.4): the embedder
   * found the right *subject* and the wrong *question* — "how much does a
   * kettle cost" against a history of kettles. Every true positive had strong
   * overlap (0.5–1.0). So overlap is what carries the decision there, and the
   * floor is what stops it reaching down into the genuinely-unrelated band,
   * where an unanswerable question can still share half its words with a
   * passage on the same topic.
   *
   * Applied to user-added sources only. `rag.ts` switches it off when the
   * winning passage came from the shipped corpus, because that is the corpus
   * `minTopScore` was fitted on and there is no mismatch there to correct.
   *
   * Set `rescueMinScore` to `Infinity` to switch the rescue off entirely.
   */
  rescueMinScore: number;
  rescueMinOverlap: number;
}

/**
 * Defaults are placeholders. `bench/calibrate.ts` fits these against the 3,012
 * unanswerable queries and writes the fitted values back — a guessed threshold
 * is exactly the kind of unfalsifiable claim this gate exists to prevent.
 */
export const DEFAULT_THRESHOLDS: ConfidenceThresholds = {
  minTopScore: 0.80,
  minAgreement: 2,
  minLexicalOverlap: 0.0,
  /**
   * Rescue floor, fitted on `bench/usersource.ts` — 39 labelled questions over
   * a field-structured document and a prose one.
   *
   * It was 0.40, and that refused nearly half of everything a user document
   * could actually answer:
   *
   *     rule            coverage        abstention
   *     no rescue       40.0% (10/25)   100.0% (14/14)
   *     floor 0.40      52.0% (13/25)   100.0% (14/14)   <- shipped
   *     floor 0.30      76.0% (19/25)    92.9% (13/14)
   *     floor 0.20      88.0% (22/25)    92.9% (13/14)   <- now
   *     floor 0.15      88.0% (22/25)    92.9% (13/14)
   *     floor 0.00      88.0% (22/25)    92.9% (13/14)
   *
   * 0.20 is chosen as the *highest* value on the coverage plateau rather than
   * the lowest: everything from 0 to 0.20 scores identically, so the floor
   * costs nothing there and still excludes the genuinely-unrelated band, where
   * "what is the capital of Peru" scored 0.123 against this pair of documents.
   * Picking the bottom of a plateau leaves a threshold doing no work at all.
   *
   * The one unanswerable question this now answers is "what is your VAT
   * number" against a policy document that says "email support with your order
   * number" — a real near-miss, and one the layer below catches: the model is
   * instructed to return INSUFFICIENT_CONTEXT when the passages do not answer
   * the question, and gate 3 checks what it produces. Three answerable
   * questions are still refused ("what did he study", "which company did he
   * work at", "where did he go to college"): pure semantic matches sharing no
   * content word with the document, whose cosines (0.26–0.33) sit inside the
   * unanswerable range (0.17–0.46). No threshold on these signals separates
   * them, and refusing is the safe side of a case that cannot be called.
   *
   * `strategyAgreement` is deliberately not part of the rescue. It looked like
   * the strongest scale-free signal available until it was measured: on user
   * documents it is 4 for 37 of 39 questions, answerable and unanswerable
   * alike, because a small document is fully covered by every chunking
   * strategy. It discriminates on a 99k-passage corpus and not here.
   *
   * Applied to user-added sources only — `rag.ts` switches it off for corpus
   * hits, which is the split `minTopScore` was fitted on.
   */
  rescueMinScore: 0.20,
  rescueMinOverlap: 0.5,
};

export function gateRetrieval(
  s: RetrievalSignals,
  t: ConfidenceThresholds = DEFAULT_THRESHOLDS,
): GateResult {
  // Two independent signals agreeing is worth more than one clearing a bar
  // that was calibrated on a different corpus. See `rescueMinScore`.
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

// ---------------------------------------------------------------------------
// Gate 3 — grounding
// ---------------------------------------------------------------------------

/**
 * Fraction of the answer's content tokens that appear in the retrieved context.
 *
 * The extractive path is grounded by construction (it returns spans), so this
 * exists for the generated path, where a fluent model can quietly add a fact
 * that was never in the sources.
 *
 * "Content tokens" is now literally true. It counted every token, including
 * function words, which penalised exactly the thing generation is for: turning
 * a passage fragment into a sentence. "Your name is Srinidhi Bhat" against a
 * passage reading `name: srinidhi bhat, age: 45` scored 3/5 = 0.60 — under the
 * 0.62 threshold — because `your` and `is` are not in the passage and never
 * could be. Every correctly-grounded rewrite was being discarded for the
 * grammar it added. See `retrieval/tokens.ts`.
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
 * The specifics in a piece of text: the tokens a model would have to *invent*
 * rather than merely phrase.
 *
 * A bag-of-words grounding score cannot tell "You are 45" from "You are 46" —
 * they differ in one token out of three, which no threshold separates from the
 * ordinary variation of writing a sentence. But that one token is the entire
 * difference between a correct answer and a fabricated one, and the tokens like
 * it are identifiable without any threshold at all: numbers, and names.
 *
 *   digits        an age, a date, an amount, a duration, a version
 *   capitals      a person, a company, a place, a product
 *
 * Sentence-initial capitals are excluded, because "Your" in "Your name is…" is
 * capitalised by grammar rather than by being a name. Scripts without letter
 * case contribute numbers only, which is the honest limit of this test there —
 * it is a check that fires when it is certain, not one that guesses in scripts
 * it cannot read.
 */
function specifics(text: string): Set<string> {
  const out = new Set<string>();
  for (const sentence of text.split(/(?<=[।.!?])\s+/)) {
    const words = sentence.trim().split(/\s+/).filter(Boolean);
    words.forEach((w, i) => {
      const bare = w.replace(/[^\p{L}\p{N}]/gu, "");
      if (bare.length < 2) return;
      if (/\p{N}/u.test(bare)) { out.add(bare.toLowerCase()); return; }
      // Not the first word of the sentence, and starts with a capital.
      if (i > 0 && bare[0] !== bare[0].toLowerCase()) out.add(bare.toLowerCase());
    });
  }
  return out;
}

/**
 * Gate 3 — is the answer traceable to the passages it was written from?
 *
 * Two tests, doing two different jobs.
 *
 * THE HARD TEST is on specifics: every number and every name in the answer must
 * appear in the retrieved text. There is no threshold here and there should not
 * be one — a date the document does not contain is not 90% grounded, it is
 * invented. This is the test that actually catches hallucination, and it is
 * exact rather than statistical.
 *
 * THE SOFT TEST is a coverage floor on content words, and it is a backstop for
 * wholesale topic drift — an answer written from the model's own knowledge
 * using only generic vocabulary. It is set low *on purpose*.
 *
 * It used to be the only test, over every token including function words, at
 * 0.62. That is a check on how much of the answer's grammar happens to appear
 * in a document, and it rejected correct answers systematically:
 *
 *     "Your name is Srinidhi Bhat."   0.60 against `name: srinidhi bhat, …`
 *     "You are 45 years old."         0.33 — "years" and "old" are not in a
 *                                     document that writes `age: 45`
 *
 * Both are perfectly grounded. The first was rejected by two hundredths, and
 * the second by a mile, for the sole offence of being written as English. A
 * gate that fires on the difference between data and a sentence is a gate
 * against the entire purpose of generating one.
 */
export function gateGrounding(
  answer: string,
  contexts: string[],
  minCoverage = 0.34,
): GateResult {
  const joined = contexts.join(" ");

  /**
   * CROSS-LINGUAL ANSWERS CANNOT BE WORD-CHECKED, AND MUST NOT BE REFUSED FOR IT.
   *
   * This corpus is MS MARCO-XI: the passages are Hindi and the questions arrive
   * in fourteen languages. Cross-lingual retrieval is the capability the dataset
   * exists to exercise. But an English answer written from a Hindi passage
   * shares no tokens with it at all — measured, this exact pair scored 0.000:
   *
   *   passage  निगमन एक नए निगम का गठन है (एक निगम एक कानूनी इकाई है …)
   *   answer   "A corporation is a legal entity recognized as a person under law."
   *
   * Correct, faithful, and rejected — so every English question over this corpus
   * silently fell back to showing raw Hindi. Word overlap between two scripts
   * measures translation, not grounding, and a threshold on it is a threshold on
   * the wrong quantity.
   *
   * This follows the rule the input gate already obeys: never judge text in a
   * script you have no markers for — skip the test rather than default to
   * refuse. What survives translation is *numbers*, so the digit half of the
   * specifics check still runs, across numeral systems. Names usually
   * transliterate rather than translate, so they are not checked here; claiming
   * to verify them would be pretending to a measurement we cannot make.
   */
  const crossScript =
    dominantScript(answer) !== dominantScript(joined) &&
    dominantScript(answer) !== "unknown" &&
    dominantScript(joined) !== "unknown";

  const ctxRaw = new Set<string>();
  for (const w of normaliseDigits(joined).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ").split(/\s+/)) {
    if (w) ctxRaw.add(w);
  }

  const found = specifics(normaliseDigits(answer));
  const checked = crossScript
    ? [...found].filter((w) => /\d/.test(w))     // numbers only — they translate
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
    // Passed on what is checkable. Reported rather than hidden: this answer
    // carries less verification than a same-script one, and the difference is
    // a fact about the answer.
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
