/**
 * Content tokens — the words in a string that carry its subject.
 *
 * Two separate guardrails in this system compare a query, or an answer, to the
 * text of a retrieved passage by counting shared words. Both of them were
 * counting *every* word, and both were silently broken by the same fact: a
 * short natural question is mostly function words, and a passage of prose is
 * mostly not.
 *
 *     "what is the name"   vs   "name: srinidhi bhat, age: 45, sex: M"
 *
 * Four query tokens, one of which appears in the passage. Overlap 0.25. The
 * mid-band rescue in gate 2 needs 0.5, so the question was refused — while the
 * one-word query "srinidhi" scored 1.0 against the same passage and was
 * answered. The dilution is entirely `what`, `is` and `the`, and it gets worse
 * the more politely the question is phrased. Gate 3 had the identical problem
 * from the other side: "Your name is Srinidhi Bhat." grounds at 3/5 = 0.60
 * against a 0.62 threshold, so a perfectly grounded answer was discarded.
 *
 * Stripping function words fixes both, and it is what the thresholds were
 * always documented as measuring — `rescueMinOverlap` says "content words".
 *
 * SCRIPTS WE HAVE NO LIST FOR ARE NOT FILTERED. This is the same rule the input
 * gate follows: never judge text in a script you have no markers for. A partial
 * Tamil stoplist would strip the wrong words and shift that language's overlap
 * scores against a threshold fitted on English and Hindi; no list at all leaves
 * it exactly where it was, which is the honest default. Adding a script here is
 * a deliberate act, not something to be inferred.
 */

/**
 * English function words.
 *
 * Deliberately short. This is not an IR stoplist — it is not trying to improve
 * ranking, and words like "name", "number", "time" or "date" stay in even
 * though a large stoplist would drop them, because in a question about a
 * document they are frequently the entire subject ("what is the name").
 * Included: determiners, copulas, auxiliaries, pronouns, prepositions and the
 * interrogatives, which are the ones that appear in a question regardless of
 * what it is about.
 */
const EN_STOP = new Set([
  "a", "an", "the", "this", "that", "these", "those",
  "is", "am", "are", "was", "were", "be", "been", "being",
  "do", "does", "did", "doing", "done",
  "have", "has", "had", "having",
  "can", "could", "will", "would", "shall", "should", "may", "might", "must",
  "i", "me", "my", "mine", "myself",
  "you", "your", "yours", "we", "us", "our", "ours",
  "he", "him", "his", "she", "her", "hers", "it", "its",
  "they", "them", "their", "theirs",
  "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
  "as", "into", "onto", "over", "under", "up", "down", "out", "off",
  "and", "or", "but", "if", "then", "than", "so", "because",
  "there", "here", "not", "no", "any", "some", "all", "each", "very",
  "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
  "tell", "give", "show", "please", "kindly",
]);

/**
 * Devanagari function words — Hindi and Marathi.
 *
 * The corpus is Hindi and the retrieval threshold was fitted on it, so this is
 * the one non-Latin script where leaving the function words in measurably moves
 * a calibrated number. Marathi's are included alongside Hindi's for the same
 * reason the input gate carries both: the script does not tell them apart.
 */
const HI_STOP = new Set([
  "का", "के", "की", "को", "में", "से", "पर", "है", "हैं", "था", "थे", "थी",
  "और", "या", "यह", "वह", "ये", "वे", "एक", "कि", "जो", "तो", "ही", "भी",
  "हूँ", "हूं", "हो", "होता", "होती", "होते", "किया", "करना", "करें", "कर",
  "क्या", "कौन", "कब", "कहाँ", "कहां", "क्यों", "कैसे", "कितना", "कितनी", "किस",
  "मेरा", "मेरी", "मेरे", "आपका", "आपकी", "तुम", "आप", "मैं", "हम", "उनका",
  "बताओ", "बताएं", "बताइए",
  // Marathi
  "चा", "ची", "चे", "मध्ये", "आहे", "आहेत", "होता", "आणि", "काय", "कोण",
  "कुठे", "कसे", "किती", "सांगा",
]);

/** Latin letters present. Cheap enough to run on every gate call. */
const HAS_LATIN = /[a-z]/i;
const HAS_DEVANAGARI = /[ऀ-ॿ]/;

/** Split on anything that is not a letter or a digit, in any script. */
function rawTokens(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((w) => w.length > 1);
}

function isStopword(w: string): boolean {
  if (HAS_LATIN.test(w)) return EN_STOP.has(w);
  if (HAS_DEVANAGARI.test(w)) return HI_STOP.has(w);
  return false;                    // script with no list — never filtered
}

/**
 * The content words of a string, lowercased and de-duplicated.
 *
 * Falls back to the unfiltered tokens when filtering would leave nothing. A
 * query that is *entirely* function words ("what is the") has no content to
 * measure overlap on, and returning an empty set would make every overlap
 * score 0 — which reads as "this passage is unrelated" when the truth is "this
 * question named no subject". Keeping the raw tokens leaves such a query
 * exactly where it was before this module existed.
 */
export function contentTokens(s: string): Set<string> {
  const raw = rawTokens(s);
  const content = raw.filter((w) => !isStopword(w));
  return new Set(content.length ? content : raw);
}

/**
 * Which script a string is mostly written in.
 *
 * Used to decide whether comparing two strings word-by-word means anything at
 * all. Returns a coarse label — "latin", "deva", "beng", … — because that is
 * the granularity the decision needs: not *which language*, only *can these two
 * share a vocabulary*.
 */
const SCRIPT_RANGES: Array<[string, RegExp]> = [
  ["latin", /[A-Za-z]/],
  ["deva",  /[\u0900-\u097F]/],
  ["beng",  /[\u0980-\u09FF]/],
  ["guru",  /[\u0A00-\u0A7F]/],
  ["gujr",  /[\u0A80-\u0AFF]/],
  ["orya",  /[\u0B00-\u0B7F]/],
  ["taml",  /[\u0B80-\u0BFF]/],
  ["telu",  /[\u0C00-\u0C7F]/],
  ["knda",  /[\u0C80-\u0CFF]/],
  ["mlym",  /[\u0D00-\u0D7F]/],
  ["arab",  /[\u0600-\u06FF]/],
];

export function dominantScript(text: string): string {
  let best = "unknown", bestN = 0;
  for (const [name, re] of SCRIPT_RANGES) {
    let n = 0;
    for (const ch of text) if (re.test(ch)) n++;
    if (n > bestN) { bestN = n; best = name; }
  }
  return bestN ? best : "unknown";
}

/**
 * Digits, normalised across numeral systems.
 *
 * "45" and "४५" are the same fact written twice, and a grounding check that
 * treats them as different rejects a correct translated answer for the crime of
 * using the reader's numerals.
 */
const DIGIT_BASES = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66, 0x0660, 0x06F0];

export function normaliseDigits(s: string): string {
  return [...s].map((ch) => {
    const c = ch.codePointAt(0)!;
    for (const base of DIGIT_BASES) {
      if (c >= base && c <= base + 9) return String(c - base);
    }
    return ch;
  }).join("");
}
