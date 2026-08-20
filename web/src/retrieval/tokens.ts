/**
 * Content tokens — the words in a string that carry its subject.
 *
 * Gates 2 and 3 both compare a query or an answer against a retrieved passage by
 * counting shared words, and a short natural question is mostly function words
 * where a passage of prose is not:
 *
 *     "what is the name"   vs   "name: srinidhi bhat, age: 45, sex: M"
 *
 * Four query tokens, one shared — overlap 0.25, under gate 2's rescue floor,
 * while the bare query "srinidhi" scores 1.0 against the same passage. The
 * dilution is entirely `what`, `is` and `the`, and it worsens the more politely
 * the question is phrased.
 *
 * Scripts with no list here are not filtered, following the same rule as the
 * input gate: never judge text in a script you have no markers for. A partial
 * stoplist would strip the wrong words and shift that language's scores against
 * a threshold fitted on English and Hindi. Adding a script is deliberate.
 */

/**
 * English function words. Deliberately short — this is not an IR stoplist and is
 * not trying to improve ranking. Words like "name", "number" or "date" stay in,
 * because in a question about a document they are often the whole subject.
 *
 * Covers determiners, copulas, auxiliaries, pronouns, prepositions and the
 * interrogatives: the words a question contains regardless of its subject.
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
 * the one non-Latin script where leaving function words in measurably moves a
 * calibrated number. Marathi is included alongside Hindi for the same reason as
 * in the input gate: the script does not tell them apart.
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

/**
 * Split on anything that is not part of a word, in any script.
 *
 * `\p{M}` is not optional. Indic vowel signs, virama, nukta and anusvara are
 * combining marks rather than letters, so a `\p{L}`-only class cuts every word
 * at its first matra: `भारत` becomes `र` + `त`, and `निगम क्या है` reduces to
 * the single token `गम`. Every Indic script is affected, and the damage is
 * silent — both sides of an overlap comparison are mangled the same way, so the
 * score stays plausible while measuring the wrong thing. ZWNJ and ZWJ are kept
 * for the same reason: Devanagari uses them to control conjunct formation.
 */
const WORD_BREAK = /[^\p{L}\p{N}\p{M}‌‍\s]/gu;

function rawTokens(s: string): string[] {
  return s
    .toLowerCase()
    .replace(WORD_BREAK, " ")
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
 * query that is entirely function words ("what is the") names no subject, and
 * an empty set would score every overlap at 0 — indistinguishable from an
 * unrelated passage.
 */
export function contentTokens(s: string): Set<string> {
  const raw = rawTokens(s);
  const content = raw.filter((w) => !isStopword(w));
  return new Set(content.length ? content : raw);
}

/**
 * Which script a string is mostly written in.
 *
 * Used to decide whether comparing two strings word-by-word means anything.
 * Returns a coarse label — "latin", "deva", "beng", … — because the question is
 * not which language, only whether the two can share a vocabulary.
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
 * Digits, normalised across numeral systems. "45" and "४५" are the same fact,
 * and a grounding check that treats them as different rejects a correct
 * translated answer for using the reader's numerals.
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
