/**
 * Romanised Hindi, folded back into Devanagari before embedding.
 *
 * The corpus is Devanagari and the embedder is multilingual, so a question in
 * either Hindi or English retrieves from it well. Romanised Hindi — "Corporation
 * kya hai" — retrieves badly, and worse than either half would alone:
 *
 *     निगम क्या है              0.7275   answered
 *     what is a corporation     0.5376   answered
 *     Corporation kya hai       0.3846   refused, threshold 0.4788
 *
 * e5's training data is written in the scripts its languages actually use, so
 * `kya` and `hai` are not Hindi to it, and they are not English either. They are
 * unmodelled tokens that pull the vector away from the region the content word
 * points at. The damage grows with the amount of scaffolding: "mujhe corporation
 * ke bare mein bataiye" scores 0.3232 and returns passages about Uber.
 *
 * Mapping the function words back to Devanagari restores the sentence shape the
 * model was trained on, and the English content word survives untouched — e5
 * handles code-mixed `corporation क्या है` (0.5984) far better than romanised
 * Hindi. Only closed-class words are mapped. Transliterating content words needs
 * an open vocabulary and a real transliteration model, so a query whose *subject*
 * is romanised ("nigam kya hai") is still not retrievable; that is a known limit
 * rather than something this file half-solves.
 */

/**
 * Closed-class romanised Hindi, with the spelling variants people actually type.
 * Interrogatives, copulas, postpositions, demonstratives and quantifiers only.
 */
const TRANSLIT: Record<string, string> = {
  // interrogatives
  kya: "क्या", kyaa: "क्या", kia: "क्या",
  kaun: "कौन", kon: "कौन", kaunsa: "कौन सा", konsa: "कौन सा",
  kab: "कब", kahan: "कहाँ", kaha: "कहाँ", kahaan: "कहाँ",
  kyon: "क्यों", kyu: "क्यों", kyun: "क्यों", kyo: "क्यों",
  kaise: "कैसे", kaisa: "कैसा", kaisi: "कैसी", kese: "कैसे",
  kitna: "कितना", kitne: "कितने", kitni: "कितनी",
  kitna_: "कितना", kis: "किस", kaunse: "कौन से",
  matlab: "मतलब", arth: "अर्थ", paribhasha: "परिभाषा",
  // copulas and auxiliaries
  hai: "है", hain: "हैं", h: "है", hota: "होता", hoti: "होती", hote: "होते",
  tha: "था", the: "थे", thi: "थी", hoga: "होगा", hogi: "होगी",
  karta: "करता", karti: "करती", karte: "करते", kare: "करे",
  // postpositions and conjunctions
  ka: "का", ke: "के", ki: "की", ko: "को", se: "से",
  me: "में", mein: "में", par: "पर", tak: "तक", liye: "लिए",
  aur: "और", ya: "या", lekin: "लेकिन", agar: "अगर", to: "तो",
  bare: "बारे", baare: "बारे",
  // demonstratives, quantifiers, negation
  yeh: "यह", ye: "ये", woh: "वह", vah: "वह", wo: "वो",
  ek: "एक", sab: "सब", kuch: "कुछ", koi: "कोई", sabhi: "सभी",
  nahi: "नहीं", nahin: "नहीं", haan: "हाँ", bhi: "भी",
  naam: "नाम", log: "लोग", cheez: "चीज़",
};

/**
 * Request scaffolding, dropped rather than mapped.
 *
 * "mujhe … bataiye" names no subject, and in Devanagari it embeds toward
 * first-person narrative passages: transliterated in full, "mujhe corporation ke
 * bare mein bataiye" returns a passage beginning "मुझे पता चला कि मेरे खिलाफ…".
 * Removing the first-person framing and keeping the rest scores better than
 * either transliterating or stripping everything.
 */
const SCAFFOLD = new Set([
  "mujhe", "muje", "mera", "meri", "mere", "hume", "humein",
  "batao", "bataiye", "bataye", "bata", "batana", "batayen",
  "samjhao", "samjhaiye", "samjha", "likho", "likhiye",
  "kripya", "kripaya", "please", "plz", "zara", "jara",
  "bhai", "yaar", "sir", "madam",
]);

/**
 * Romanised tokens that are also ordinary English words.
 *
 * Excluded from *detection* only. "how much does he pay me" must not read as
 * Hindi on the strength of `he` and `me`; but once a query is established as
 * romanised Hindi by its other words, `me` in it really is में.
 */
const AMBIGUOUS = new Set([
  "he", "me", "the", "to", "so", "par", "ki", "se", "ye", "wo", "h",
  "bare", "log", "sab", "ka", "tak", "kis", "arth", "hai",
]);

/** Words carrying no `\p{M}` problem: see `tokens.ts` for why marks are kept. */
const WORD_BREAK = /[^\p{L}\p{N}\p{M}\s]/gu;

function words(q: string): string[] {
  return q.toLowerCase().replace(WORD_BREAK, " ").split(/\s+/).filter(Boolean);
}

/** Fraction of the query that is unambiguously romanised Hindi. */
function markerRatio(w: string[]): number {
  if (!w.length) return 0;
  let n = 0;
  for (const t of w) {
    if (SCAFFOLD.has(t)) { n++; continue; }
    if (TRANSLIT[t] && !AMBIGUOUS.has(t)) n++;
  }
  return n / w.length;
}

/**
 * Minimum share of romanised-Hindi markers before a query is treated as Hinglish.
 *
 * Real romanised Hindi measures 0.5 and up; English and Devanagari queries
 * measure 0. The gap is wide because `AMBIGUOUS` removes the words that appear in
 * both languages, so the floor exists to stop a single stray token — a name, a
 * typo, an acronym — from rewriting an English question.
 */
const MIN_MARKER_RATIO = 0.3;

/** Latin script with enough romanised-Hindi markers to be worth folding. */
export function isRomanisedHindi(q: string): boolean {
  if (!/[a-z]/i.test(q)) return false;
  return markerRatio(words(q)) >= MIN_MARKER_RATIO;
}

/**
 * Fold romanised Hindi function words into Devanagari, leaving everything else
 * as written. Returns the query unchanged when it is not romanised Hindi.
 */
export function foldRomanisedHindi(q: string): string {
  if (!isRomanisedHindi(q)) return q;
  const out = words(q)
    .filter((w) => !SCAFFOLD.has(w))
    .map((w) => TRANSLIT[w] ?? w);
  return out.length ? out.join(" ") : q;
}
