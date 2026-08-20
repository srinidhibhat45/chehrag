/**
 * Minimal `SpeechRecognition` typings.
 *
 * The Web Speech *synthesis* half is in `lib.dom.d.ts`; the *recognition* half
 * is not, because it never left the WICG stage and remains a Chromium-only
 * prefixed API in practice. TypeScript ships `SpeechRecognitionResult` but not
 * the interface that produces one.
 *
 * Only the members `stt/index.ts` actually touches are declared. Typing more of
 * it would be inventing a contract for an API that has no agreed one, and the
 * extra surface would just be a place for a wrong assumption to hide.
 */

interface SpeechRecognitionAlternative {
  readonly transcript: string;
  readonly confidence: number;
}

interface SpeechRecognitionResult {
  readonly isFinal: boolean;
  readonly length: number;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionResultList {
  readonly length: number;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionEvent extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  readonly error: string;
  readonly message: string;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => unknown) | null;
  onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => unknown) | null;
  onend: ((this: SpeechRecognition, ev: Event) => unknown) | null;
  onstart: ((this: SpeechRecognition, ev: Event) => unknown) | null;
  start(): void;
  stop(): void;
  abort(): void;
}
