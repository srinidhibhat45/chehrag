#!/usr/bin/env bash
#
# Verify every configured key by actually calling the provider.
#
# "The key is in the file" and "the key works" are different facts, and only the
# second one matters. This makes the smallest real request each API allows and
# reports what came back. Keys are read from the files and sent only to their
# own vendor; nothing is printed but status.
#
#   ./scripts/check-keys.sh

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

bold=$'\033[1m'; dim=$'\033[2m'; grn=$'\033[32m'; ylw=$'\033[33m'; red=$'\033[31m'; off=$'\033[0m'

read_key() {   # $1 file  $2 KEY
  [ -f "$1" ] || return 1
  local v
  v="$(grep -E "^[[:space:]]*$2=" "$1" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]')"
  [ -n "$v" ] && printf '%s' "$v"
}

ok()   { printf '  %s✓%s %s\n' "$grn" "$off" "$1"; }
bad()  { printf '  %s✗%s %s\n' "$red" "$off" "$1"; FAILED=1; }
skip() { printf '  %s-%s %s\n' "$dim" "$off" "$1"; }

FAILED=0
echo "${bold}Chehrag - key check${off}"

# --- Groq --------------------------------------------------------------------
echo
echo "${bold}Groq${off} ${dim}- writes the answers${off}"
GROQ="$(read_key web/.env.local GROQ_API_KEY)"
if [ -z "${GROQ:-}" ]; then
  skip "not set - the app will quote passages instead of answering"
  echo "    ${dim}fix: ./scripts/setup-keys.sh groq${off}"
else
  # A real one-token generation, not just /models: it proves the key can do the
  # thing the app needs, on the model the app names.
  MODEL="$(read_key web/.env.local LLM_MODEL)"; MODEL="${MODEL:-openai/gpt-oss-20b}"
  start=$(date +%s%N)
  body=$(curl -s -w '\n%{http_code}' --max-time 25 \
    https://api.groq.com/openai/v1/chat/completions \
    -H "authorization: Bearer $GROQ" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ready\"}]}")
  ms=$(( ($(date +%s%N) - start) / 1000000 ))
  code=$(printf '%s' "$body" | tail -1)
  case "$code" in
    200) ok "key works · model $MODEL · ${ms} ms round trip" ;;
    401|403) bad "key rejected (HTTP $code) - wrong or revoked key" ;;
    404) bad "model '$MODEL' not available to this key (HTTP 404)"
         echo "    ${dim}see https://console.groq.com/docs/models${off}" ;;
    429) bad "rate limited (HTTP 429) - free tier is 30 req/min" ;;
    *)   bad "HTTP $code"
         printf '%s' "$body" | head -c 200 | sed 's/^/    /' ;;
  esac
fi

# --- Anthropic (only if used instead of Groq) --------------------------------
ANTH="$(read_key web/.env.local ANTHROPIC_API_KEY)"
if [ -n "${ANTH:-}" ]; then
  echo
  echo "${bold}Anthropic${off} ${dim}- alternative answer writer${off}"
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
    https://api.anthropic.com/v1/messages \
    -H "x-api-key: $ANTH" -H 'anthropic-version: 2023-06-01' -H 'content-type: application/json' \
    -d '{"model":"claude-haiku-4-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}')
  [ "$code" = 200 ] && ok "key works" || bad "HTTP $code"
fi

# --- Sarvam ------------------------------------------------------------------
echo
echo "${bold}Sarvam${off} ${dim}- speech in, and Indic speech out${off}"
SARVAM="$(read_key worker/.dev.vars SARVAM_API_KEY)"
if [ -z "${SARVAM:-}" ]; then
  skip "not set - voice input falls back to the browser's own recogniser"
  echo "    ${dim}fix: ./scripts/setup-keys.sh sarvam${off}"
else
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
    https://api.sarvam.ai/text-to-speech \
    -H "api-subscription-key: $SARVAM" -H 'content-type: application/json' \
    -d '{"text":"ready","target_language_code":"hi-IN"}')
  case "$code" in
    200) ok "key works" ;;
    401|403) bad "key rejected (HTTP $code)" ;;
    *)   bad "HTTP $code - reachable but the test call failed" ;;
  esac
fi

# --- ElevenLabs --------------------------------------------------------------
echo
echo "${bold}ElevenLabs${off} ${dim}- spoken answers in English, Hindi, Tamil${off}"
ELEVEN="$(read_key worker/.dev.vars ELEVENLABS_API_KEY)"
if [ -z "${ELEVEN:-}" ]; then
  skip "not set - answers are spoken by Sarvam or the browser voice"
  echo "    ${dim}fix: ./scripts/setup-keys.sh elevenlabs${off}"
else
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
    https://api.elevenlabs.io/v1/user -H "xi-api-key: $ELEVEN")
  case "$code" in
    200) ok "key works" ;;
    401) bad "key rejected (HTTP 401)" ;;
    *)   bad "HTTP $code" ;;
  esac
fi

echo
if [ "$FAILED" = 1 ]; then
  echo "${red}${bold}some keys failed.${off} Fix them, then re-run this."
  exit 1
fi
echo "${grn}${bold}all configured keys work.${off}"
echo "  ${dim}npm run dev  →  http://localhost:5173${off}"
