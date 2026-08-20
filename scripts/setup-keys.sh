#!/usr/bin/env bash
#
# Interactive key setup.
#
# Prompts for each key one at a time and writes it to the file that needs it.
# Input is read straight from your terminal into the file — it is not echoed,
# not logged, and never passes through a chat transcript or an agent's context.
#
#   ./scripts/setup-keys.sh          set up everything, skipping what you lack
#   ./scripts/setup-keys.sh groq     just one
#
# Re-running is safe: it replaces the line for a key you re-enter and leaves
# every other line alone.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

WEB_ENV="web/.env.local"
WORKER_ENV="worker/.dev.vars"

bold=$'\033[1m'; dim=$'\033[2m'; grn=$'\033[32m'; ylw=$'\033[33m'; red=$'\033[31m'; off=$'\033[0m'

# --- write KEY=value into a file, replacing any existing line for that key ----
put() {
  local file="$1" key="$2" val="$3"
  [ -f "$file" ] || { mkdir -p "$(dirname "$file")"; touch "$file"; }
  # Rewrite without the old line, then append. Done via a temp file so an
  # interrupted run cannot leave the file half-written.
  local tmp; tmp="$(mktemp)"
  grep -v -E "^[[:space:]]*${key}=" "$file" > "$tmp" 2>/dev/null || true
  printf '%s=%s\n' "$key" "$val" >> "$tmp"
  mv "$tmp" "$file"
  chmod 600 "$file"
}

# --- prompt for one key ------------------------------------------------------
# $1 file  $2 KEY  $3 label  $4 where to get it  $5 what it buys  $6 expected prefix ("" = any)
ask() {
  local file="$1" key="$2" label="$3" url="$4" buys="$5" prefix="$6"
  echo
  echo "${bold}${label}${off}"
  echo "  ${dim}buys:${off} $buys"
  echo "  ${dim}get: ${off} $url"
  echo "  ${dim}file:${off} $file"
  printf '  paste key (or press Enter to skip): '

  local val
  read -rs val            # -s: not echoed to the screen
  echo

  if [ -z "$val" ]; then
    echo "  ${ylw}skipped${off}"
    return
  fi
  # Strip accidental surrounding whitespace/quotes from a paste.
  val="$(printf '%s' "$val" | tr -d '[:space:]' | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"

  if [ -n "$prefix" ] && [[ "$val" != ${prefix}* ]]; then
    echo "  ${red}that does not start with '${prefix}' — not saved.${off}"
    echo "  ${dim}re-run this script to try again${off}"
    return
  fi

  put "$file" "$key" "$val"
  echo "  ${grn}saved${off} to $file  ${dim}(${#val} chars, chmod 600)${off}"
}

want() { [ "$#" -eq 0 ] || [ "${TARGET:-all}" = "all" ] || [ "${TARGET}" = "$1" ]; }

TARGET="${1:-all}"

echo "${bold}Chehrag — key setup${off}"
echo "${dim}Nothing you type here is shown on screen or written anywhere except the"
echo "file named under each prompt. Both files are gitignored.${off}"

if [ "$TARGET" = all ] || [ "$TARGET" = groq ]; then
  ask "$WEB_ENV" GROQ_API_KEY \
      "1. Groq  ${grn}[REQUIRED — free, no card]${off}" \
      "https://console.groq.com/keys" \
      "written answers. Without this the app finds the right passage but can only quote it." \
      "gsk_"
fi

if [ "$TARGET" = all ] || [ "$TARGET" = sarvam ]; then
  ask "$WORKER_ENV" SARVAM_API_KEY \
      "2. Sarvam  ${ylw}[optional — voice in]${off}" \
      "https://dashboard.sarvam.ai" \
      "real speech input, and spoken answers in the 8 Indic languages ElevenLabs lacks." \
      ""
fi

if [ "$TARGET" = all ] || [ "$TARGET" = elevenlabs ]; then
  ask "$WORKER_ENV" ELEVENLABS_API_KEY \
      "3. ElevenLabs  ${ylw}[optional — voice out]${off}" \
      "https://elevenlabs.io/app/settings/api-keys" \
      "spoken answers in English, Hindi and Tamil." \
      ""
fi

echo
echo "${bold}what is configured now${off}"
for f in "$WEB_ENV" "$WORKER_ENV"; do
  [ -f "$f" ] || continue
  echo "  ${dim}$f${off}"
  # Names and lengths only — never the values.
  grep -E '^[A-Z_]+=' "$f" 2>/dev/null | while IFS='=' read -r k v; do
    printf '    %-22s %s\n' "$k" "${grn}set${off} ${dim}(${#v} chars)${off}"
  done
done

echo
echo "${bold}next${off}"
echo "  ./scripts/check-keys.sh     ${dim}verify each key actually works${off}"
echo "  npm run dev                 ${dim}http://localhost:5173${off}"
echo
