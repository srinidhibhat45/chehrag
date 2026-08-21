#!/usr/bin/env bash
#
# Check that a deployed Chehrag actually works, from outside.
#
#   ./scripts/verify-deploy.sh https://chehrag.pages.dev https://chehrag-worker.x.workers.dev
#
# "It deployed" and "it works" are different claims. This makes the requests a
# visitor's browser would make and checks the answers, including the two headers
# that are load-bearing rather than cosmetic: without cross-origin isolation the
# embedding graph silently drops to single-threaded WASM and every query gets
# roughly three times slower for no visible reason.

set -uo pipefail
SITE="${1:?usage: verify-deploy.sh <site-url> <worker-url>}"
WORKER="${2:?usage: verify-deploy.sh <site-url> <worker-url>}"

grn=$'\033[32m'; red=$'\033[31m'; ylw=$'\033[33m'; dim=$'\033[2m'; off=$'\033[0m'
FAILED=0
ok()   { echo "  ${grn}✓${off} $1"; }
bad()  { echo "  ${red}✗${off} $1"; FAILED=1; }
warn() { echo "  ${ylw}!${off} $1"; }

# --- the page loads ----------------------------------------------------------
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$SITE/")
[ "$code" = 200 ] && ok "site responds 200" || bad "site returned HTTP $code"

# --- cross-origin isolation --------------------------------------------------
H=$(curl -s -D - -o /dev/null --max-time 20 "$SITE/")
echo "$H" | grep -qi 'cross-origin-opener-policy: same-origin' \
  && ok "COOP header present" \
  || bad "COOP missing - threaded WASM will be disabled and queries ~3x slower"
echo "$H" | grep -qi 'cross-origin-embedder-policy: credentialless' \
  && ok "COEP header present" \
  || bad "COEP missing - threaded WASM will be disabled and queries ~3x slower"

# --- the index is actually served --------------------------------------------
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$SITE/index/manifest.json")
[ "$code" = 200 ] && ok "index manifest served" || bad "index manifest HTTP $code"

# The largest blob is the one most likely to have been rejected at upload.
read -r code bytes < <(curl -s -o /dev/null -w '%{http_code} %{size_download}' --max-time 60 "$SITE/index/passage_int8.bin" -r 0-1048575 2>/dev/null; echo)
[ "$code" = 206 ] || [ "$code" = 200 ] && ok "largest index blob reachable" || bad "passage_int8.bin HTTP $code"

# --- passage text is compressed ----------------------------------------------
enc=$(curl -s -o /dev/null -D - --max-time 30 -H 'accept-encoding: gzip, br' "$SITE/index/passages.000.json" | grep -i '^content-encoding:' | tr -d '\r' | awk '{print $2}')
if [ -n "$enc" ]; then ok "passage text served as $enc (74 MB of JSON → ~14 MB over the wire)"
else warn "passage text is NOT compressed - first load will be ~60 MB heavier than it needs to be"; fi

# --- the Worker --------------------------------------------------------------
HEALTH=$(curl -s --max-time 20 "$WORKER/health")
echo "$HEALTH" | grep -q '"ok":true' && ok "worker healthy" || bad "worker /health: $HEALTH"
echo "$HEALTH" | grep -q '"llm":true'          && ok "generator configured" || bad "no generator - answers will be quoted passages"
echo "$HEALTH" | grep -q '"sarvam":true'       && ok "Sarvam configured"    || warn "Sarvam not configured - voice falls back to the browser"

# --- a real generated answer, end to end -------------------------------------
OUT=$(curl -sN --max-time 45 -X POST "$WORKER/synthesize" \
  -H 'content-type: application/json' \
  -d '{"query":"what is a corporation","sources":[{"title":"MS MARCO-XI","text":"निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है।"}]}' \
  | python3 -c "
import sys, json
out = ''
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data:'):
        try: e = json.loads(line[5:])
        except Exception: continue
        if 't' in e: out += e['t']
print(out.strip())" 2>/dev/null)
if [ -n "$OUT" ]; then ok "live cross-lingual answer: \"$(echo "$OUT" | head -c 70)...\""
else bad "generation returned nothing"; fi

echo
if [ "$FAILED" = 1 ]; then echo "  ${red}deployment has problems - see above.${off}"; exit 1; fi
echo "  ${grn}deployment verified.${off}"
