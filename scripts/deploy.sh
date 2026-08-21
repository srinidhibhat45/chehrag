#!/usr/bin/env bash
#
# Deploy Chehrag to Cloudflare: Worker first, then Pages, then lock CORS.
#
#   ./scripts/deploy.sh
#
# Order is not arbitrary. The web build inlines `VITE_WORKER_BASE`, so the
# Worker has to exist and have a URL before the site is built, or the site ships
# pointing at nothing. And `ALLOWED_ORIGIN` cannot be set until the Pages origin
# exists, so the Worker is deployed twice: once to get its URL, once to lock it
# down. Skipping the second pass leaves `/fetch-url` open as a fetch proxy on
# your quota.
#
# Safe to re-run. Secrets are read from the local files you already filled in.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

bold=$'\033[1m'; dim=$'\033[2m'; grn=$'\033[32m'; ylw=$'\033[33m'; red=$'\033[31m'; off=$'\033[0m'
step() { echo; echo "${bold}$1${off}"; }
ok()   { echo "  ${grn}✓${off} $1"; }
warn() { echo "  ${ylw}!${off} $1"; }
die()  { echo "  ${red}✗${off} $1"; exit 1; }

PROJECT="${PAGES_PROJECT:-chehrag}"

read_key() {   # $1 file  $2 KEY
  [ -f "$1" ] || return 1
  local v
  v="$(grep -E "^[[:space:]]*$2=" "$1" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]')"
  [ -n "$v" ] && printf '%s' "$v"
}

# --- 0. preflight ------------------------------------------------------------
step "0 · preflight"

# `wrangler whoami` exits 0 whether or not you are logged in, so the exit code
# proves nothing and the output has to be read.
WHO="$(npx wrangler whoami 2>&1)"
if echo "$WHO" | grep -q "not authenticated"; then
  die "not logged in to Cloudflare.

  ${bold}Run this once, in your own terminal${off} - it opens a browser and takes ~20 seconds:

      ${bold}npx wrangler login${off}

  Then run this script again. (Alternatively, if you would rather use a token:
  create one at https://dash.cloudflare.com/profile/api-tokens with the
  \"Edit Cloudflare Workers\" template, then re-run with
  CLOUDFLARE_API_TOKEN=... ./scripts/deploy.sh)"
fi
ACCOUNT="$(echo "$WHO" | grep -oE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+' | head -1)"
ok "Cloudflare authenticated${ACCOUNT:+ as $ACCOUNT}"

GROQ="$(read_key web/.env.local GROQ_API_KEY)"
SARVAM="$(read_key worker/.dev.vars SARVAM_API_KEY)"
ELEVEN="$(read_key worker/.dev.vars ELEVENLABS_API_KEY)"
[ -n "${GROQ:-}" ]   && ok "Groq key found"        || warn "no Groq key - the site will quote passages, not answer"
[ -n "${SARVAM:-}" ] && ok "Sarvam key found"      || warn "no Sarvam key - voice input will fall back to the browser"
[ -n "${ELEVEN:-}" ] && ok "ElevenLabs key found"  || true

# The largest index blob sits close to the Pages per-file ceiling, and a rebuilt
# index could cross it. Cheaper to fail here than halfway through an upload.
BIG=$(find web/public/index -type f -exec stat -f "%z" {} + 2>/dev/null | sort -rn | head -1)
if [ -n "${BIG:-}" ] && [ "$BIG" -ge 26214400 ]; then
  die "an index blob is $(echo "scale=1; $BIG/1048576" | bc) MiB - Cloudflare Pages rejects files over 25 MiB."
fi
ok "largest index blob $(echo "scale=1; ${BIG:-0}/1048576" | bc) MiB, under the 25 MiB Pages limit"

# --- 1. Worker ---------------------------------------------------------------
step "1 · deploying the Worker"
WORKER_OUT="$(cd worker && npx wrangler deploy 2>&1)"
echo "$WORKER_OUT" | grep -E "https://" | sed 's/^/  /'
WORKER_URL="$(echo "$WORKER_OUT" | grep -oE 'https://[a-z0-9.-]+\.workers\.dev' | head -1)"
[ -n "$WORKER_URL" ] || die "could not determine the Worker URL. Output above."
ok "Worker at $WORKER_URL"

# --- 2. secrets --------------------------------------------------------------
step "2 · uploading secrets"
put_secret() {  # $1 NAME  $2 value
  [ -n "${2:-}" ] || return 0
  if printf '%s' "$2" | (cd worker && npx wrangler secret put "$1" >/dev/null 2>&1); then
    ok "$1"
  else
    warn "$1 failed to upload"
  fi
}
put_secret GROQ_API_KEY       "${GROQ:-}"
put_secret SARVAM_API_KEY     "${SARVAM:-}"
put_secret ELEVENLABS_API_KEY "${ELEVEN:-}"

echo "  ${dim}waiting for the Worker to pick up its secrets...${off}"
for i in $(seq 1 10); do
  H="$(curl -s --max-time 10 "$WORKER_URL/health" 2>/dev/null)"
  echo "$H" | grep -q '"llm":true' && break
  sleep 2
done
echo "  ${dim}$H${off}"

# --- 3. Pages ----------------------------------------------------------------
step "3 · building the site against $WORKER_URL"
# Written to .env.production so it is picked up by `vite build` without
# disturbing .env.local, which is what `npm run dev` reads.
printf '# Written by scripts/deploy.sh. Points the built site at the deployed Worker.\nVITE_WORKER_BASE=%s\n' "$WORKER_URL" > web/.env.production
npm run build >/dev/null 2>&1 || die "build failed - run 'npm run build' to see why"
ok "built ($(du -sh web/dist | cut -f1))"

step "4 · uploading to Cloudflare Pages"
# `wrangler pages deploy` does not create a missing project - it fails with a
# hint. Creating it up front is idempotent and turns a confusing failure
# halfway through a 131 MB upload into a no-op.
npx wrangler pages project create "$PROJECT" --production-branch main >/dev/null 2>&1 \
  && ok "created Pages project '$PROJECT'" || true
echo "  ${dim}131 MB of index blobs - this is the slow step, give it a few minutes${off}"
PAGES_OUT="$(npx wrangler pages deploy web/dist --project-name "$PROJECT" --commit-dirty=true 2>&1)"
echo "$PAGES_OUT" | tail -5 | sed 's/^/  /'
SITE_URL="$(echo "$PAGES_OUT" | grep -oE 'https://[a-z0-9.-]+\.pages\.dev' | tail -1)"
[ -n "$SITE_URL" ] || die "could not determine the Pages URL. Output above."
ok "site at $SITE_URL"

# --- 4. lock CORS ------------------------------------------------------------
step "5 · locking the Worker to that origin"
# /fetch-url makes outbound requests on your quota. Leaving ALLOWED_ORIGIN at
# "*" lets any site on the internet use your Worker as a proxy.
# `wrangler pages deploy` prints the per-deploy preview origin
# (https://<hash>.chehrag.pages.dev). The stable production origin is that with
# the leading label removed - but only when there IS a leading label, or
# "https://chehrag.pages.dev" would be mangled into "https://pages.dev".
PREVIEW_ORIGIN="$SITE_URL"
if [ "$(echo "$SITE_URL" | tr '.' '\n' | wc -l | tr -d ' ')" -ge 4 ]; then
  PROD_ORIGIN="https://$(echo "$SITE_URL" | sed -E 's#^https://[^.]+\.##')"
else
  PROD_ORIGIN="$SITE_URL"
fi
# Both are pinned: a visitor may be handed either.
ORIGINS="$PROD_ORIGIN"
[ "$PREVIEW_ORIGIN" != "$PROD_ORIGIN" ] && ORIGINS="$PROD_ORIGIN,$PREVIEW_ORIGIN"
if grep -q 'ALLOWED_ORIGIN = "\*"' worker/wrangler.toml; then
  perl -pi -e "s#ALLOWED_ORIGIN = \"\\*\"#ALLOWED_ORIGIN = \"$ORIGINS\"#" worker/wrangler.toml
  (cd worker && npx wrangler deploy >/dev/null 2>&1) && ok "ALLOWED_ORIGIN = $ORIGINS" \
    || warn "redeploy failed - set ALLOWED_ORIGIN manually and redeploy"
else
  ok "ALLOWED_ORIGIN already set"
fi

# --- 5. verify ---------------------------------------------------------------
step "6 · verifying the live deployment"
./scripts/verify-deploy.sh "$SITE_URL" "$WORKER_URL"

echo
echo "${bold}${grn}live${off}  $SITE_URL"
echo "${dim}worker${off} $WORKER_URL"
echo
