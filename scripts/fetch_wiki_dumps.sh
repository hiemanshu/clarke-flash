#!/usr/bin/env bash
# Download + verify Wikimedia "mediawiki_content_current" enwiki xml.bz2 shards.
#
# Transport notes (measured 2026-10-24):
#   dumps.wikimedia.org      ~4-5 MB/s and rate limits to ~4 concurrent streams per IP
#                            (HTTP 429) - this is a hard, documented cap.
#   Umea mirror (accum.se)   ~63 MB/s single stream, same bytes, same SHA256SUMS.
# So: pull from the mirror first, fall back to the origin only if the mirror fails.
# Always send a descriptive User-Agent (Wikimedia/429 friendly).
# Transfers are resumable (curl -C -) and every file is sha256-verified; a resumed
# file that fails its hash is discarded and re-fetched from zero, so resume can never
# leave a corrupt file behind.
set -uo pipefail

TAIL="other/mediawiki_content_current/enwiki/2026-10-01/xml/bzip2"
HOSTS=(
  "https://ftp.acc.umu.se/mirror/wikimedia.org"
  "https://mirror.accum.se/mirror/wikimedia.org"
  "https://dumps.wikimedia.your.org"
  "https://dumps.wikimedia.org"
)
UA="clarke-flash-dump-fetch/1.0 (personal research corpus build; contact: hiemanshu@localhost) curl"

DEST="data/raw"
SUMS="$DEST/SHA256SUMS"
LOG="logs/wiki_download.log"
PARALLEL="${PARALLEL:-2}"
ATTEMPTS="${ATTEMPTS:-10}"
MINRATE="${MINRATE:-200000}"        # abort transfers slower than this (bytes/s) ...
MINRATE_TIME="${MINRATE_TIME:-60}"  # ... sustained for this many seconds, then retry (resumes)

mkdir -p "$DEST" logs
cd "$(dirname "$0")/.." || exit 1

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

if [[ ! -s "$SUMS" ]]; then
  curl -fsSL -A "$UA" --retry 5 -o "$SUMS.tmp" "${HOSTS[-1]}/$TAIL/SHA256SUMS" && mv "$SUMS.tmp" "$SUMS"
fi
log "SHA256SUMS present ($(grep -c '^[0-9a-f]' "$SUMS") entries)"

fetch_one() {
  local name="$1" want="$2" attempt have rc host
  local out="$DEST/$name" part="$DEST/$name.part" errlog="logs/curl.$name.err"
  local resume=1
  # arrays are not exported to subshells, so rebuild the host list here
  local -a hosts=()
  mapfile -t hosts <<< "$HOSTS_STR"
  local nhosts=${#hosts[@]}

  if [[ -f "$out" ]]; then
    have=$(sha256sum "$out" | cut -d' ' -f1)
    if [[ "$have" == "$want" ]]; then
      log "OK       $name (already complete, verified)"
      return 0
    fi
    log "MISMATCH $name (existing file bad) -> discarding"
    rm -f "$out"
  fi

  for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
    host="${hosts[$(( (attempt - 1) % nhosts ))]}"
    local args=( -L --fail --no-progress-meter -A "$UA"
                 --connect-timeout 30 --speed-time "$MINRATE_TIME" --speed-limit "$MINRATE"
                 -o "$part" "$host/$TAIL/$name" )

    if [[ $resume -eq 1 && -s "$part" ]]; then
      log "RESUME   $name at $(stat -c%s "$part") bytes from $(printf '%s' "$host" | cut -d/ -f3) (attempt $attempt/$ATTEMPTS)"
      curl -C - "${args[@]}" 2>>"$errlog"
    else
      rm -f "$part"
      log "GET      $name from scratch from $(printf '%s' "$host" | cut -d/ -f3) (attempt $attempt/$ATTEMPTS)"
      curl "${args[@]}" 2>>"$errlog"
    fi
    rc=$?

    if [[ $rc -ne 0 ]]; then
      log "FAILED   $name curl rc=$rc at $( [[ -f $part ]] && stat -c%s "$part" || echo 0 ) bytes (attempt $attempt) - will resume"
      continue
    fi

    have=$(sha256sum "$part" | cut -d' ' -f1)
    if [[ "$have" == "$want" ]]; then
      mv "$part" "$out"
      log "DONE     $name verified ($(stat -c%s "$out") bytes)"
      rm -f "$errlog"
      return 0
    fi

    log "MISMATCH $name sha256=$have (attempt $attempt) -> discarding, re-fetching from zero"
    rm -f "$part"
    resume=0
  done

  log "GIVEUP   $name after $ATTEMPTS attempts (partial kept for resume)"
  return 1
}
export -f fetch_one log
export DEST TAIL LOG ATTEMPTS MINRATE MINRATE_TIME UA
export HOSTS_STR
HOSTS_STR=$(printf '%s\n' "${HOSTS[@]}")

log "=== verify existing / download missing (parallel=$PARALLEL, hosts: Umea -> accum -> your.org -> origin) ==="
grep '^[0-9a-f]' "$SUMS" | xargs -P "$PARALLEL" -n 2 bash -c 'fetch_one "$1" "$0"' \
  || log "WARNING: at least one shard failed"

log "=== final verification pass ==="
ok=0; bad=0
while read -r want name; do
  [[ -z "${name:-}" ]] && continue
  if [[ -f "$DEST/$name" ]]; then
    have=$(sha256sum "$DEST/$name" | cut -d' ' -f1)
    if [[ "$have" == "$want" ]]; then ok=$((ok+1)); else bad=$((bad+1)); log "BAD     $name"; fi
  else
    bad=$((bad+1)); log "MISSING $name"
  fi
done < "$SUMS"
log "=== summary: ok=$ok bad=$bad ==="
[[ $bad -eq 0 ]]
