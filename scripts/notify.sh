#!/usr/bin/env bash
# Two-tier mobile notifier (owner design 2026-10-24):
#   ntfy  = short ping  ("check Signal / headline") -> phone buzz
#   Signal = full description (note-to-self)        -> the archive + detail
# modes: short (ntfy only) | full (Signal detail + ntfy ping) | critical (both, high prio)
# usage: notify.sh "title" "body" [short|full|critical]
set -u
TITLE="${1:-Clarke}"; BODY="${2:-}"; MODE="${3:-full}"
CFG="$HOME/.config/clarke/notify.env"; [ -f "$CFG" ] && . "$CFG"

sig=0; ntf=0
signal_send() {  # $1 = message text
  [ -n "${SIGNAL_ACCOUNT:-}" ] || return 1
  command -v signal-cli >/dev/null 2>&1 || return 1
  signal-cli -a "$SIGNAL_ACCOUNT" send --note-to-self -m "$1" >/dev/null 2>&1
}
ntfy_send() {    # $1 title  $2 priority  $3 tag  $4 body
  [ -n "${NTFY_URL:-}" ] || return 1
  curl -fsS --max-time 10 -H "Title: $1" -H "Priority: $2" -H "Tags: $3" \
       ${NTFY_TOKEN:+-H "Authorization: Bearer $NTFY_TOKEN"} \
       -d "$4" "$NTFY_URL" >/dev/null 2>&1
}
case "$MODE" in
  short)
    ntfy_send "$TITLE" default robot "$BODY" && ntf=1 ;;
  full)
    signal_send "$TITLE

$BODY" && sig=1
    ntfy_send "📬 Check Signal" default inbox "Clarke: $TITLE" && ntf=1 ;;
  critical)
    signal_send "⚠️ $TITLE

$BODY" && sig=1
    ntfy_send "⚠️ $TITLE" high warning "$BODY" && ntf=1 ;;
esac
echo "$(date '+%H:%M:%S') [$MODE] ntfy=$ntf signal=$sig  $TITLE — $BODY" >> /tmp/opencode/notify.log
