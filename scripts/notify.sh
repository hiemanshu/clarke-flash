#!/usr/bin/env bash
# Mobile-first notifier. Channels (checked in order):
#   1) Signal  : if $SIGNAL_TO set and signal-cli works -> sends via linked device
#   2) ntfy.sh : if $NTFY_TOPIC set                       -> HTTP POST (phone app)
#   3) fallback: log only (NO desktop popups - owner's explicit preference)
# usage: notify.sh "title" "body" [low|normal|critical]
set -u
TITLE="${1:-Clarke}"; BODY="${2:-}"; URG="${3:-normal}"
CFG="$HOME/.config/clarke/notify.env"
[ -f "$CFG" ] && . "$CFG"
sent=0
if [ -n "${SIGNAL_ACCOUNT:-}" ] && command -v signal-cli >/dev/null 2>&1; then
  signal-cli -a "$SIGNAL_ACCOUNT" send --note-to-self -m "$TITLE: $BODY" >/dev/null 2>&1 && sent=1
fi
if [ "$sent" -eq 0 ] && [ -n "${NTFY_TOPIC:-}" ]; then
  curl -s --max-time 10 -H "Title: $TITLE" -H "Priority: ${NTFY_PRIORITY:-default}" \
    -d "$BODY" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 && sent=1
fi
echo "$(date '+%H:%M:%S') [$URG] sent=$sent $TITLE — $BODY" >> /tmp/opencode/notify.log
