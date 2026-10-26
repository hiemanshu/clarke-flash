#!/usr/bin/env bash
# Live status for scripts/fetch_wiki_dumps.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
DEST=data/raw
SUMS="$DEST/SHA256SUMS"
LOG=logs/wiki_download.log

total=$(grep -c '^[0-9a-f]' "$SUMS")
present=$(grep '^[0-9a-f]' "$SUMS" | awk '{print $2}' | while read -r n; do
  [[ -f "$DEST/$n" ]] && echo x
done | wc -l)

echo "shards complete: $present/$total"
echo "in-flight:"
for p in "$DEST"/*.part; do
  [[ -e "$p" ]] || continue
  printf '  %-55s %6.2f GB\n' "$(basename "$p")" \
    "$(awk -v b="$(stat -c%s "$p")" 'BEGIN{print b/1e9}')"
done
echo
command -v pgrep >/dev/null && pgrep -af 'fetch_wiki_dumps|curl.*mediawiki_content' | sed 's/^/  /'
echo
echo "--- last log lines ---"
tail -n "${1:-15}" "$LOG" 2>/dev/null
