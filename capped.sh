#!/usr/bin/env bash
# capped.sh -- run a command and kill it if its process tree exceeds the
# resident-memory ceiling (CODING_RULES.md 10a: 8 GB). Polls every second;
# reports the peak. Usage: ./capped.sh <command...>
CEILING_KB=${CAPPED_CEILING_KB:-8388608}
"$@" &
ROOT=$!
peak=0
while kill -0 "$ROOT" 2>/dev/null; do
  total=$(ps -o pid=,ppid=,rss= -ax | awk -v root="$ROOT" '
    { pid[NR]=$1; ppid[NR]=$2; rss[NR]=$3 }
    END {
      inset[root]=1
      changed=1
      while (changed) { changed=0
        for (i=1;i<=NR;i++) if (!(pid[i] in inset) && (ppid[i] in inset)) { inset[pid[i]]=1; changed=1 } }
      s=0; for (i=1;i<=NR;i++) if (pid[i] in inset) s+=rss[i]; print s }')
  [ "$total" -gt "$peak" ] && peak=$total
  if [ "$total" -gt "$CEILING_KB" ]; then
    echo "capped: process tree at $((total/1024)) MB resident, above the $((CEILING_KB/1024)) MB ceiling -- killed" >&2
    pkill -9 -P "$ROOT" 2>/dev/null; kill -9 "$ROOT" 2>/dev/null
    exit 137
  fi
  sleep 1
done
wait "$ROOT"; rc=$?
echo "capped: peak resident $((peak/1024)) MB, exit $rc" >&2
exit $rc
