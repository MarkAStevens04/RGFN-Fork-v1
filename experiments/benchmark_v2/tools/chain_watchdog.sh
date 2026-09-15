#!/bin/bash
# chain_watchdog.sh -- top the afterany chains back up as the queue drains.
#
# WHY: the QOS cap is SIXTY submitted jobs per user (measured -- sbatch returns
# QOSMaxSubmitJobPerUserLimit at exactly 60, which is why scale5k used CAP=60). The campaign needs
# more links than that at once, so links have to be added as earlier ones finish. No cron on this
# login node, so this is the bounded flock'd loop the project already uses.
#
#   setsid nohup bash experiments/benchmark_v2/tools/chain_watchdog.sh > ~/chain_watchdog.log 2>&1 &
#
# Bounded on purpose: it exits after MAX_ROUNDS so a forgotten watchdog cannot run forever.
set -uo pipefail
[ -z "${SLURM_JOB_ID:-}" ] || { echo "FATAL: login node only." >&2; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO" || exit 1

INTERVAL=${INTERVAL:-1800}          # 30 min
MAX_ROUNDS=${MAX_ROUNDS:-96}        # ~48 h, then it stops by itself
LOCK=${LOCK:-$HOME/.v2_chain_watchdog.lock}

# flock, never pgrep -f -- pgrep matches its own command line and a watchdog that sees itself
# never starts.
exec 9>"$LOCK"
flock -n 9 || { echo "another watchdog holds $LOCK; exiting"; exit 0; }

echo "=== chain watchdog start $(date '+%F %T')  interval=${INTERVAL}s rounds=$MAX_ROUNDS ==="
for r in $(seq 1 "$MAX_ROUNDS"); do
  n=$(squeue -u "$USER" -h -o "%i" 2>/dev/null | wc -l)
  echo "--- round $r/$MAX_ROUNDS  $(date '+%F %T')  submitted=$n ---"
  bash experiments/benchmark_v2/tools/extend_chains_v2.sh --links 1 --execute 2>&1 \
    | grep -vE "^ .*-- COMPLETE, no link$" | tail -12
  # stop early once every arm-B cell has banked its budget
  if ! bash experiments/benchmark_v2/tools/extend_chains_v2.sh --links 1 2>/dev/null \
       | grep -qE "WOULD sbatch|not active"; then
    echo "=== every arm-B cell is complete; watchdog exiting at round $r ==="; break
  fi
  sleep "$INTERVAL"
done
echo "=== chain watchdog done $(date '+%F %T') ==="
