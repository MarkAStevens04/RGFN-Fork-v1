#!/bin/bash
# extend_chains_v2.sh -- append afterany links so a cell that outruns one walltime continues.
#
# WHY THIS EXISTS, MEASURED. The 3-day walltime is the hard cap (sinfo), and the docking cells do not
# fit in it. From the live rate on 2026-09-14, distinct training molecules per hour against the
# 320,000 arm-B budget:
#
#     rxnflow/seh   57,015/h ->  0.2 d      rxnflow/clpp  3,602/h ->  3.7 d   EXCEEDS
#     rxnflow/drd2  45,790/h ->  0.3 d      scent/clpp    3,097/h ->  4.3 d   EXCEEDS
#     scent/seh     29,586/h ->  0.5 d      rgfn/clpp     2,673/h ->  5.0 d   EXCEEDS
#                                           rgfn/6td3b    2,271/h ->  5.9 d   EXCEEDS
#
# Docking is 13-25x slower per molecule. Surrogate cells finish inside one link; every docking cell
# needs two or three. Requeue=0 on these jobs, so without a chain they are simply killed at 72 h at
# 50-80% of budget.
#
# WHY A CHAIN AND NOT --requeue: a job cannot sbatch from inside a job here, so the successor must be
# queued from a LOGIN node ahead of time. `--dependency=afterany` runs the next link whether the
# previous one finished or was killed at walltime, which is exactly the "killed at 72 h" case.
#
# NO --kill-on-invalid-dep: this SLURM does not have it (sbatch rejects the option outright -- found
# by the chain smoke, which is what that smoke was for). With afterany the flag matters little, since
# afterany is satisfied by ANY termination including a walltime kill; a never-satisfiable dependency
# would instead sit PENDING with DependencyNeverSatisfied, which is visible rather than silent.
#
# WHY IT IS SAFE TO RUN A CELL TWICE: all three reaction-GFN runners RESUME -- rxnflow at
# `remaining = n_train_steps - loop._it`, scent from its checkpoint into the same run dir ("stable dir
# for auto-requeue chain links"), rgfn via start_iteration. And the arm-B stop counts distinct
# `phase=="train"` molecules across trace.csv AND its rotated .N siblings, seeded from earlier rounds,
# so the budget continues rather than restarting. A link on an already-finished cell stops immediately.
#
# VERIFIED 2026-09-14 with a 3-link smoke whose FIRST link was deliberately killed at its walltime,
# because a clean exit is not the case we need:
#     76317 TIMEOUT 00:01:19   (killed mid-sleep, never reached its own "finished" line)
#     76318 COMPLETED          started 19:57:01, immediately after 76317 died
#     76319 COMPLETED          started 19:57:04, after 76318
# Strictly sequential, no overlap, and the successor ran despite the predecessor being KILLED.
#
#   bash experiments/benchmark_v2/tools/extend_chains_v2.sh [--links N] [--execute]
set -uo pipefail
LINKS=${LINKS:-2}; EXECUTE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --links) LINKS="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    *) echo "usage: extend_chains_v2.sh [--links N] [--execute]" >&2; exit 2 ;;
  esac
done

[ -z "${SLURM_JOB_ID:-}" ] || { echo "FATAL: run from a LOGIN node -- a job cannot sbatch here." >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO" || exit 1
TRAINER=experiments/benchmark_v2/train/submit_train_v2.sh
[ -f "$TRAINER" ] || { echo "FATAL: no trainer at $TRAINER" >&2; exit 2; }
OUT_ROOT=${OUT_ROOT:-/scratch/markymoo/rgfn_runs/v2}
CAP=${CAP:-60}; HEADROOM=${HEADROOM:-2}

CUR=$(squeue -u "$USER" -h -o "%i" 2>/dev/null | wc -l)
BUDGET=$(( CAP - HEADROOM - CUR ))
echo "=== extend v2 chains @ $(date '+%F %T') ==="
echo "submitted=$CUR/$CAP -> budget=$BUDGET links; appending up to $LINKS per cell"
[ "$EXECUTE" = 1 ] || echo "(dry run -- nothing submitted; pass --execute)"

# the chain TAIL is the highest jobid carrying this cell's name, running or pending
tail_of(){ squeue -u "$USER" -h -t R,PD -o "%i|%j" 2>/dev/null \
             | awk -F'|' -v t="$1" '{gsub(/ /,"",$2)} $2==t{print $1}' | sort -n | tail -1; }

# distinct train molecules already banked, across trace.csv and its rotated siblings
banked(){ python - "$1" <<'PY' 2>/dev/null || echo 0
import csv,sys,glob,os
d=sys.argv[1]; seen=set()
for f in [os.path.join(d,"trace.csv")]+sorted(glob.glob(os.path.join(d,"trace.csv.*"))):
    if not os.path.isfile(f): continue
    try:
        with open(f,newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("phase")=="train" and r.get("smiles"): seen.add(r["smiles"])
    except Exception: pass
print(len(seen))
PY
}

APPENDED=0
while read -r tag gen tgt seed budget; do
  [ -n "$tag" ] || continue
  RUN_DIR="$OUT_ROOT/train/$tag/armb"
  have=$(banked "$RUN_DIR")
  if [ "${have:-0}" -ge "$budget" ]; then
    printf "  %-22s banked %s/%s -- COMPLETE, no link\n" "$tag" "$have" "$budget"; continue
  fi
  t=$(tail_of "v2_$tag")
  if [ -z "$t" ]; then
    printf "  %-22s banked %s/%s -- not active; launch it first with submit_grid.py\n" "$tag" "$have" "$budget"; continue
  fi
  for i in $(seq 1 "$LINKS"); do
    [ "$BUDGET" -le 0 ] && { echo "  submit budget exhausted"; break 2; }
    if [ "$EXECUTE" = 1 ]; then
      new=$(sbatch --parsable --job-name="v2_$tag" --comment="v2cells:$tag" \
              --dependency=afterany:"$t" \
              --export=ALL,OUT_ROOT="$OUT_ROOT" "$TRAINER" "$gen" "$tgt" "$seed" 2>&1) || {
                echo "  $tag: sbatch failed: $new" >&2; break; }
      printf "  %-22s link %s: %s afterany:%s\n" "$tag" "$i" "$new" "$t"; t="$new"
    else
      printf "  %-22s link %s: WOULD sbatch afterany:%s\n" "$tag" "$i" "$t"; t="(new)"
    fi
    BUDGET=$((BUDGET-1)); APPENDED=$((APPENDED+1))
  done
done < <(python - <<'PY'
import csv
rows=[]
for r in csv.DictReader(open("experiments/benchmark_v2/grid.csv")):
    if (r.get("train_plan") or "").strip()!="generate": continue
    if r["generator"] not in ("rgfn","rxnflow","scent"): continue      # only these carry an arm B
    # DOCKING ONLY. The surrogate cells finish inside one walltime (measured: rxnflow/seh 0.2 d,
    # rxnflow/drd2 0.3 d, scent/seh 0.5 d), so a link on them is a no-op that spends a submit slot a
    # docking cell needs. scale5k/extend_chains.sh says exactly this and I chained them anyway on the
    # first run -- 11 wasted links, cancelled.
    if r["target"] not in ("clpp","6td3b"): continue
    b=(r.get("arm_b_calls") or "").strip()
    if not b.isdigit(): continue
    tag=f'{r["generator"]}_{r["target"]}_s{r["seed"]}'
    rows.append((0 if r["target"]=="6td3b" else 1, 0 if r["generator"]=="rgfn" else 1,
                 tag, r["generator"], r["target"], r["seed"], b))
for _,_,tag,g,t,sd,b in sorted(rows):   # LONGEST POLE FIRST: 6td3b before clpp, rgfn before others
    print(tag, g, t, sd, b)
PY
)
echo "appended $APPENDED link(s)"
