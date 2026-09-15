#!/bin/bash
#SBATCH --job-name=clean_pool
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# ONE clean pool per (target, seed), large enough that EVERY depth rung 1..5 can draw a full 500
# modes from it -- the fix for figure 2's central unfairness.
#
# THE UNFAIRNESS BEING FIXED. Until now the competitor got a 500-mode pool built with NO depth
# requirement, which the depth filter then SHRANK (ClpP seed 42: 138 tau-distinct -> 80 routed -> 4
# at depth>=5). Our side applied the same filter to a 644k-child enumeration that could absorb it.
# So the constraint ate their pool and not ours. Here they get a pool big enough that the depth-5
# subset alone is ~500, i.e. the constraint is SATISFIED by the pool rather than served from its
# remains.
#
# WHERE THE COST ACTUALLY IS. Sampling and the tau-distinct walk are ~free, so a big --target-above-gate
# is cheap insurance. The ONLY expensive stage is MultiAiZ at ~27 s/molecule, and that scales with
# POOL SIZE. Sizing: depth>=5 holds ~23% of routed, routing solves ~70% of a pool, so a 4,500 pool
# -> ~3,150 routed -> ~725 at depth>=5, i.e. 1.45x the 500 target. Estimated ~34 h against a 72 h
# wall: 2.1x margin.
#
# WHY NOT CHUNK THE ROUTING. MultiAiZ writes its artifact only at the END, so a walltime kill loses
# the run -- chunking would make it resumable. We do not, because its whole value is SET-BASED
# discovery of shared intermediates, and chunking would shrink the set and hand the competitor worse
# routes. The project's stance is that they get their best planner. Hence the 3-day wall instead.
# Sampling and pool-building DO persist, so a resubmit after a kill redoes only the routing.
#
# NOT RUN FOR ClpP, deliberately: its tau-distinct yield is 1.4-1.7% and its walk already scanned its
# ENTIRE draw (8,144 of 8,144) to find 115-143 modes. That is generator mode-collapse, not a routing
# shortfall -- more hours would buy more copies of the same 138 molecules.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

TARGET=${TARGET:?set TARGET (seh|drd2)}
SEED=${SEED:?set SEED}
ABOVE=${ABOVE:-10000}          # cheap: buys tau-distinct headroom
POOL=${POOL:-4500}             # the cost driver: this is what gets routed
MAXSAMP=${MAXSAMP:-150000}
read -r GATE HIB <<EOF
$(source ~/bin/rgfn-smoke-env.sh >/dev/null 2>&1; python - "$TARGET" <<'PYG'
import sys; sys.path.insert(0,"experiments/lsd_hubs/matrix16")
from targets import TARGETS
t=TARGETS[sys.argv[1]]; print(t.mode_reward_threshold, "true" if t.higher_is_better else "false")
PYG
)
EOF
[ -n "${GATE:-}" ] || { echo "FATAL: no gate for $TARGET" >&2; exit 1; }
# S3-GFN names every checkpoint s3gfn_seh-* regardless of target; only the CELL dir disambiguates.
RUN=$SCRATCH/rgfn_runs/experiments/fixed_reward/s3gfn_${TARGET}/seed${SEED}
BIG=$SCRATCH/rgfn_runs/clean_depth/s3gfn_${TARGET}_seed${SEED}
POOLROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools
TAG=s3gfn_${TARGET}_seed${SEED}_CLEAN
RES=$SCRATCH/rgfn_runs/lsdflow_sparrow/results
mkdir -p "$BIG"
echo "host=$(hostname) target=$TARGET seed=$SEED gate=$GATE hib=$HIB above=$ABOVE pool=$POOL"
[ -d "$RUN" ] || { echo "FATAL: no run dir $RUN" >&2; exit 1; }
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

# --- 1. sample (cheap, persists across a resubmit) -----------------------------------------------
CANDS=$BIG/fixed_reward/candidates/candidates.csv
if [ ! -s "$CANDS" ]; then
  echo "=== [1/4] enlarging the draw to $ABOVE above-gate (frozen checkpoint) ==="
  LOWER=""; [ "$HIB" = "false" ] && LOWER="--lower-is-better"
  conda run --no-capture-output -n s3gfn python experiments/lsd_hubs/campaign/s3gfn_sample_more.py \
      --run-dir "$RUN" --target-above-gate "$ABOVE" --gate "$GATE" $LOWER \
      --max-samples "$MAXSAMP" --out-dir "$BIG" || { echo "SAMPLE FAILED" >&2; exit 1; }
else
  echo "=== [1/4] candidates present, skipping sampling ==="
fi
echo "  candidates rows: $(($(wc -l < "$CANDS") - 1))"

# --- 2. pool (cheap, persists) -------------------------------------------------------------------
source ~/bin/rgfn-smoke-env.sh
echo "=== [2/4] tau-distinct pool of up to $POOL ==="
LOWER=""; [ "$HIB" = "false" ] && LOWER="--lower-is-better"
python experiments/lsd_hubs/campaign/build_s3gfn_pools.py \
    --candidates "$CANDS" --out-root "$POOLROOT" --tag "$TAG" --sizes "$POOL" \
    --gate "$GATE" $LOWER --pruned --cutoff 0.5 || { echo "POOL FAILED" >&2; exit 1; }
D=$(ls -d $POOLROOT/${TAG}_N* 2>/dev/null | head -1)
[ -n "$D" ] || { echo "FATAL: no pool dir" >&2; exit 1; }
echo "  pool: $(basename "$D") ($(wc -l < "$D/pool.smi") molecules)"

# --- 3. MultiAiZ: the expensive stage ------------------------------------------------------------
echo "=== [3/4] MultiAiZ over the whole pool ==="
POOL_DIR="$D" bash experiments/lsd_hubs/campaign/submit_multiaiz_discover.sh || exit 1
[ -s "$D/multiaiz_routes.json" ] || { echo "FATAL: no routes" >&2; exit 1; }

# --- 4. read every rung + the budget sweep off the one routed pool --------------------------------
echo "=== [4/4] depth rungs + budget sweep ==="
RC=0
for K in 1 2 3 4 5; do
  python experiments/lsd_hubs/campaign/depth_pruned_frontier.py --pool-dir "$D" \
    --out-dir "$RES/${TAG}_depth${K}_select" --gate "$GATE" --higher-is-better "$HIB" \
    --min-steps "$K" --budgets 100 --max-seconds 3600 || RC=1
done
python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
  --routes "$D/multiaiz_routes.json" --pool "$D/pool_scores.csv" --route-source multiaiz \
  --gate "$GATE" --higher-is-better "$HIB" --cutoff 0.5 \
  --budgets 50,100,150,200,300,400,500,600,800,1000,1500,2000,3000,5000 \
  --max-seconds 5400 --out-dir "$RES/budget_scale_s3gfn_${TARGET}CLEAN_seed${SEED}" \
  --tag "budget_clean_${TARGET}_s${SEED}" || RC=1
echo ""; echo "CLEAN-POOL rc=$RC"; exit "$RC"
