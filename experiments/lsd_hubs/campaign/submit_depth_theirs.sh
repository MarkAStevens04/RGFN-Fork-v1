#!/bin/bash
#SBATCH --job-name=depth_theirs
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# The competitor's DEPTH rungs for DRD2 and ClpP -- the last gap in the shared-axis depth figure.
# Pure SPARROW re-solves over ALREADY-CACHED MultiAiZ routes, so this is minutes per cell, not hours.
#
# POOL CHOICE IS GATE-DRIVEN, NOT CONVENIENCE. Our own zinc_axis runs for these targets used the
# gates in matrix16/targets.py (DRD2 0.345, ClpP -9.1), so only pools BUILT at those gates are
# comparable. That rules out every `_pruned_N1xx/N2xx` and `_N3xx` pool (built at the superseded
# DRD2 0.5 / ClpP -8.0) even though several of them hold MORE routed molecules -- a bigger pool at
# the wrong bar describes a different population. depth_pruned_frontier.py re-checks this and aborts
# on a mismatch rather than quietly mixing two bars.
#
# CLPP GETS BOTH POOL VARIANTS, ON PURPOSE. Its post-NaN-fix stage-2 PRUNED pools are genuinely small
# (N115-N143, 86-104 routed) so every rung will be pool-limited; the NAIVE stage-2 pools hold 388-423
# routed. Pruned is the variant the rest of the campaign uses and stays primary; naive is the more
# generous reading for them and costs minutes. Report the pair, and flag the pruned cells as
# pool-limited rather than scoring them as cost wins.
#
# DRD2 seed 42 is NOT included: its gate-matched pool routes 50 of 500 (the catalogue-coverage
# failure from entry [075]) and its five rungs are already on disk from that pool.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
P=$SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools
R=$SCRATCH/rgfn_runs/lsdflow_sparrow/results
source ~/bin/rgfn-smoke-env.sh
RC=0

run_cell () {            # $1 pool dir, $2 gate, $3 higher-is-better
  local D=$1 G=$2 HIB=$3 TAG
  TAG=$(basename "$D")
  [ -s "$D/multiaiz_routes.json" ] || { echo "SKIP $TAG (no cached routes)"; return 0; }
  for K in 1 2 3 4 5; do
    local O="$R/${TAG}_depth${K}_select"
    if [ -s "$O/select_frontier.csv" ]; then echo "  have $TAG d$K"; continue; fi
    echo ""; echo "######## $TAG  min-steps $K  (gate $G) ########"
    python experiments/lsd_hubs/campaign/depth_pruned_frontier.py \
        --pool-dir "$D" --out-dir "$O" --gate "$G" --higher-is-better "$HIB" \
        --min-steps "$K" --budgets 100 --max-seconds 3600 || RC=1
  done
}

echo "=== DRD2 (gate 0.345), gate-matched stage2fix pruned pools ==="
run_cell "$P/s3gfn_drd2_seed43_stage2fix_pruned_N500" 0.345 true
run_cell "$P/s3gfn_drd2_seed44_stage2fix_pruned_N394" 0.345 true

echo ""; echo "=== ClpP (gate -9.1), PRUNED = primary (expect pool-limited) ==="
for S in 42 43 44; do
  run_cell "$(ls -d $P/s3gfn_clpp_seed${S}_stage2_pruned_N* 2>/dev/null | head -1)" -9.1 false
done

echo ""; echo "=== ClpP (gate -9.1), NAIVE = generous variant, 388-423 routed ==="
for S in 42 43 44; do
  run_cell "$P/s3gfn_clpp_seed${S}_stage2_N500" -9.1 false
done

echo ""; echo "DEPTH-THEIRS rc=$RC"; exit "$RC"
