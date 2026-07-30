#!/bin/bash
#SBATCH --job-name=clone_smoke
#SBATCH --time=00:40:00
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Two-step training smoke for ONE baseline, used to prove a re-created external/ clone actually
# works before the campaign queue reaches it (docs/REFACTOR_LOG.md 2026-07-30 incident).
#
# What it is really testing: that the generator can build its env, resolve its package out of the
# restored clone, and load the sEH proxy weights from the IN-CLONE cache. Compute nodes have no
# internet, so a missing cache surfaces here as a ConnectTimeout on github.com rather than as a
# failed 3-day campaign link.
#
# Writes to an ISOLATED --run-dir. It must never share the campaign's
# $SCRATCH/rgfn_runs/experiments/fixed_reward/<gen>_<sys>_5k/seed<N> path: those links RESUME from
# whatever checkpoint they find, so a 2-step smoke checkpoint left there would corrupt a real run.
#
#   GEN=fraggfn SYSTEM=seh sbatch experiments/fixed_reward/scale5k/smoke_after_clone_restore.sh
#
# debug QoS allows ONE job at a time — submit these serially.

set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cd "$REPO" || exit 2

GEN=${GEN:?set GEN=<fraggfn|rxnflow>}
SYSTEM=${SYSTEM:-seh}
STEPS=${STEPS:-2}
SAMPLES=${SAMPLES:-8}
OUT=${OUT:-$SCRATCH/rgfn_runs/smoke_clone_restore/${GEN}_${SYSTEM}}

case "$GEN" in
  fraggfn) ENVNAME=fraggfn; RUNNER=validation/generators/fraggfn/run_fraggfn_fixed.py
           CFG=validation/configs/fraggfn_${SYSTEM}_fixed_5k.yaml ;;
  rxnflow) ENVNAME=rxnflow; RUNNER=validation/generators/rxnflow/run_rxnflow_fixed.py
           CFG=validation/configs/rxnflow_${SYSTEM}_fixed_stdlib_5k.yaml ;;
  *) echo "FATAL: unknown GEN '$GEN'"; exit 2 ;;
esac

export WANDB_MODE=offline
export TORCH_HOME=$SCRATCH/.cache/torch
export HF_HOME=$SCRATCH/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$OUT" "$TORCH_HOME" "$HF_HOME"

module load cuda/11.8.0 2>/dev/null
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

echo "host=$(hostname)"; nvidia-smi -L
echo "GEN=$GEN SYSTEM=$SYSTEM CFG=$CFG STEPS=$STEPS OUT=$OUT"
echo "clone: $(ls -d external/* 2>/dev/null | tr '\n' ' ')"

conda run --no-capture-output -n "$ENVNAME" python "$RUNNER" \
    --cfg "$CFG" --seed 42 --run-dir "$OUT" \
    --n-train-steps "$STEPS" --n-samples "$SAMPLES"
RC=$?

echo "----"
[ $RC -eq 0 ] && echo "SMOKE PASS ($GEN/$SYSTEM): clone + env + in-clone proxy cache all work" \
              || echo "SMOKE FAIL ($GEN/$SYSTEM) exit $RC"
exit $RC
