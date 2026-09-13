#!/bin/bash
#SBATCH --job-name=v2train
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
#
# benchmark_v2 PRODUCTION TRAINER. One cell = one (generator x target x seed).
#
#   OUT_ROOT=<root> sbatch experiments/benchmark_v2/train/submit_train_v2.sh <gen> <target> <seed>
#
# NO ARM ARGUMENT AND NO N_ITERS, EVER -- that is the whole point of promoting the pilot:
#
#   * The budget lives on the ORACLE-CALL axis, so both the arm-A checkpoint and the arm-B stop are
#     placed by the TRACE COUNTER and never by arithmetic. Measured: RGFN crossed 10,000 calls at
#     iteration 83, not the 100 that "100 trajectories/iteration" predicts (v2_pilot arm_a.json),
#     and RxnFlow's dedup cache makes its effective rate ~31/iter against a nominal 64. Iteration
#     counts would give 603,000 / 320,000 / 155,000 calls for one declared budget of 320,000.
#   * A reaction-GFN cell trains ONCE, straight through to arm B, and arm A is EXTRACTED from the
#     checkpoint taken on the way past. Two runs at one seed would not agree on a docking target --
#     the reward is genuinely stochastic -- so extraction is the only way both arms describe one
#     trajectory at zero extra training compute.
#
# The caller passes no paths and no budgets: this script reads them from manifest.py, which is the
# one place that resolves a cell.
set -uo pipefail

GEN=${1:?usage: submit_train_v2.sh <generator> <target> <seed>}
TARGET=${2:?usage: submit_train_v2.sh <generator> <target> <seed>}
SEED=${3:?usage: submit_train_v2.sh <generator> <target> <seed>}

# N_ITERS is not merely unused, it is REFUSED. A caller who sets it believes the budget is a step
# count, and silently ignoring it would let that belief survive into the analysis.
if [ -n "${N_ITERS-}" ]; then
  echo "FATAL: N_ITERS is set ($N_ITERS). This trainer places the budget from the trace counter;" >&2
  echo "       an iteration count cannot express an oracle-call budget. Unset it." >&2
  exit 2
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "FATAL: cannot cd to '$REPO'" >&2; exit 1; }
[ -f experiments/lsd_hubs/matrix16/targets.py ] || { echo "FATAL: not an RGFN-Fork checkout ($REPO)" >&2; exit 1; }

# ---- resolve the cell ---------------------------------------------------------------------------
# conda base always has a python and manifest.py is stdlib-only, so the spec can be read BEFORE the
# cell's own env is activated -- which is necessary, since the spec is what says which env to use.
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
MANIFEST="experiments/benchmark_v2/tools/manifest.py"
SPEC="$(python "$MANIFEST" --emit "$GEN" "$TARGET" "$SEED" --arm a 2>&1)" || {
  echo "FATAL: manifest could not resolve $GEN/$TARGET/$SEED:" >&2; echo "$SPEC" >&2; exit 2; }
eval "$SPEC"

# ---- refusals, each keyed on an ARTIFACT rather than on a generator name -------------------------
[ "${TRAIN_PLAN:-}" = "generate" ] || {
  echo "FATAL: $CELL_TAG has train_plan='${TRAIN_PLAN:-}', not 'generate'. copy_forward owns it." >&2
  exit 2; }

[ "${FROZEN:-false}" = "false" ] || {
  echo "FATAL: $CELL_TAG is FROZEN. Re-invoking a runner against it would overwrite a trace, the" >&2
  echo "       recipes that are observable only during training, and a trained checkpoint." >&2
  exit 2; }

# An empty CFG is how a launcher trains the wrong target or dies at argparse three hours in. Every
# one of the nine runners has --cfg required=True while manifest's _CFG covers only the three
# reaction-GFNs, so the 18 competitor 6TD3-B cells resolve to CFG='' today. Keying on the EMPTY
# VALUE rather than on a generator list means this stops firing by itself the moment those configs
# are written -- and cannot stop firing for any other reason.
[ -n "${CFG:-}" ] || {
  echo "FATAL: $CELL_TAG has no training config (manifest emitted CFG='')." >&2
  echo "       Expected for the competitor 6TD3-B cells: the six configs do not exist, AND five of" >&2
  echo "       the six competitor docking bridges hardcode max(-raw/norm, 0), which maps every" >&2
  echo "       higher-is-better cnn_vs molecule to exactly 0.0 -- a flat reward, silently, for the" >&2
  echo "       whole docking budget. The configs and the sign seam must land TOGETHER." >&2
  exit 2; }
[ -f "$CFG" ] || { echo "FATAL: config '$CFG' does not exist" >&2; exit 2; }

# ---- roots --------------------------------------------------------------------------------------
# Taken from the manifest, never spelled here: two spellings of one root is how a driver wrote to
# .../benchmark_v2 while the real tree was .../v2.
SCRATCH_ROOT_DEFAULT="$(python -c "
import sys; sys.path.insert(0, 'experiments/benchmark_v2/tools')
import manifest; print(manifest.SCRATCH_ROOT)")"
ROOT="${OUT_ROOT:-$SCRATCH_ROOT_DEFAULT}"

V1_ROOT="$SCRATCH/rgfn_runs/experiments"
case "$(readlink -m "$ROOT")" in
  "$(readlink -m "$V1_ROOT")"|"$(readlink -m "$V1_ROOT")"/*)
    echo "FATAL: OUT_ROOT '$ROOT' is inside the v1 experiments tree. Refusing." >&2; exit 2 ;;
esac

# A reaction-GFN trains ONE run that yields both arms, and that run lives in the arm it trains TO.
if [ "$GEN_CLASS" = "reaction_gfn" ]; then
  TRAIN_ARM=b
else
  TRAIN_ARM=a
fi
RUN_DIR="$ROOT/train/$CELL_TAG/arm$TRAIN_ARM"
mkdir -p "$RUN_DIR"

# ---- environment ---------------------------------------------------------------------------------
module load cuda/11.8.0
export PYTHONUNBUFFERED=1
# LOAD-BEARING FOR REPRODUCIBILITY, NOT HYGIENE: --seed alone gives 377 vs 387 routes; both knobs
# give 730/730 byte-identical. submit_baseline.sh omitted this, which is why a clean s3gfn retrain
# was correctly REJECTED at verification after burning its GPU time.
export PYTHONHASHSEED=0
# $HOME is read-only on compute nodes; anything that caches there kills the job minutes in.
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/.cache/xdg}
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
         "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# THE BUDGET, handed to _trace.py through the environment so the rule lives in ONE place rather than
# in three per-runner copies. ARM_CALLS comes from the manifest, which reads grid.csv.
ARM_B_CALLS="$(python "$MANIFEST" --emit "$GEN" "$TARGET" "$SEED" --arm b 2>/dev/null \
                | sed -n 's/^ARM_CALLS=//p')"
if [ "$TRAIN_ARM" = b ]; then
  [ -n "$ARM_B_CALLS" ] || { echo "FATAL: no arm-B budget for $CELL_TAG" >&2; exit 2; }
  # ⛔ THE STOP MUST ACTUALLY BE WIRED, and this checks the CONSUMER rather than the export.
  # Exporting BENCHMARK_V2_ARM_B_CALLS proves nothing: if _trace.py does not consume it, the run
  # has NO budget stop and trains to walltime instead of to 320,000 oracle calls -- silently, with
  # every artifact looking healthy, which is the exact failure the stop exists to prevent.
  #
  # This guard exists because the launcher's readiness and the launcher's EXISTENCE came apart:
  # the driver keys on this file being present, and for a while that was the same thing as being
  # able to train. It stopped being the same thing the moment arm B needed a hook in another file.
  # A guard that reads the other file is the only kind that notices. It deletes itself -- wire the
  # hook and it stops firing, for the one reason it should.
  if ! grep -q "budget_stop\|BudgetStopper" validation/generators/_trace.py 2>/dev/null; then
    echo "FATAL: arm B requested for $CELL_TAG, but validation/generators/_trace.py does not" >&2
    echo "       consume the budget stop. validation/generators/_budget_stop.py exists and is" >&2
    echo "       tested, but nothing calls it, so this run would train to WALLTIME rather than to" >&2
    echo "       ${ARM_B_CALLS} oracle calls -- and would look completely healthy doing it." >&2
    echo "       Wire BudgetStopper into _trace.py's on_end_sampling hook, then re-submit." >&2
    exit 2
  fi
  export BENCHMARK_V2_ARM_B_CALLS="$ARM_B_CALLS"
fi

# ANY job that docks must source this -- batch jobs included. QuickVina2-GPU links against boost
# libs that live outside the conda env; omitting them makes the docker fail to start on EVERY node
# and reads exactly like a degraded GPU (it has already been misdiagnosed as one). It also supplies
# the CUDA libs without which the ingest subprocess dies on libnvrtc after training succeeds.
if [ "$REWARD_TYPE" = docking ]; then
  # shellcheck disable=SC1090
  source ~/bin/rgfn-smoke-env.sh
fi

echo "host=$(hostname) cell=$CELL_TAG class=$GEN_CLASS target=$TARGET seed=$SEED"
echo "train_arm=$TRAIN_ARM run_dir=$RUN_DIR"
echo "cfg=$CFG env=$CONDA_ENV runner=$RUNNER"
echo "arm_a_calls=$ARM_CALLS arm_b_calls=${ARM_B_CALLS:-n/a}"
nvidia-smi -L || true
T0=$(date +%s)

# ---- the persistent docking server, for docking cells only ---------------------------------------
SERVER_PID=""
if [ "$REWARD_TYPE" = docking ]; then
  export RGFN_DOCK_SOCKET="$RUN_DIR/dock.sock"
  ( source "$HOME/bin/rgfn-smoke-env.sh" >/dev/null 2>&1
    exec python -m glue.oracles.docking_server \
        --oracle "$ORACLE" --socket "$RGFN_DOCK_SOCKET" \
        --stats "$RUN_DIR/dock_server_stats.json" ) &
  SERVER_PID=$!
  trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null' EXIT
fi

# ---- train ---------------------------------------------------------------------------------------
# THE "HOW MUCH TRAINING" FLAG IS NOT UNIFORM, and passing the wrong one is an argparse exit 2 nine
# seconds in. Three of the nine do not count steps at all -- saturn/tango/synformer take --budget in
# ORACLE CALLS, which is already the right axis and is why their landed traces sit at 10,000-10,038
# while the step-driven runners land at 10,048. Where a runner counts steps we pass a NON-BINDING
# ceiling and let the trace stop decide; the ceiling exists only so a stop that never fires cannot
# run forever.
case "$GEN" in
  rgfn)
    conda activate rgfn
    python scripts/fixed_reward.py \
        --cfg "$CFG" --seed "$SEED" --root-dir "$ROOT/train" --run-name "$CELL_TAG/arm$TRAIN_ARM"
    RC=$? ;;
  scent)
    # --log-recipes: promoted-fragment routes are observable ONLY during training, so a run without
    # them can never be repaired afterwards. Default-on since 2026-07-29; passed explicitly so this
    # script states the campaign's intent rather than inheriting it.
    conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" \
        --cfg "$CFG" --seed "$SEED" --root-dir "$ROOT/train" --run-dir "$RUN_DIR" --log-recipes
    RC=$? ;;
  rxnflow)
    conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" \
        --cfg "$CFG" --seed "$SEED" --run-dir "$RUN_DIR"
    RC=$? ;;
  saturn|tango|synformer)
    conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" \
        --cfg "$CFG" --seed "$SEED" --run-dir "$RUN_DIR" --budget "$ARM_CALLS"
    RC=$? ;;
  reinvent)
    conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" \
        --cfg "$CFG" --seed "$SEED" --run-dir "$RUN_DIR"
    RC=$? ;;
  fraggfn|s3gfn)
    conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" \
        --cfg "$CFG" --seed "$SEED" --run-dir "$RUN_DIR"
    RC=$? ;;
  *) echo "FATAL: unknown generator '$GEN'" >&2; exit 2 ;;
esac
T1=$(date +%s)
echo "[v2train] $CELL_TAG exit=$RC train wall=$((T1-T0))s"

[ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }

# ---- record what this job ACTUALLY executed under -------------------------------------------------
# write_arm_meta must run from INSIDE the job with --pythonhashseed "${PYTHONHASHSEED-}": reading the
# environment afterwards records a LATER process's env and attributes it to the run, which is a
# fabricated provenance that looks identical on disk to a real one.
write_meta () {  # $1 = arm, $2 = run dir, $3 = budget
  python experiments/benchmark_v2/tools/write_arm_meta.py \
      --run-dir "$2" --generator "$GEN" --target "$TARGET" --seed "$SEED" \
      --arm "$1" --budget "$3" \
      --pythonhashseed "${PYTHONHASHSEED-}" \
      --job-id "${SLURM_JOB_ID-}" \
      --launcher experiments/benchmark_v2/train/submit_train_v2.sh
}

ARMA_DIR="$ROOT/train/$CELL_TAG/arma"
MATERIALISED=false
if [ "$TRAIN_ARM" = b ]; then
  write_meta b "$RUN_DIR" "$ARM_B_CALLS" || echo "[v2train] WARNING arm-B meta not written" >&2
  # ARM A IS MATERIALISED FROM THIS RUN, not trained again: copy the checkpoint the BudgetCheckpointer
  # took on the way past 10,000 and slice the trace at the Nth phase=="train" ROW, carrying eval rows
  # along IN POSITION so the slice is a genuine prefix of the run rather than a filtered copy.
  python experiments/benchmark_v2/train/materialise_arm_a.py \
      --from "$RUN_DIR" --to "$ARMA_DIR" --budget "$ARM_CALLS" && MATERIALISED=true
  if [ "$MATERIALISED" = true ]; then
    write_meta a "$ARMA_DIR" "$ARM_CALLS" || echo "[v2train] WARNING arm-A meta not written" >&2
  else
    echo "[v2train] WARNING arm A not materialised; the arm-A checkpoint may not have fired" >&2
  fi
else
  write_meta a "$RUN_DIR" "$ARM_CALLS" || echo "[v2train] WARNING arm-A meta not written" >&2
fi

# ---- the completion signal the login sweep gates on ------------------------------------------------
# ARM IS STATED, NOT IMPLIED BY THE DIRECTORY. arma/ can be materialised by copy+slice, so it is the
# one artifact whose directory name and provenance can legitimately disagree; the sweep must be able
# to tell a materialised arm A from a trained one without inferring it from a path.
# NO ACCEPTANCE HAPPENS HERE. This job says what it did; verify/freeze/backup run outside the
# allocation, because a job that grades its own work cannot fail.
for _arm_dir in "$RUN_DIR" "$ARMA_DIR"; do
  [ -d "$_arm_dir" ] || continue
  [ "$_arm_dir" = "$ARMA_DIR" ] && [ "$TRAIN_ARM" = b ] && [ "$MATERIALISED" != true ] && continue
  if [ "$_arm_dir" = "$ARMA_DIR" ]; then _a=a; else _a="$TRAIN_ARM"; fi
  if [ "$_a" = a ] && [ "$TRAIN_ARM" = b ]; then _mat=true; else _mat=false; fi
  python - "$_arm_dir" "$_a" "$_mat" "$RC" "$((T1-T0))" <<'PY'
import csv, json, sys
from pathlib import Path
run_dir, arm, materialised, rc, wall = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
train_rows = total_rows = 0
steps_blank = True
tp = run_dir / "trace.csv"
if tp.is_file():
    with open(tp, newline="") as fh:
        for r in csv.DictReader(fh):
            total_rows += 1
            if (r.get("phase") or "") == "train":
                train_rows += 1
            if (r.get("step") or "").strip():
                steps_blank = False
json.dump({
    "arm": arm,
    "materialised": materialised == "true",
    "exit_code": rc,
    "train_wall_s": wall,
    "n_train_rows": train_rows,
    "n_rows": total_rows,
    # A blank step column makes the arm-B stop refuse to sum across a requeue, so it is reported
    # rather than assumed: asserted here instead of trusted from a reading of the hook chain.
    "step_column_populated": not steps_blank,
    "launcher": "experiments/benchmark_v2/train/submit_train_v2.sh",
}, (run_dir / "TRAIN_DONE.json").open("w"), indent=2)
print(f"[v2train] TRAIN_DONE.json arm={arm} materialised={materialised} "
      f"train_rows={train_rows} steps={'yes' if not steps_blank else 'BLANK'}")
PY
done

exit $RC
