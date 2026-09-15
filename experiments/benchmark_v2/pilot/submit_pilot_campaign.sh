#!/bin/bash
#SBATCH --job-name=pilotB_campaign
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# ARM-A VIABILITY PILOT, stage 3 — enumerate the arm-A hub set and price it.
#
# WHAT THIS TURNS FROM PREDICTION INTO MEASUREMENT. Stage 2 showed the arm-A flow field is well
# separated (top10-median 4.28 nats against v1's 3.76) and far better evidenced (median 101 estimates
# per walked hub against 10) -- so it is NOT noise. But it ranks almost exclusively DEPTH-3 hubs
# ({2:3, 3:37} in the top 40) where the 320,000-call field ranks depth-1 ({0:2, 1:31, 2:7}). A depth-3
# hub costs 3 reactions before its first child, against 1 for depth-1 and 0 for a bought depth-0
# scaffold, so the prediction is a materially worse reactions/mode. This measures it.
#
# WHY ONLY 64 HUBS. At R=100 the campaign walks ~5. 64 is a generous envelope and keeps the
# enumeration affordable; the hub set is flow-descending so the first 64 are the ones a 100-reaction
# budget could ever reach.
#
# ALSO THE FIRST REAL EXERCISE OF `depth_mix`. Agent C could not run run_campaign end to end (the
# sandbox refuses the env helper), so its depth-mix block is unit-tested by AST extraction, never
# executed. This job is where it either populates or does not -- checked explicitly at the end
# rather than assumed.
#
# Usage:
#   HUBS=<...>/pool_all/hubs.csv CKPT=<...>/last_gfn.pt OUT=<...>/campaign_all \
#     sbatch experiments/benchmark_v2/pilot/submit_pilot_campaign.sh
set -uo pipefail

GEN=${GEN:-scent}
SYSTEM=${SYSTEM:-seh}
SEED=${SEED:-42}
HUBS=${HUBS:?set HUBS to a hubs.csv}
CKPT=${CKPT:?set CKPT to the arm-A checkpoint}
SAMPLE=${SAMPLE:?set SAMPLE to the stage-2 sample dir}
OUT=${OUT:?set OUT}
GUIDANCE=${GUIDANCE:-}
N_HUBS=${N_HUBS:-64}
GATE=${GATE:-5.68}
PYBIN=${PYBIN:-/home/markymoo/miniconda3/envs/rgfn/bin/python}
ENUM_MAX=${ENUM_MAX:-4000}

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "FATAL: cannot cd to '$REPO'"; exit 1; }
[ -f experiments/lsd_hubs/campaign/run_campaign.py ] || { echo "FATAL: not an RGFN-Fork checkout"; exit 1; }
[ -f "$HUBS" ] || { echo "FATAL: no hubs at $HUBS"; exit 1; }

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export XDG_CACHE_HOME=$SCRATCH/.cache/xdg
mkdir -p "$OUT/enum" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

head -n $((N_HUBS + 1)) "$HUBS" > "$OUT/hubs_head.csv"
echo "host=$(hostname) gen=$GEN hubs=$(( $(wc -l < "$OUT/hubs_head.csv") - 1 )) gate=$GATE"
nvidia-smi -L || true

echo "=== [1/2] ENUMERATE ==="
T0=$(date +%s)
GARG=(); [ -n "$GUIDANCE" ] && GARG=(--guidance "$GUIDANCE")
# --no-freeze and --guidance are SCENT-ONLY flags; rgfn_worker.py rejects them (argparse exit 2).
# The GEN test is load-bearing, not defensive: RGFN has no dynamic library, so it NEVER has an
# additional_fragments/ directory, and a snapshot test alone therefore fires on every RGFN run.
if [ "$GEN" = scent ] && \
   [ -z "$(ls "$(dirname "$(dirname "$CKPT")")"/additional_fragments/fragments_*.json 2>/dev/null)" ]; then
    echo "[pilot-campaign] no additional_fragments snapshot -> --no-freeze (418 base library)"
    GARG+=(--no-freeze)
fi
# Resolve CFG and CONDA_ENV from the manifest, never from a naming pattern. There ISN'T one:
# fixed_reward_6td3_5k.gin sits beside fixed_reward_drd2_stdlib_5k.gin, and
# rxnflow_6td3_docking_fixed_5k.yaml beside rxnflow_seh_fixed_stdlib_5k.yaml. An earlier version of
# this script built the RGFN path as configs/glue/fixed_reward_${SYSTEM}_proxy_stdlib_5k.gin, which
# resolves for sEH ONLY -- clpp and 6td3b both point at files that do not exist. Grep the specific
# keys rather than eval'ing the whole emit: submit_cell.sh's `eval "$SPEC"` is what clobbered env
# overrides in v1, and this script has its own.
manifest_val() {  # manifest_val KEY -> value, or empty
    python experiments/benchmark_v2/tools/manifest.py --emit "$GEN" "$SYSTEM" "$SEED" 2>/dev/null \
        | sed -n "s/^$1=//p" | head -1
}
CFG=$(manifest_val CFG)
[ -n "$CFG" ] || { echo "FATAL: manifest emitted no CFG for $GEN/$SYSTEM/$SEED"; exit 2; }
[ -f "$CFG" ] || { echo "FATAL: manifest CFG does not exist on disk: $CFG"; exit 2; }
ENVN=$(manifest_val CONDA_ENV); ENVN=${ENVN:-$GEN}
echo "[pilot] cfg=$CFG env=$ENVN (from manifest)"
conda run --no-capture-output -n "$ENVN" python validation/lsdflow/adapters/workers/${GEN}_worker.py \
    --mode enumerate --config "$CFG" \
    --checkpoint "$CKPT" --reward-name "$SYSTEM" --model-name "$GEN" --seed "$SEED" \
    --hubs-file "$OUT/hubs_head.csv" --enum-max-children "$ENUM_MAX" \
    --run-dir "$OUT/rundir" --out-dir "$OUT/enum" "${GARG[@]}"
RC=$?
ENUM_S=$(( $(date +%s) - T0 ))
echo "[pilot-campaign] enumerate exit=$RC wall=${ENUM_S}s"
[ "$RC" -ne 0 ] && exit $RC

echo "=== [2/2] CAMPAIGN (--pool all hub set, free_frag, prebuild-k 0) ==="
T1=$(date +%s)
/home/markymoo/miniconda3/envs/rgfn/bin/python experiments/lsd_hubs/campaign/run_campaign.py \
    --analysis-dir "$SAMPLE" --enum-children "$OUT/enum/enum_children.json" \
    --reward-threshold "$GATE" --similarity 0.5 --higher-is-better true \
    --budget-reactions 100 --budget-modes 300 \
    --child-policy free_frag --prebuild-k 0 \
    --enum-timings "$OUT/enum/enum_timings.json" \
    --tag "pilot_armA_${GEN}_${SYSTEM}" --out-dir "$OUT/campaign"
RC=$?
echo "[pilot-campaign] campaign exit=$RC wall=$(( $(date +%s) - T1 ))s  enumerate=${ENUM_S}s"

# C's depth_mix block has never been EXECUTED (its sandbox refused the env helper). Say plainly
# whether it populated, rather than leaving the next reader to discover it did not.
echo "=== required per-cell outputs present? ==="
# WHY THIS ASSERTS RATHER THAN PRINTS. The first version of this block printed "<ABSENT>" for a key
# that was present, because it looked for summary["hub_batching"]["depth_mix"] when the nesting is
# summary["depth_mix"]["hub_batching"] -- the other way round. A checker that reports the OPPOSITE of
# the truth is worse than no checker, and this one is the version that runs on every cell. So it now
# fails loudly on a shape it does not recognise instead of formatting a wrong answer, and it
# distinguishes MISSING (key absent) from NULL (key present, value None) -- which are different
# defects: absent means the writer never ran, null means it ran and could not resolve its input.
"$PYBIN" - "$OUT/campaign/summary.json" <<'PYEOF'
import json, sys

try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    sys.exit(f"FATAL: cannot read summary.json: {exc}")

MISSING = object()
bad = []

# benchmark_v2/README.md requires exactly two things per cell.
dm = d.get("depth_mix", MISSING)
if dm is MISSING:
    bad.append("depth_mix ABSENT -- run_campaign did not emit it at all")
elif not isinstance(dm, dict):
    bad.append(f"depth_mix has unexpected type {type(dm).__name__}; expected a dict keyed by arm")
else:
    for arm in ("hub_batching", "best_candidate"):
        a = dm.get(arm)
        if not isinstance(a, dict):
            bad.append(f"depth_mix[{arm!r}] missing or not a dict")
            continue
        frac, hist = a.get("depth0_mode_frac"), a.get("modes_by_hub_depth")
        print(f"  depth_mix[{arm}]  depth0_mode_frac={frac}  by_depth={hist} "
              f"n_unmapped={a.get('n_unmapped')}")
        if frac is None:
            bad.append(f"depth_mix[{arm}].depth0_mode_frac is null")

for k in ("walked_depth_hist", "hub_pool", "min_hub_depth", "max_hub_depth",
          "min_synth_depth", "n_promoted_fragments"):
    v = d.get(k, MISSING)
    state = "ABSENT" if v is MISSING else ("NULL" if v is None else v)
    print(f"  {k:<22} = {state}")
    # walked_depth_hist is the second README requirement; the rest are provenance.
    if k == "walked_depth_hist" and (v is MISSING or v is None):
        bad.append("walked_depth_hist is not populated -- required per cell by benchmark_v2/README")

if bad:
    print("\nFAILED required-output checks:")
    for b in bad:
        print(f"  - {b}")
    sys.exit(1)
print("\nall required per-cell outputs present")
PYEOF
CHECK_RC=$?
echo "[pilot-campaign] required-output check exit=$CHECK_RC"
exit $RC
