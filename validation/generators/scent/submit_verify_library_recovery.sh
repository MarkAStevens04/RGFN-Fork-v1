#!/bin/bash
#SBATCH --job-name=scentverify
#SBATCH --partition=debug
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
#
# Run the SCENT resume verification on a COMPUTE node rather than a login node.
#
#   OUT_ROOT=$SCRATCH/rgfn_runs/smoke/agentA_rngN sbatch \
#       validation/generators/scent/submit_verify_library_recovery.sh
#
# WHY THIS EXISTS RATHER THAN JUST RUNNING IT ON THE LOGIN NODE. The harness trains SCENT twice, and
# a login node cannot give that reliably. Both failures were environmental and neither looked it:
#
#   * torch.cuda.OutOfMemoryError trying to allocate 20 MiB, because the shared login GPU was at
#     32.5 GB of 41 GB from someone else's work. It comes and goes -- minutes later the same GPU
#     had 36 GB free -- so a retry can "fix" it and teach you the wrong lesson.
#   * OpenBLAS "blas_thread_init: pthread_create failed for thread 17 of 64" against RLIMIT_NPROC,
#     which counts THREADS and is 1024 per user. Our own Claude sessions hold ~490 of them, so a
#     run with 59 processes was still over the line. That reads exactly like a code bug.
#
# A compute node gives a dedicated GPU and its own process budget, so a FAIL here is about the
# mechanism under test rather than about who else was on the node.
set -uo pipefail

OUT_ROOT=${OUT_ROOT:?set OUT_ROOT to an isolated smoke dir}
ITERATIONS=${ITERATIONS:-8}
EVERY=${EVERY:-2}
VALID_EVERY=${VALID_EVERY:-2}

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "FATAL: cannot cd to '$REPO'" >&2; exit 1; }

# Sources the env AND the QuickVina2/CUDA library paths; without it the ingest subprocess dies on
# libnvrtc after training has already succeeded, which exits non-zero for a reason unrelated to the test.
# shellcheck disable=SC1090
source ~/bin/rgfn-smoke-env.sh

export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/.cache/xdg}
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
         "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# Kept from the login-node runs: harmless on a compute node, and it means the two environments
# differ in as few ways as possible when comparing results.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Per-iteration RNG fingerprints. The verdict alone cannot say WHERE two runs parted; these can.
export SCENT_RNG_DEBUG=1

echo "host=$(hostname) out_root=$OUT_ROOT iterations=$ITERATIONS every=$EVERY valid_every=$VALID_EVERY"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true

conda run --no-capture-output -n scent python \
    validation/generators/scent/verify_library_recovery.py \
    --out-root "$OUT_ROOT" --iterations "$ITERATIONS" --every "$EVERY" --valid-every "$VALID_EVERY"
RC=$?
echo "[verify] exit=$RC"
exit $RC
