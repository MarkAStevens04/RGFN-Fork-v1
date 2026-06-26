#!/bin/bash
#SBATCH --job-name=al_seh
#SBATCH --time=12:00:00                       # 5 rounds x (1000-iter GFN train + 200-mol GPU dock)
#SBATCH --partition=compute                   # 1-GPU job -> the regular (non full_node) partition
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Multi-round sEH active-learning loop on a Balam COMPUTE node, using the real
# QuickVina2-GPU docking oracle (DockingSEHOracle) and the learned MPNN proxy.
# See configs/glue/active_learning_seh.gin. Submit with:
#   sbatch experiments/active_learning/seh/submit_al_seh.sh
#
# Why a compute node: the login node has a per-process CPU-time rlimit (~3600s)
# that would SIGXCPU-kill long conformer-prep/docking; compute nodes have none.

set -euo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

# --- Caches + run outputs: redirect everything away from the read-only $HOME ---
# On a compute node $HOME is read-only, so run dirs / per-round dataset CSVs /
# checkpoints / wandb / the generated seed must land on $SCRATCH.
export WANDB_PROJECT=rgfn
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export WANDB_DATA_DIR=$SCRATCH/.cache/wandb
export WANDB_DIR=$SCRATCH/wandb
export WANDB_MODE=offline
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export PIP_CACHE_DIR=$SCRATCH/.cache/pip

AL_ROOT_DIR=$SCRATCH/rgfn_runs/experiments
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" \
        "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$AL_ROOT_DIR"

module load cuda/11.8.0                 # CUDA-11.8: dgl graphbolt AND QuickVina2-GPU OpenCL runtime
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn

# QuickVina2-GPU-2.1 needs its boost runtime libs on LD_LIBRARY_PATH (the build
# lives on $SCRATCH; quickvina_dir symlink -> $SCRATCH/vina_gpu/Vina-GPU-2.1).
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}

export PYTHONUNBUFFERED=1
nvidia-smi

# --- Step 1: generate a fresh in-distribution seed D_0 on $SCRATCH. ------------
# The repo's committed experiments/active_learning/seh/seed_seh.csv is only a
# 36-molecule starter; for a real run we draw a larger D_0 (300) from the
# untrained RGFN policy and dock it. Writes to $SCRATCH ($HOME is read-only here).
SEED_CSV=$AL_ROOT_DIR/seed_seh_300.csv
if [ ! -f "$SEED_CSV" ]; then
    python experiments/active_learning/seh/make_seh_seed.py \
        --cfg configs/glue/active_learning_seh.gin \
        --n 300 --oversample 2.0 --seed 42 \
        --out "$SEED_CSV"
fi

# --- Step 2: a per-run gin overlay pointing the loop at the $SCRATCH seed. -----
# (active_learning.py takes only --cfg/--seed/--root-dir, so we override the seed
# path by including the base config and re-binding seed_csv. gin resolves the
# include relative to the cwd = repo root.)
OVERLAY=$AL_ROOT_DIR/active_learning_seh_run.gin
cat > "$OVERLAY" <<EOF
include 'configs/glue/active_learning_seh.gin'
OracleLabeledDataset.seed_csv = '$SEED_CSV'
EOF

# --- Step 3: run the outer active-learning loop. ------------------------------
python scripts/active_learning.py \
        --cfg "$OVERLAY" \
        --seed 42 \
        --root-dir "$AL_ROOT_DIR"
