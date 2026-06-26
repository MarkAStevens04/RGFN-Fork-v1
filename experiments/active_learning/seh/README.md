# sEH active-learning loop — GPU-docking oracle (experiment 010)

**Tooling for the sEH active-learning run.** Full write-up:
[`Logs/010_seh-gpu-docking-oracle.md`](../../../Logs/010_seh-gpu-docking-oracle.md).

## What this is

The active-learning restructuring of upstream's `configs/rgfn_seh_docking.gin`
(`[bengio2021gflownet]` Alg. 1): a cheap learned MPNN proxy `M` is the in-loop
reward RGFN trains against, and a **real QuickVina2-GPU sEH docking oracle** `O`
is queried only on each round's query batch. So docking never enters the inner
loop — the GFN trains on the plain `reaction` env.

The oracle itself is production code and lives under `glue/` (it ships):
- `glue/oracles/docking_seh_oracle.py` — `DockingSEHOracle`, a thin SMILES-facing
  wrapper over upstream `DockingMoleculeProxy` (the docking is byte-for-byte
  RGFN's own QuickVina2-GPU path).
- `configs/glue/active_learning_seh.gin` — wires proxy + oracle + dataset + loop.
- `scripts/active_learning.py` — the generic loop entry point (shared with 6TD3).

This directory holds the **sEH-specific run tooling** around that oracle: seed
generation, a quick live validation, and the HPC run script. Timestamped run
outputs land here too (git-ignored).

## Files

**Scripts**
- `make_seh_seed.py` — builds the seed dataset `D_0`: samples unique valid
  molecules from the *untrained* RGFN forward policy (the same sampler the loop
  uses each round, so `D_0` is in-distribution), docks them with `DockingSEHOracle`,
  and writes `smiles,label` to `experiments/active_learning/seh/seed_seh.csv`.
  Reports average docking time per molecule.
- `validate_seh_oracle.py` — quick correctness/speed smoke: docks a few known
  molecules + one invalid SMILES, asserts finite negative Vina energies and a
  `nan` for the invalid input. Cross-checks aspirin against the known −6.3.
- `submit_al_seh.sh` — Balam **compute-node** SLURM script: regenerates a larger
  `D_0` on `$SCRATCH`, then runs the multi-round loop via a gin overlay that
  repoints `seed_csv` at that seed. (Compute node, not login: the login node's
  per-process CPU-time cap would SIGXCPU-kill long conformer prep.)

## How to run

All commands from the **repo root**, in the `rgfn` conda env, after:

```bash
module load cuda/11.8.0                                      # dgl graphbolt + vina OpenCL
export LD_LIBRARY_PATH=$SCRATCH/vina_gpu/boost/lib:$LD_LIBRARY_PATH   # QuickVina2-GPU boost libs
```

```bash
# quick validation (a few molecules)
python experiments/active_learning/seh/validate_seh_oracle.py

# (re)generate the seed D_0
python experiments/active_learning/seh/make_seh_seed.py \
    --cfg configs/glue/active_learning_seh.gin --n 300

# full multi-round loop on a compute node
sbatch experiments/active_learning/seh/submit_al_seh.sh
```

## Status

Oracle created and **validated live** on the balam-login01 A100 (see Logs/010):
QuickVina2-GPU sEH docking, aspirin −6.3 reproduced, fast. The **full multi-round
loop has not yet been run end-to-end** — pending a compute node
(`submit_al_seh.sh` is ready).
