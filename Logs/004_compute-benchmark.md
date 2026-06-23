# 004 — Compute benchmark: login node vs Balam debug_full_node (4× A100)

**Date:** 2026-06-18
**Objective:** Quantify the speedup from moving the docking runs onto a full Balam debug node, and
decompose *where* the speedup comes from (parallelism vs. multi-ligand batching).

## Configurations compared
- **A — sequential (this node):** login node, 1 GPU, one gnina invocation **per molecule** (CNN
  model reloads every call). The original per-molecule `dock_6td3_batch.py` / `batch_anchor_dock.py`
  design — both since removed (consolidated into the batching `dock_cluster*.py` drivers).
- **B — single-GPU batched+parallel (this node):** login node, 1 GPU, 4 worker processes, each
  docking a **multi-ligand shard in one gnina call** (model loads once per shard). `dock_cluster.py`
  with `N_GPU=1 PROCS_PER_GPU=4`.
- **C — full node (cluster):** `debug_full_node`, 4× A100, 16 workers (4/GPU), batched. The real runs
  (jobs 69271 / 69272).

Method: A and B timed on the login node over a fixed 24-molecule subset (measured s/mol); C from the
SLURM logs of the full runs. Sequential full-run totals are *extrapolated* from the measured per-mol
rate (the local 6TD3 run was paused; CRBN's sequential rate is from the earlier completed Pass-B
local run, 368 mols in 101 min).

## Results

### 6TD3 — global box docking (exh 16), 408 molecules
| config | dock rate | wall (408 mols) |
|---|---|---|
| A sequential (1 GPU) | 12.5 s/mol | ~88 min (extrapolated) |
| B 1 GPU / 4 proc / batched | 2.5 s/mol | ~17 min |
| **C 4 GPU / 16 proc (cluster)** | **1.57 s/mol** | **13.4 min (measured)** |

- **Total speedup A→C ≈ 6.6×** (dock phase 7.9×).
- Decomposition: **batching+4-proc on ONE GPU already gives 5.0×** (A→B); adding the other 3 GPUs
  only adds 1.6× (B→C). → 6TD3 global docking is **CPU-bound** (the Vina Monte-Carlo search), so
  extra GPUs help little; the win is batching + 64 cores.

### CRBN — warhead-anchored `--minimize`, 100 confs/mol, 437 molecules
| config | dock rate | wall (437 mols) |
|---|---|---|
| A sequential (1 GPU, measured on prior 368-mol run) | 16.1 s/mol | ~120 min (extrapolated) |
| **C 4 GPU / 16 proc (cluster)** | **0.32 s/mol** | **5.8 min (measured)** |

- **Total speedup A→C ≈ 21×** (dock phase **51×**).
- Much bigger than 6TD3 because the anchored workload is **model-reload-bound**: locally each
  molecule reloads the CNN to minimize its 100 confs. Batching a whole shard (~2700 confs) into one
  gnina call amortizes that reload ~27×, on top of 16-way parallelism.
- **Amdahl flip:** after the 51× dock speedup, conformer **embedding (3.5 min) is now 60% of the
  5.8-min wall** — the new bottleneck. Further speedup needs faster/parallel conformer generation,
  not more GPUs.

## Takeaways
1. **Multi-ligand batching is the single biggest lever** (amortizes the per-call CNN model load),
   especially for many-conf `--minimize` workloads (CRBN: ~27× of the win). Always batch.
2. **The docking search is CPU-bound, not GPU-bound** — 4 GPUs gave only ~1.6× over 1 GPU for global
   docking. Cores + batching matter more than GPU count for this workload.
3. Net: the full node turned a ~2-hour CRBN run into **5.8 min** and a ~1.5-hour 6TD3 run into
   **13.4 min**. Both full cluster runs together = ~19 min of wall time.
4. For the CRBN pipeline, the next optimization target is **conformer embedding** (now the Amdahl
   floor), e.g. cheaper/parallel ETKDG or fewer confs.

## Files & where results live
- Benchmark configs: A = the now-removed per-molecule sequential runners; B/C =
  `docking_6td3/dock_cluster.py` / `docking_gnina/dock_cluster_crbn.py` (which run sequentially with
  `N_GPU=1 PROCS_PER_GPU=1`, batched).
- Cluster timings from SLURM logs: `/scratch/markymoo/rgfn_runs/dock6td3-69271.out`,
  `dockcrbn-69272.out`. (Benchmark A/B subset runs were transient, not saved.)
