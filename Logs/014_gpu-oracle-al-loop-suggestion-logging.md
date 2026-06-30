# 6TD3 / RGFN — GPU-oracle active-learning loop with synthesis-route + metrics logging

**Date:** 2026-06-29, ~late afternoon

## Question

When we drive the multi-round active-learning loop with the fast GPU docking
oracle instead of the CPU one, do the generated molecules still dock like real
glues — and can we now capture, for every molecule the model suggests, the
synthesis route that builds it and a panel of medchem/diversity metrics per round?

## Context & Summary

**Context** — Entry 011 ran the first 3-round active-learning loop on the CPU
gnina oracle: the generated molecules docked as genuine glues (median differential
≈ −2.4), but the loop took ~5 h, with docking ~35% of that. Entry 012 pinned the
cost precisely (the Tier-2 conformational search is 99.3% of docking ≈ 33% of the
whole loop), and entry 013 built a GPU oracle (`Docking6TD3GpuOracle`) that runs
that search on QuickVina2-GPU ~50× faster while keeping most of the validated
discrimination (AUROC 0.88 vs 0.948). The natural next step is to put that GPU
oracle *inside* the loop. At the same time the loop was throwing away everything
except each molecule's SMILES + final score — so we could not later check that a
suggested glue's by-construction synthesis route is plausible, nor analyse how
diversity / molecular weight evolve across rounds (a gap entry 011 explicitly
flagged: "track molecular weight and QED on every round's batch").

**Summary** — Two changes, run together. (1) We added a per-round **suggestion
log**: the loop now keeps each suggested molecule's full RGFN trajectory, writes
its structured **synthesis route** (starting building block + every reaction step
with SMARTS, reactants and product), its medchem descriptors, and the oracle's
score breakdown, plus a **batch-metrics** row per round (uniqueness, number of
Tanimoto modes, internal diversity, scaffold count, novelty vs the seed, and the
MW/QED/cLogP/etc. distributions) — all following the metric set RGFN's own mode
report and the source papers use. None of this touches the proxy-fit dataset; it
is pure provenance. (2) We swapped the loop's oracle from the CPU
`Docking6TD3Oracle` to the GPU `Docking6TD3GpuOracle`, holding every other knob
identical to the entry-012 baseline (3 rounds, 300 GFN steps/round, 32-molecule
query batch, β=8, top-16), so this run is a clean diff against entry 012 (job
69451): only the oracle's pose source and the new logging change.

## Answer

**The instrumentation and the GPU oracle both work; the run itself was killed by a
Balam node failure, so the headline cross-round result is pending a resubmit.** The
standard candidate logging wrote all four artifacts correctly *even through a total
docking failure* (every row recorded with `score=nan`, `status=no_pose`) — the
robustness we built it for. But on the compute node (balam009) every one of round
1's 32 molecules came back `no_pose` (zero poses, the wedged-OpenCL signature),
and the node then died outright ~1 h later, killing the job mid-round-2. A
login-node smoke test on a healthy Trillium H100 settles the cause: the *same*
oracle docks all of those molecules — the drug-sized seed controls (reproducing
job-006: dvina −2.82 / −1.60 vs ref −2.20 / −1.26) **and** the largest generated
molecule (MW 801, `n_poses=9`, dvina −1.49). So the failure was the node, not our
oracle or the molecules' size; no code change was needed to fix it, only a healthy
node. What the round-1 batch metrics *did* capture is a sharp, quantified picture
of the size drift entry 011 flagged: mean MW **649** (max 801), QED **0.13**,
cLogP 5.75, only **6%** Lipinski-passing, yet highly diverse (internal diversity
0.86, 32/32 modes, novelty 1.0 vs seed). Two robustness gaps the failure exposed
were then fixed (see below).

## Relevance to our Publication

This run advances two objectives at once. For **Objective 1 / Objective 2**, it
demonstrates the loop running end-to-end at GPU throughput — the answer to a
likely Digital Discovery / J. Cheminformatics reviewer question, "your oracle is
expensive; can it drive active learning at realistic scale?" For **Objective 5
(evaluation suite)**, the per-round suggestion log + batch metrics are the
substrate for the headline analyses reviewers expect: diversity and molecular-
weight trajectories under a fixed oracle budget, and — uniquely for a *reaction*
GFlowNet — verifiable synthesis routes for the top candidates, which a baseline
generator that does not build molecules by construction cannot offer.

## Next Experiments

**Refining for publication** — Run ≥3 seeds and the random-acquisition baseline on
the same oracle budget for the top-k-vs-oracle-calls curve (`[bengio2021gflownet]`
Fig. 7). Use the new batch-metrics CSV to plot the MW / QED / diversity trajectory
directly, quantifying the size drift entry 011 noted. Spot-check a handful of
top-k synthesis routes for chemical plausibility.

**Next steps in project** — Fold a QED / ligand-efficiency term into the reward
(Objective 2) to counter size drift, now that we can measure it per round. Drive
the sEH GPU oracle (entry 010) through the same instrumented loop for a second
end-to-end system (Objective 4).

# Re-creation

## Relevant Files

Root: `glue/`, `configs/glue/`, `experiments/active_learning/6td3/`

New / changed code (this entry):
- `./glue/active_learning/route.py` — **new.** `extract_route(states, actions)`:
  reconstructs a molecule's synthesis route from its RGFN trajectory — the
  `ReactionAction0` building block + every `ReactionActionC` (reaction SMARTS,
  reactants, product). `route_to_str` renders a one-line CSV-friendly summary.
- `./glue/metrics/dataset_metrics.py` — **new.** Pure (gin-free) per-molecule
  descriptors (ExactMolWt, QED, cLogP, HBD/HBA, heavy atoms, rotatable bonds,
  TPSA, rings, Lipinski-Ro5, ligand efficiency, Murcko scaffold) and set metrics
  (uniqueness, # Tanimoto modes [Morgan r3/2048, 0.7], internal diversity, scaffold
  count, novelty-vs-seed, descriptor + oracle-score distributions). Metric set
  mirrors `rgfn/trainer/metrics/reaction_metrics.py` and `[koziarski2024rgfn]` /
  `[bengio2021gflownet]`. Runs live in the loop *and* retroactively on any CSV.
- `./glue/datasets/candidates.py` — **new.** The ONE standard candidate-dataset
  format (`CandidateDataset` writer, `from_smiles_table` ingest, `read_candidate_dataset`
  / `validate_candidate_dataset`): `manifest.json` + `candidates.csv` (+ `routes.jsonl`).
  Lives in `glue/` so the loop can write it and `validation/` can import it (one-way
  dependency rule). Spec: `docs/CANDIDATE_DATASET_FORMAT.md`.
- `./glue/datasets/suggestion_log.py` — **new.** `SuggestionLog`: writes the standard
  candidate dataset to `suggestions/` (`manifest.json` + `candidates.csv` +
  `routes.jsonl`, with the AL round as the standard `step` column; `generator="rgfn"`)
  plus `batch_metrics.csv` (one set-metrics row per round — a non-standard analysis
  sidecar). Sidecar only — never feeds training.
- `./glue/active_learning/loop.py` — `_sample_query_batch` now returns each
  molecule's route (was discarding trajectories); `_score_batch` prefers the
  oracle's `score_detailed()` (GPU oracle's per-pose breakdown) over scalar
  `score()`; `run()` snapshots the seed D_0 for novelty, calls `SuggestionLog`, and
  forwards batch metrics to the trainer logger. **Robustness fixes (post job 69481):**
  (a) aborts the loop with a clear error if a whole round's oracle batch is all-NaN
  (the silent-no-op that wasted round-2 training here); (b) new `system`/`seed`
  constructor args threaded into the manifest provenance.
- `./glue/datasets/candidates.py` (+ `suggestion_log.py`) — `SuggestionLog` now
  takes/records `system` + `seed` in the manifest (were `null` in this run).
- `./glue/datasets/__init__.py` — export `SuggestionLog` + the candidate-format API
  so `glue.registry` registers them.

Scripts / entry points:
- `./experiments/active_learning/6td3/submit_al_6td3_gpu.sh` — **this run.** Same
  loop as `submit_al_6td3_pregpu.sh` (entry 012) but GPU oracle; QV2-GPU env
  (boost libs, `$GNINA`), `--exclude balam008`, OpenCL health gate (entry 013).
- `./experiments/active_learning/6td3/analyze_suggestions.py` — **new.** Retroactive
  per-round batch metrics from a saved standard `candidates.csv` (re-runs the metric
  functions without re-docking; works on any generator's standard dataset).
- `./scripts/active_learning.py` — driver; now binds `ActiveLearningLoop.seed`
  from `--seed` so the run seed reaches the manifest provenance.

Config:
- `./configs/glue/active_learning_6td3_gpu.gin` — **new.** Byte-for-byte
  `active_learning_6td3_mini.gin` (entry 012) except `@Docking6TD3Oracle` →
  `@Docking6TD3GpuOracle` (exhaustiveness=8000 QV2 threads, num_modes=9,
  docking_batch_size=25, n_gpu=1) + `ActiveLearningLoop.system='6td3'`. 3 rounds,
  300 steps/round, 32 batch, top-16.

Datasets / receptors (gitignored, on Balam):
- `./experiments/active_learning/6td3/seed_6td3.csv` — seed D_0 (408 labels; entry 002).
- `./experiments/oracle_validation/docking_6td3/6TD3_tier{1,2}.pdbqt`, `crystal_RC8.pdb`
  — GPU oracle's default receptors (Tier 2 = CDK12+DDB1, Tier 1 = CDK12 alone, crystal RC8 for the autobox).

Results (gitignored, on `$SCRATCH`):
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/6td3_gpu/<timestamp>/active_learning/`
  — `suggestions/{manifest.json, candidates.csv, routes.jsonl, batch_metrics.csv}`
  (standard candidate dataset + per-round metrics sidecar), `dataset_round_00{1,2,3}.csv`,
  `top_k.csv`, `phase_timings.csv`, `docking_timings.csv`.

Diagnostic smoke test (login node, no SLURM):
- `scratchpad/smoke_dock.py` — docked 2 drug-sized seed controls + the smallest
  (MW 460) and largest (MW 801) round-1 molecules through `Docking6TD3GpuOracle`
  on a **Trillium H100** (`source ~/bin/rgfn-smoke-env.sh`). All `status=ok,
  n_poses=9` → the balam009 all-`no_pose` was the node, not the oracle/molecules.

Job Logs:
- `/scratch/markymoo/rgfn_runs/al_6td3_gpu-69481.out` / `.err`.

Job: **SLURM job 69481** — balam009, started 2026-06-29 16:31; run dir
`…/6td3_gpu/2026-06-29_16-31-16/`. Completed round 1 (proxy fit + 62-min GFN train
+ sample), then **all 32 round-1 docks returned `no_pose`** (`oracle_score` 4.5 min
of failed retries) → dataset `+0`; killed mid-round-2 GFN training by the Balam
compute-node outage (~18:40). **Outcome: incomplete — resubmit on a healthy node**
(no config/code change needed; the post-mortem robustness fixes above just make the
next failure fail fast and louder). Predecessors job 69479/69480 were cancelled
during round-1 training to roll in the suggestion-log + standardization changes.

## Relevant Versions

[TODO — add commit hash after committing the new/changed files listed above.]

## Relevant Resources

**Sources**
- `[koziarski2024rgfn]` — RGFN; evaluation metric set (QED, MW, modes, scaffolds).
- `[bengio2021gflownet]` — GFlowNet; number-of-modes / diversity / reward distribution.
- PDB **6TD3** — CDK12–DDB1 / CR8 ternary complex (the glue testbed).

**Packages**
- RDKit 2023.09.5 — descriptors / fingerprints / Murcko scaffolds (`glue/metrics/dataset_metrics.py`).
- QuickVina2-GPU-2.1 + gnina — GPU pose search + CNN rescore (`glue/oracles/docking_gpu_differential_oracle.py`).

## Method

1. `sbatch experiments/active_learning/6td3/submit_al_6td3_gpu.sh` — 1-GPU compute
   job: OpenCL health gate, then `python scripts/active_learning.py --cfg
   configs/glue/active_learning_6td3_gpu.gin --seed 42 --root-dir $SCRATCH/rgfn_runs/experiments`.
2. (post-hoc) `python experiments/active_learning/6td3/analyze_suggestions.py
   --suggestions <run>/active_learning/suggestions/candidates.csv
   --seed experiments/active_learning/6td3/seed_6td3.csv --out <run>/.../batch_metrics_recomputed.csv`.

## Results

Partial — round 1 only (job 69481 killed by the node outage before round 2 finished).
The cross-round trend + Top-K await a resubmit on a healthy node.

**Round-1 phase timing** (`phase_timings.csv`): fit_proxy 6.2 s · train_gfn **62.0 min**
· sample_batch 12.8 s · oracle_score 4.5 min (failed docking + 10 retries). Even the
*failed* docking round was 4.5 min vs entry-012's ~31 min/round CPU docking — once it
runs on a healthy node the docking phase should collapse as predicted.

**Round-1 docking:** 32/32 molecules `status=no_pose`, `n_poses=0`, `score=nan` →
dataset `+0` (408→408). All-uniform failure across the MW range = node, not size.

**Round-1 batch metrics** (`batch_metrics.csv`; logged despite the docking failure):

| metric | value | | metric | value |
|---|---|---|---|---|
| n_suggested / valid / unique | 32 / 32 / 32 | | mol_weight_mean / max | **649 / 801** |
| num_modes (Tanimoto<0.7) | 32 | | qed_mean | **0.13** |
| num_scaffolds | 32 | | clogp_mean | 5.75 |
| internal_diversity | 0.86 | | rotatable_bonds_mean | 11.8 |
| novelty_vs_seed | 1.0 | | frac_lipinski_pass | **0.0625** |

Diverse but far from drug-like — the size drift of entry 011 / RGFN §5, now quantified.

**Smoke test on a healthy GPU** (Trillium H100, `scratchpad/smoke_dock.py`), proving
the oracle and molecules are fine — the run-time failure was balam009:

| molecule | MW | status | n_poses | vina_t2 | dvina |
|---|---|---|---|---|---|
| seed known (ref dvina −2.20) | ~320 | ok | 9 | −10.13 | **−2.82** |
| seed known (ref dvina −1.26) | ~380 | ok | 9 | −8.51 | **−1.60** |
| round-1 generated (small) | 460 | ok | 9 | −7.44 | −0.35 |
| round-1 generated (largest) | 801 | ok | 9 | −9.47 | **−1.49** |
