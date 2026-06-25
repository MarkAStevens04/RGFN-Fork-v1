# 6TD3 — does pose selection (CNN vs. Vina) change the glue/decoy discrimination?

**Date:** 2026-06-25, ~4pm

## Question

When we dock a candidate glue and keep one pose to score, does it matter whether we keep the pose our neural-network ranker likes best or simply the one the docking program scores best?

## Context & Summary

Our oracle docks a molecule into the CDK12+DDB1 complex, keeps a single pose, and scores how much extra binding the recruited partner adds (the "Vina Tier 2 − Tier 1 differential"). Entry 006 ranked six candidate scoring signals and found this differential is the best discriminator between real glues and decoys (AUROC 0.946), but every one of those six was measured on the *same* pose — the one with the highest gnina CNN score. We have never tested whether the rule used to *pick* that pose matters. The natural cheaper alternative is to just keep the docking program's own top-scored pose (best Vina energy) and skip the CNN entirely. We added CNN pose-picking in the first place because plain Vina ranking buried the correct pose in earlier work (the CC-885/5HXB redock, entry 001), so this entry asks whether that still matters for 6TD3.

This experiment re-docks the same known-glue and decoy molecules from entry 002 but keeps **every** pose and scores all of them against both receptors. With every pose's scores in hand we can, without any further docking, pick a pose two ways — by CNN score (what we do now) and by Vina energy (the "just take the top pose" alternative) — form the validated differential each way, and compare how well each separates real glues from decoys. The CNN-picked version doubles as an in-run sanity check that should land near the published 0.946.

## Answer

*[TODO — fill after the Balam run. Expected shape: "Selecting by Vina instead of CNN changes the differential's AUROC from X to Y; the two rules pick the same pose Z% of the time, so pose selection {does / does not} materially affect discrimination."]*

## Relevance to our Publication

This is a design-justification ablation that Digital Discovery / J. Cheminformatics reviewers will ask for: we claim the gnina CNN is worth the extra machinery for pose selection, and a reviewer can reasonably ask "would the plain docking pose have done just as well?" This entry answers it head-on. A null result (CNN and Vina selection discriminate equally) would let us simplify and defend a cheaper oracle; a positive result (CNN selection wins) directly justifies the current design with evidence rather than assertion, and complements the entry-006 scoring-signal ablation by closing the remaining pose-*selection* gap.

## Next Experiments

**Refining for publication**
- If the two rules diverge, add per-molecule pose RMSD between the CNN-picked and Vina-picked poses (the all-poses SDF is retained) to quantify *how different* the chosen geometries are, not just how often they differ.
- Repeat on the MW-matched subsets from entry 007 to confirm the conclusion is not an artifact of ligand size.

**Next steps in project**
- Lock the pose-selection rule justified here into the production `Docking6TD3Oracle`, then run RGFN against the validated oracle and show generated molecules populate the high-scoring region the known glues occupy.

# Re-creation

## Relevant Files

Root: `research/preprocessing/pose_selection_ablation/`

**Scripts**
- `dock_allposes.py` — variant of `docking_6td3/dock_cluster.py` that retains **all** `num_modes` docked poses and `--score_only`s every pose against Tier 1, emitting a per-pose CSV. Identical gnina flags / embedding / autobox / exhaustiveness / num_modes / seed / sharding to `dock_cluster.py` so the CNN-selected arm reproduces production as an in-run control.
- `analyze_pose_selection.py` — offline (no docking): for each molecule picks one pose by CNNscore (`max cnnsc_t2`, production) and by Vina (`min vina_t2`, ablation), forms `ddb1_dvina = vina_t2 − vina_t1` on each, and reports Cohen's d / AUROC / Mann-Whitney U per rule plus the pose-agreement fraction. Writes `pose_selection_stats.csv` + `pose_selection_violins.png`. Reuses the discrimination math from `full_comparison/discrimination_stats.py`. Reads `$DATA_DIR`, writes `$OUT_DIR` (defaults to `$DATA_DIR`, since `$HOME` is read-only on Balam compute nodes).
- `submit_pose_ablation.sh` — SLURM submit (debug_full_node, 4× A100, no explicit cpu/mem per Balam rules); runs the redock then the analysis, writing to `$SCRATCH`.
- `collect_results.sh` — run from a **login node** after the job: copies the result CSVs + figure from `$OUTDIR` (scratch) back into the repo dir, since the compute-node job can't write to read-only `$HOME`.
- `README.md` — standalone run + interpretation guide for a future agent (what each file is, how to run on Balam, how to read `pose_selection_stats.csv`).
- `smoke_test.py` — off-Balam gate (no gnina/GPU): exercises `load_all`/`embed`/`_poses` and the full analysis on synthetic per-pose CSVs; exits non-zero on failure. Run before spending a Balam allocation.

**Datasets** (inputs; same molecule set as entry 002)
- `../docking_6td3/../test-data/DDB1_CDK12_Glues.csv` — known CDK12-DDB1 glues (positives).
- `../docking_6td3/decoys_cdk.smiles` — warhead-matched decoys (negative control).
- `../docking_6td3/6TD3_tier{1,2}.pdbqt`, `../docking_6td3/crystal_RC8.pdb` — Tier-1 (CDK12 alone) / Tier-2 (CDK12+DDB1) receptors and the CR8 autobox reference (Balam-only; git-ignored `.pdbqt`).

**Results** (produced on Balam, into `$OUTDIR` on scratch; copy `*_allposes.csv` into the repo dir to commit)
- `known_allposes.csv` / `decoy_allposes.csv` — one row per (molecule, pose): `vina_t2`, `cnnsc_t2`, `cnnaff_t2`, `vina_t1`, `cnnaff_t1`, `pose_rank`.
- `pose_selection_stats.csv` — per-rule discrimination summary (Cohen's d, AUROC, AUROC vs. CNN).
- `pose_selection_violins.png` — known-vs-decoy `ddb1_dvina` violins, one panel per selection rule.

## Relevant Versions

*[TODO — add commit hash after pushing.]* The three scripts in `research/preprocessing/pose_selection_ablation/` are uncommitted as of writing. Result CSVs do not exist yet (Balam run pending). **Action needed:** commit `dock_allposes.py`, `analyze_pose_selection.py`, `submit_pose_ablation.sh`, and this log; once the run completes, commit the `*_allposes.csv` outputs and update this section + the Results table.

## Relevant Resources

**Sources**
- Molecule set + 6TD3 docking pipeline: entry 002 (`002_6td3-cr8-validation-and-discrimination.md`), Balam job 69271.
- Scoring-signal ablation this extends (held pose selection fixed at max-CNNscore): entry 006 (`006_6td3-violin-distributions.md`); MW control: entry 007.
- Motivation for CNN pose selection (Vina ranking buried the native pose): entry 001 (`001_5hxb-crbn-anchored-docking.md`).
- CR8/6TD3 structure: Słabicki et al., *Nature* 2020 (doi:10.1038/s41586-020-2133-z).

**Packages**
- gnina v1.3.2 (CNN-rescored docking) — `dock_allposes.py`.
- RDKit (embedding/MMFF, SDF I/O) — `dock_allposes.py`.
- pandas / numpy / scipy (`mannwhitneyu`) / matplotlib — `analyze_pose_selection.py`.

## Method

*(Planned; not yet run — Balam compute node unavailable at time of writing.)*

1. **Redock retaining all poses** — `sbatch research/preprocessing/pose_selection_ablation/submit_pose_ablation.sh`. Embeds the entry-002 known+decoy molecules, docks each into Tier 2 (CDK12+DDB1, autoboxed on crystal CR8, exhaustiveness 16, num_modes 9, seed 42), keeps every pose, and `--score_only`s all poses against Tier 1 (CDK12 alone) → `known_allposes.csv` / `decoy_allposes.csv` in `$OUTDIR`.
2. **Pose-selection ablation** — `DATA_DIR=$OUTDIR python analyze_pose_selection.py` (run automatically by the submit script): re-pick each molecule's pose by `max cnnsc_t2` vs. `min vina_t2`, compute `ddb1_dvina` per rule, and report discrimination + pose agreement.
3. **Sanity check** — confirm the CNN-selected arm's AUROC lands near the entry-006 baseline (0.946); large divergence flags a pipeline mismatch, not a real effect.

## Results

*[TODO — fill after the run. Planned table:]*

| selection rule | select by | known median dVina | decoy median dVina | Cohen's d | AUROC | ΔAUROC vs CNN |
|---|---|---|---|---|---|---|
| cnn (production) | max(cnnsc_t2) | — | — | — | — | 0.000 |
| vina (ablation) | min(vina_t2) | — | — | — | — | — |

Pose agreement (same pose chosen by both rules): known —%, decoy —%.
