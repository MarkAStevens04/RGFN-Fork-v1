# 002 — 6TD3 / CR8 (CDK12-cyclinK) validation + discrimination run

**Date:** 2026-06-18
**System:** 6TD3 (DDB1·CDK12–cyclinK + glue CR8/RC8). E3 adaptor = DDB1, neosubstrate = cyclin K,
warhead = 2,6,9-trisubstituted purine (CDK12 ATP-pocket binder). Słabicki et al., *Nature* 2020.

## Objective
Test the hypothesis that CRBN's discrimination ceiling (entry 001) is system-specific: pick a glue
system where the neosubstrate interface is **formed by the small molecule's own arm** (so docking
can see it) and check whether the proxy then discriminates.

## Why this system should work
CR8 binds the deep, druggable **CDK12 ATP pocket** (no cage-sampling problem) and its solvent-
exposed pyridyl-phenyl **arm protrudes to directly contact DDB1**. The glue interface is
ligand-mediated and sits *inside the docking box*.

## Method
1. **Structure prep** — `pre-processing/clean_6td3.py` carves 6TD3 copy 1: Tier 1 = CDK12
   (`models/6TD3_tier1_CDK12.pdb`), Tier 2 = CDK12+DDB1 (`6TD3_tier2_CDK12_DDB1.pdb`), Tier 3 =
   +cyclinK; native CR8 → `crystal_RC8.pdb`. Receptors → pdbqt (obabel). No anchoring — straight box
   docking (`--autobox_ligand crystal_RC8.pdb --autobox_add 4`).
2. **Validation** — `docking_6td3/redock_cr8.py`: blind-redock CR8 + native in-place min + Tier1/Tier2.
3. **Discrimination run** — `docking_6td3/dock_cluster.py` on `debug_full_node` (job 69271):
   160 real glues (`test-data/DDB1_CDK12_Glues.csv`) vs 248 purine-armed decoys
   (`make_decoys_cdk.py`). Dock Tier2, take best-CNN pose, score that pose vs Tier1 →
   **DDB1 differential** = Tier2 − Tier1.

## Results
**Validation:**
- Blind redock **recovers the native pose at 1.23 Å, ranked #1** (Vina −10.68, CNN 0.99). Native
  in-place min −10.56 / CNN 0.985. (Contrast CRBN: never sampled native.)
- **DDB1 cooperativity is captured:** CR8 Tier1 (CDK12) −8.39 vs Tier2 (CDK12+DDB1) −10.68 →
  **DDB1 bonus −2.16 to −2.29 kcal/mol** on the same pose.

**Discrimination run (job 69271, 13 min, full node):**
| metric | known (n=160) | decoy (n=248) | gap |
|---|---|---|---|
| **frac DDB1 dVina < −1.5** | **85.6%** | **7.3%** | **78 pts** |
| median DDB1 dVina | −2.20 | −0.60 | −1.60 |
| median Tier2 Vina | −10.15 | −7.96 | −2.19 |
| median Tier2 CNNaff | 7.82 | 6.70 | 1.13 |
| frac Tier2 Vina < −10 | 55.6% | 3.6% | 52 pts |

## Conclusion
**Decisive discrimination.** 85.6% of real glues get a strong DDB1 bonus vs 7.3% of decoys — the
separation CRBN never produced. The proxy rewards a *productive DDB1-contacting arm*, not just
warhead presence. → This system is the right RGFN testbed. Reward = Tier2 Vina + DDB1-differential
gate. Warhead for warhead-constrained generation = the purine ATP-hinge binder.

## Files & where results live
- Scripts: `pre-processing/clean_6td3.py`, `docking_6td3/redock_cr8.py`, `make_decoys_cdk.py`,
  `dock_cluster.py`, `submit_dock_6td3.sh`, `pre-processing/compare_systems.py`.
- Receptors/refs: `docking_6td3/6TD3_tier{1,2}.pdbqt`, `crystal_RC8.pdb`; structure `models/6TD3*.pdb`.
- Result CSVs (repo): `docking_6td3/known_results.csv`, `decoy_cdk_results.csv`.
- Scratch (full run + SLURM log): `/scratch/markymoo/rgfn_runs/dock_6td3_69271/`,
  `/scratch/markymoo/rgfn_runs/dock6td3-69271.out`.
- Memory: `6td3-cr8-cyclink-glue-system`, `balam-slurm-submission`.
