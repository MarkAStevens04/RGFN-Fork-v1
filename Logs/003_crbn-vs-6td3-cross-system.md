# 003 — 5HXB vs 6TD3 head-to-head on the neosubstrate differential

**Date:** 2026-06-18
**Status:** ✅ Complete (CRBN job 69272, 5.8 min; 6TD3 job 69271 from entry 002).

## Objective
Put both glue systems on **identical methodology** and ask the same question of each: does the
**neosubstrate-contact differential** (Tier2 − Tier1, same pose) separate real glues from random
warhead-bearing decoys? This isolates *why* 6TD3 works and 5HXB hits a ceiling — on one metric.

- 5HXB/CRBN → **GSPT1 differential** = (CRBN+GSPT1) − (CRBN), warhead-anchored Pass-B docking.
- 6TD3/CR8 → **DDB1 differential** = (CDK12+DDB1) − (CDK12), straight box docking (from entry 002).

## Method
- New curated CRBN positives: `test-data/CRBN_GSPT1_Glues.csv` (200 real glues, 199 active, 188
  glutarimide-bearing → 177 unique anchorable). Replaces the earlier purchasable *scaffold* library
  (which was potential, not validated, glues). Decoys: `docking_gnina/decoys.smiles` (260
  glutarimide + random arm).
- `docking_gnina/dock_cluster_crbn.py` on `debug_full_node` (job 69272): build 100 flexible-anchor
  confs/mol → gnina `--minimize` vs Tier2 (CRBN+GSPT1) → best on-tether pose by clash-aware Vina →
  `--score_only` vs Tier1 (CRBN) → GSPT1 differential. Sharded 16 workers / 4 GPUs, per-shard CSV
  checkpoints.
- Compare: `pre-processing/compare_systems.py` (prints both systems + the known−decoy gap on the
  differential).

## Results

**The neosubstrate differential (known vs decoy):**
| system | metric | known | decoy | gap |
|---|---|---|---|---|
| 5HXB / CRBN | median GSPT1 dVina | −2.95 | −2.35 | **−0.60** |
| 5HXB / CRBN | frac dVina < −1.5 | 85.3% | 88.0% | **−3 pts** |
| 6TD3 / CDK12 | median DDB1 dVina | −2.20 | −0.60 | **−1.60** |
| 6TD3 / CDK12 | frac dVina < −1.5 | 85.6% | 7.3% | **+78 pts** |

**Supporting metrics:**
| system | fit-rate (k / d) | median Tier2 Vina (k / d) | frac Tier2 Vina < −10 (k / d) |
|---|---|---|---|
| 5HXB / CRBN | 136/177 (77%) / 117/260 (45%) | −8.63 / −7.35 | 35.3% / 12.0% |
| 6TD3 / CDK12 | 160/160 / 248/248 | −10.15 / −7.96 | 55.6% / 3.6% |

## Conclusion
On **identical neosubstrate-differential methodology**, the two systems split cleanly:
- **6TD3 — the differential IS the glue signal.** DDB1 bonus separates glues from decoys by 78
  points (85.6% vs 7.3%). Only a real CR8-like arm reaches DDB1; a random purine arm doesn't.
- **5HXB — the differential is NOT a glue signal.** ~86% of *both* known glues and decoys show a
  "strong" GSPT1 bonus (gap −3 pts; decoys marginally higher). Pinning any glutarimide in the cage
  places its arm against the adjacent GSPT1 surface, so the bonus is a **nonspecific geometric
  artifact**, not a productive, glue-defining contact. This is the structural reason CRBN hits the
  ceiling (entry 001): GSPT1 cooperativity lives in a large protein–protein surface, not in a
  ligand-mediated contact the score can resolve.

**Nuance:** with *real* glues (vs the old scaffold library) CRBN does show modest discrimination in
**absolute binding** — fit-rate 77% vs 45%, Tier2 Vina<−10 35% vs 12%. So a CRBN reward isn't
worthless, but the glue-*specific* cooperativity term that makes 6TD3 shine is absent. **6TD3 is
the far better RGFN testbed.**

**Headline:** known−decoy gap on the differential — 5HXB **−0.60 / −3 pts** vs 6TD3 **−1.60 / +78 pts**.

## Caveats
- The two docking methods differ (anchored vs box) because the systems demand it — comparison is on
  the *discrimination* (known−decoy gap), not absolute scores.
- `CRBN_GSPT1_Glues.csv` spans MW up to ~1009; a few entries are likely bivalent/PROTAC-range, not
  pure molecular glues, and won't behave as glues under docking (dataset is imperfect by the
  author's note). Consider an MW < ~600 sensitivity check.

## Files & where results live
- Scripts: `docking_gnina/dock_cluster_crbn.py`, `submit_dock_crbn.sh`,
  `pre-processing/compare_systems.py`.
- Receptors: `docking_gnina/5HXB_tier2.pdbqt` (CRBN+GSPT1), `5HXB_tier1_CRBN.pdbqt` (CRBN only).
- Result CSVs (repo): `docking_gnina/known_crbn_results.csv`, `decoy_crbn_results.csv`.
- Scratch (full run + SLURM log): `/scratch/markymoo/rgfn_runs/dock_crbn_69272/`,
  `/scratch/markymoo/rgfn_runs/dockcrbn-69272.out` (+ per-shard `shard*.csv` checkpoints).
- Note: one gnina worker segfaulted mid-run (libc, one shard) but the per-shard try/except +
  checkpointing contained it; job exited 0. A few molecules in that shard may be missing — not
  material to the distributions.
