# 001 — 5HXB / CRBN warhead-anchored docking oracle + decoy control

**Dates:** 2026-06-11 → 2026-06-18
**System:** 5HXB (CRBN·DDB1·GSPT1 + glue CC-885). E3 = CRBN, neosubstrate = GSPT1, warhead = glutarimide.

## Objective
Build a docking reward for CRBN molecular glues and test whether it can tell real glues from
random warhead-bearing molecules (the property an RGFN reward must have).

## Method
1. **Structure prep** — `pre-processing/clean.py` carves 5HXB copy 1 into tiers:
   `models/5HXB_tier1_CRBN.pdb` (CRBN only), `models/5HXB_tier2_CRBN_GSPT1.pdb` (CRBN+GSPT1);
   Zn retained, glue removed. Receptors → pdbqt via `obabel ... -xr -p 7.4`.
2. **Sampling problem** — blind docking (Vina, gnina, even exh 64 + CNN) NEVER samples CC-885's
   deep tri-Trp-cage pose (tops out ~−8 kcal/mol, CNN ~0.5, all displaced). But the native pose
   *minimizes in place* to −13.8 / CNN 0.98 → it's a **sampling**, not scoring, failure.
3. **Warhead-anchored oracle** — `docking_gnina/anchor_dock.py`: pin the glutarimide to the
   crystal cage coords, sample/relax only the arm, gnina-minimize, select clash-aware best pose.
   - **Pass A** (hard-frozen warhead): recovers native (−11.5/CNN0.88) but ~29–42% of library/decoy
     compounds give no clash-free pose.
   - **Pass B** (flexible anchor — warhead wiggles within 1.5 Å + output tether filter rejecting
     poses whose warhead escapes the cage > 2.5 Å): CC-885 → **−13.55 / CNN 0.97** (≈ native gold),
     fit-rate jumps to 94% (known) / 80% (decoy). Pass B is the validated pipeline.
4. **Decoy control** — `docking_gnina/make_decoys.py` builds realistic glutarimide molecules
   (IMiD scaffolds + random drug-like arms via amide/urea/sulfonamide/reductive-amination).
   Score known (purchasable library 10% sample) vs decoys via `batch_anchor_dock.py`; compare with
   `compare_passes.py`.

## Results
- **Pass B recovers the native deep well** and is a faithful *docker*. ✔
- **But it does NOT discriminate** known glues from random warhead-bearing decoys. Even after Pass B
  fixed sampling (rescued 24% more compounds), the known−decoy medians stay close:
  median Vina −6.70 vs −5.86 (Δ0.85 kcal/mol), median CNNaff 6.21 vs 6.09 (Δ0.12). Separation lives
  only in the strong tail (frac Vina<−10: 15% vs 7%) and fit-rate (94% vs 80%).
- The Vina-physics channel sharpened ~65% from Pass A→B, but medians didn't close → **a true
  ceiling, not sampling noise.**

## Conclusion
Anchored CRBN docking measures **CRBN-pocket (warhead) binding**, which real glues and random
IMiD-derivatives share. The glue-defining **GSPT1 cooperativity comes from a large recruited
protein–protein surface the proxy is structurally blind to** → weak RGFN reward gradient. This
motivated the pivot to a system where the glue interface is ligand-mediated (→ entry 002).

## Files & where results live
- Scripts: `pre-processing/clean.py`, `docking_gnina/anchor_dock.py`,
  `docking_gnina/batch_anchor_dock.py`, `docking_gnina/make_decoys.py`,
  `docking_gnina/compare_passes.py`.
- Receptors/refs: `docking_gnina/5HXB_tier2.pdbqt`, `5HXB_tier1_CRBN.pdbqt`, `crystal_85C.pdb`.
- Result CSVs (repo): `docking_gnina/batch_results_passB.csv` (known), `decoy_results_passB.csv`
  (decoys); Pass-A: `batch_results.csv`, `decoy_results.csv`.
- Memory: `crbn-warhead-anchored-oracle`, `5hxb-glue-redock-vina-limitation`.
