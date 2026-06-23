# Experimental Logs — CRBN/6TD3 molecular-glue docking oracle

Lab notebook for building a docking-based reward (oracle) for an RGFN generative model that
designs molecular glue degraders. Each entry is a dated, self-contained record of an experiment:
objective, method, key scripts, results, **where the result files live**, and conclusions.

## Index

| # | Date | Title | Verdict |
|---|------|-------|---------|
| [001](001_5hxb-crbn-anchored-docking.md) | 2026-06-11 → 06-18 | 5HXB / CRBN — warhead-anchored docking oracle + decoy control | Works as a docker; **discrimination ceiling** — GSPT1 bonus is a geometric artifact, not a glue signal |
| [002](002_6td3-cr8-validation-and-discrimination.md) | 2026-06-18 | 6TD3 / CDK12-DDB1 — oracle validation + discrimination run | **Decisive discrimination** — 85.6% vs 7.3% on DDB1 differential (+78 pts); validated oracle |
| [003](003_crbn-vs-6td3-cross-system.md) | 2026-06-18 | 5HXB vs 6TD3 — head-to-head neosubstrate differential comparison | **6TD3 differential discriminates (+78 pts); 5HXB does not (−3 pts)** — failure is structural, not methodological |
| [004](004_compute-benchmark.md) | 2026-06-18 | Compute benchmark — login node vs Balam debug_full_node (4× A100) | **CRBN 21× / 6TD3 6.6× faster; batching is the dominant lever** — conformer embedding is the new Amdahl bottleneck |
| [005](005_tier2-vina-roc-pr-curves.md) | 2026-06-23 | Tier 2 Vina — ROC and PR curves for 6TD3 and 5HXB | **6TD3 AUC=0.890 / AP=0.872; CRBN AUC=0.627 / AP=0.726** — absolute Tier 2 score alone is a strong 6TD3 oracle; CRBN ceiling confirmed as structural |

## The systems

- **5HXB** = CRBN·DDB1·GSPT1 ternary complex + glue **CC-885**. E3 = CRBN; neosubstrate = GSPT1.
  Warhead = glutarimide (binds the CRBN tri-Trp cage). Blind docking can't sample the deep cage →
  we **anchor** the glutarimide to its crystal pose and relax the arm (Pass B, flexible anchor).
- **6TD3** = DDB1·CDK12–cyclinK + glue **CR8** (RC8). E3 adaptor = DDB1; neosubstrate partner =
  cyclin K (CDK12-bound). Warhead = 2,6,9-trisubstituted purine (binds the CDK12 ATP pocket).
  Deep druggable pocket → straight box docking samples the native pose (no anchoring needed).

## Common methodology

- Docking engine: **gnina v1.3.2** (CNN-rescored), launcher `/scratch/markymoo/gnina/run_gnina.sh`
  (loads CUDA-12). Conformers via RDKit.
- **Receptor tiers:** Tier 1 = E3 (pocket) only; Tier 2 = E3 + neosubstrate. The
  **neosubstrate differential = Tier2 − Tier1 score of the SAME pose** (via gnina `--score_only`)
  isolates the glue-specific cooperativity (GSPT1 bonus for CRBN, DDB1 bonus for 6TD3).
- **Decoy control:** realistic "fake" molecules = the conserved warhead + a random drug-like arm
  (not selected for glue activity). If decoys score like real glues, the proxy only reads E3-pocket
  binding; if real glues win, the proxy rewards a productive arm.
- **Cluster:** SciNet Balam, `debug_full_node` (4× A100, 64 cores, 1 h). Sharded across 4 GPUs
  (16 workers). See `pre-processing/*/submit_dock_*.sh`. Outputs go to `$SCRATCH` ($HOME is
  read-only on compute nodes).

## Where results live

- **Repo CSVs** (committed, small): `pre-processing/docking_6td3/{known,decoy_cdk}_results.csv`,
  `pre-processing/docking_gnina/{batch_results_passB,decoy_results_passB}.csv`.
- **Scratch run dirs** (full outputs + SLURM logs): `/scratch/markymoo/rgfn_runs/dock_6td3_<jobid>/`,
  `/scratch/markymoo/rgfn_runs/dock_crbn_<jobid>/`, and `dock6td3-<jobid>.out` / `dockcrbn-<jobid>.out`.
- **Compare script:** `pre-processing/compare_systems.py` (per-system known-vs-decoy + cross-system).

## Datasets (`pre-processing/test-data/`)

- `DDB1_CDK12_Glues.csv` — 175 real CDK12-CCNK/DDB1 glues (161 unique, 123 purine-based). KNOWN+.
- `CRBN_GSPT1_Glues.csv` — 200 real CRBN/GSPT1 glues (199 active, 188 glutarimide-bearing, MW up
  to ~1009 → a few are likely bivalent/PROTAC-range, not pure glues). KNOWN+.
- `Enamine_CRBN_Molecular_Glue_Library_..._4560cmpds_*.smiles` — purchasable SCAFFOLD library
  (potential glues, NOT validated degraders). Used in early CRBN runs; superseded by the curated
  set above for the "known" positives.
