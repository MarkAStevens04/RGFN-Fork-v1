#!/usr/bin/env bash
# Pass B (flexible-anchor, warhead wiggles + output tether filter) on the SAME sets as pass A:
#   - random 10% of glutarimide-eligible known glues (seed=42 -> identical 368 compounds)
#   - all 260 realistic-random decoys
# New output files so the pass-A CSVs survive for the head-to-head comparison.
set -euo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export ANCHOR_PASS=B
export ANCHOR_EMBED=free

KNOWN="../test-data/Enamine_CRBN_Molecular_Glue_Library_plated_4560cmpds_20260614.smiles"

echo "########## PASS B: KNOWN GLUES (10% sample) ##########"
python batch_anchor_dock.py "$KNOWN" batch_results_passB.csv 0.10 100

echo "########## PASS B: DECOYS (all) ##########"
python batch_anchor_dock.py decoys.smiles decoy_results_passB.csv 1.0 100

echo "ALL PASS B DONE"
