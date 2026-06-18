#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
echo "########## KNOWN GLUES (DDB1_CDK12_Glues) ##########"
python dock_6td3_batch.py ../test-data/DDB1_CDK12_Glues.csv known_results.csv known
echo "########## DECOYS (CR8-purine + random arms) ##########"
python dock_6td3_batch.py decoys_cdk.smiles decoy_cdk_results.csv decoy
echo "ALL 6TD3 DOCKING DONE"
