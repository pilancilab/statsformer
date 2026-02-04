#!/bin/bash

# Run experiment for all datasets
DATASETS=("ETP" "bank_marketing" "superconductivity" "credit_g" "breast_cancer" "internet_ads" "lung_TCGA" "nomao")

for DATASET in "${DATASETS[@]}"; do
    PRIOR_DIR=data/generated_priors/$DATASET/RAG_disabled/o3__temp_0__batch_40/default_prompt__default_system_prompt/initial_scores
    python scripts/stats/baseline_comparison.py $DATASET \
        --prior_dir $PRIOR_DIR --trial_name adversarial --adversarial
done