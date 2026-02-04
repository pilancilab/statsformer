METHODS=("random_forest" "xgboost" "lasso" "kernel")
DATASETS=("ETP" "bank_marketing" "superconductivity" "credit_g" "breast_cancer" "internet_ads" "lung_TCGA" "nomao")

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        PRIOR_DIR=data/generated_priors/$DATASET/RAG_disabled/o3__temp_0__batch_40/default_prompt__default_system_prompt/initial_scores
        python scripts/tuning/single_learner_study.py \
            --learner $METHOD --dataset $DATASET \
            --prior_dir $PRIOR_DIR
    done
done