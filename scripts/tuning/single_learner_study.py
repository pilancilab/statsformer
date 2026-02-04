# Do this at the very beginning because the openblas/openmp configuration
# from adelie sometimes interacts badly with other libraries (e.g., xgboost),
# making adelie sometimes very slow. Importing it first seems to help.
# It seems like other libraries adapt to adelie's config, but adelie doesn't
# adapt to other libraries
#
# **IMPORTANT**: TLDR; IMPORT adelie FIRST!
# import adelie

from enum import Enum
import os
from statsformer.data.dataset import Dataset
from statsformer.data.preselection import load_dataset_or_preselected
from statsformer.experiment.base_learner_study import BaseLearnerStudy
from statsformer.llm.generated_prior import GeneratedPrior
from statsformer.llm.with_preselection import GeneratedPriorWithPreselection
from statsformer.metalearning.base import ModelAndKwargs
from statsformer.models.base import CVMetrics
from statsformer.prior import FeatureMap, FeaturePriorConfig, FeaturePriorSweepConfig, FunctionalForm
from statsformer.utils import find_first_dir_with_filename
import argparse

########################################################
###### HERE ARE SOME PARAMETERS THAT CAN BE TUNED ######
########################################################
CV_METRIC = CVMetrics.MCC
LASSO_CV_METRIC = CVMetrics.LOSS
OVERSAMPLE_CV = False

# Lasso
LAMBDA_PATH_SIZE = 100 # number of values in the regularization path
LAMBDA_MIN_RATIO = 0.01 # ratio of smallest lambda to largest

# XGBoost
XGBOOST_SAMPLE_WEIGHTS = False
PER_NODE_SUBSAMPLE = 0.1 # At each node of the base learner (decision tree),
                         # we subsample the features by this factor. Must be
                         # <= 1 for feature weights to work
NUM_BOOSTING_ROUNDS = 50

PICK_ONE_SCORE_PER_LEARNER = False

# random forest
NUM_TREES = 50
########################################################


class BaseLearners(Enum):
    LASSO = "lasso"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    KERNEL = "kernel"

    def format_name(self):
        if self == BaseLearners.LASSO:
            return "LASSO"
        elif self == BaseLearners.XGBOOST:
            return "XGBoost"
        elif self == BaseLearners.RANDOM_FOREST:
            return "Random Forest"
        elif self == BaseLearners.KERNEL:
            return "Kernel SVM"
        else:
            raise NotImplementedError(f"Unknown base learner {self}")

    def get_model_and_kwargs(self, task):
        if self == BaseLearners.LASSO:
            from statsformer.models.glm import Lasso
            return ModelAndKwargs(
                Lasso,
                dict(
                    task=task, cv_metric=LASSO_CV_METRIC,
                    lambda_min_ratio=LAMBDA_MIN_RATIO,
                    lambda_path_size=LAMBDA_PATH_SIZE,
                    oversample_cv=OVERSAMPLE_CV
                )
            )
        elif self == BaseLearners.XGBOOST:
            from statsformer.models.xgboost import XGBoost
            return ModelAndKwargs(
                XGBoost,
                dict(
                    task=task, add_feature_weights=True,
                    add_instance_weights=XGBOOST_SAMPLE_WEIGHTS,
                    feature_weight_colsample_bynode=PER_NODE_SUBSAMPLE,
                    num_boost_round=NUM_BOOSTING_ROUNDS
                )
            )
        elif self == BaseLearners.RANDOM_FOREST:
            from statsformer.models.random_forest import RandomForest
            return ModelAndKwargs(
                RandomForest,
                dict(
                    task=task,
                    num_estimators=NUM_TREES,
                )
            )
        elif self == BaseLearners.KERNEL:
            from statsformer.models.kernel import WeightedKernelSVM
            return ModelAndKwargs(
                WeightedKernelSVM,
                dict(
                    task=task
                )
            )
        else:
            raise NotImplementedError(f"Unknown base learner {self}")


def main(
    trial_name: str,
    learner=BaseLearners.LASSO,
    dataset_name: str="lung_TCGA",
    preselected: bool=False,
    prior_dir: str=None,
    rerun=False,
    cv_k: int=5,
    num_threads: int=8,
    seed=42,
    just_plot: bool=False
):  
    dataset_dir = "datasets" if not preselected else "preselected_datasets"
    dataset = load_dataset_or_preselected(f"data/{dataset_dir}/{dataset_name}")

    base_data_name = dataset_name if not preselected else os.path.basename(
        os.path.normpath(dataset.dataset.save_dir)
    )
    if prior_dir is None:
        prior_dir = (
            f"data/generated_priors/{base_data_name}/RAG_OMIM/"
            "o3__temp_0__batch_40/"
            "default_prompt__default_system_prompt/initial_scores"
        ) # see if we have o3 scores
        if not os.path.exists(prior_dir):
            prior_dir = find_first_dir_with_filename(
                f"data/generated_priors/{base_data_name}/",
                "metadata.json"
            )
    print(f"Using priors from: {prior_dir}")
    prior = GeneratedPrior.from_dir(prior_dir)
    if preselected:
        prior = GeneratedPriorWithPreselection(
            dataset, prior
        )

    task = dataset.problem_type

    model = learner.get_model_and_kwargs(task=task)

    base_output_dir = f"data/experiments/atomic_study/{trial_name}/{learner.value}/{dataset_name}"

    sweep_config = FeaturePriorSweepConfig(
        FunctionalForm.POWER,
        temperatures=[0, 1, 2],
        feature_map=FeatureMap.IDENTITY
    )
    if just_plot:
        sweep_config = FeaturePriorSweepConfig(
            FunctionalForm.POWER,
            temperatures=[0],
        )

    experiment = BaseLearnerStudy(
        dataset=dataset,
        model=model,
        cv_metric=CV_METRIC,
        prior=prior,
        base_output_dir=base_output_dir,
        feature_prior_sweep=sweep_config,
        cv_k=cv_k,
        num_threads=num_threads,
        seed=seed,
        oversample_cv=OVERSAMPLE_CV
    )

    if not just_plot:
        print("\nRunning experiments...")
        experiment.run_all(rerun=rerun)

    print("\nGenerating plots...")
    plot_kwargs = dict(extra_title=learner.format_name())
    if task.is_classification():
        plot_filenames = [
            experiment.plot_accuracy(error_bars=True, **plot_kwargs),
            experiment.plot_auroc(error_bars=True, **plot_kwargs),
            experiment.plot_accuracy(error_bars=False, **plot_kwargs),
            experiment.plot_auroc(error_bars=False, **plot_kwargs),
        ]
    else:
        plot_filenames = [
            experiment.plot_mse(error_bars=True, **plot_kwargs),
            experiment.plot_mse(error_bars=False, **plot_kwargs)
        ]
    plot_filenames += [
        experiment.plot_win_ratio(
            "oof_positive",
            "InverseImportance_T_0",
            metric_names=["auroc", "accuracy"],
            metric_display_names=["AUROC", "Accuracy"],
            save_filename="win_ratio"
        ),
    ]

    for fname in plot_filenames:
        print(f"Plot saved to: {fname}")


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Run single model study with generated prior.")
    parser.add_argument("--learner", type=str, default="lasso",
                        choices=[e.value for e in BaseLearners], help="Base learner to use")
    parser.add_argument("--dataset", type=str, default="lung_TCGA", help="Name of the dataset")
    parser.add_argument("--preselected", action="store_true", help="Whether to look for the dataset in the preselected folder")
    parser.add_argument("--prior_dir", type=str, default=None, help=(
        "Directory of the prior. Defaults to o3 scores, or the first scores "
        "found if o3 does not exist."
    ))
    parser.add_argument("--trial_name", type=str, default="initial_test", help="Determines the output directory for plots, etc.")
    parser.add_argument("--rerun", action="store_true", help="Rerun experiments")
    parser.add_argument("--cv_k", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--just_plot", action="store_true", help="Only generate plots, do not run experiments")
    args = parser.parse_args()

    main(
        trial_name=args.trial_name,
        learner=BaseLearners(args.learner),
        dataset_name=args.dataset,
        preselected=args.preselected,
        prior_dir=args.prior_dir,
        rerun=args.rerun,
        cv_k=args.cv_k,
        num_threads=args.num_threads,
        seed=args.seed,
        just_plot=args.just_plot
    )

if __name__ == "__main__":
    parse_args_and_run()