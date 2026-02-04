# Do this at the very beginning because the openblas/openmp configuration
# from adelie sometimes interacts badly with other libraries (e.g., xgboost),
# making adelie sometimes very slow. Importing it first seems to help.
# It seems like other libraries adapt to adelie's config, but adelie doesn't
# adapt to other libraries
#
# **IMPORTANT**: TLDR; IMPORT adelie FIRST!
import adelie

import argparse
import os
from statsformer.adversarial.prior import AdversarialTransformation, TransformedPrior
from statsformer.baselines.ablations import LLMLassoBaseline, NoLLMAblation
from statsformer.baselines.data_driven import LassoBaseline
from statsformer.models.glm import Lasso 

from statsformer.data.preselection import DatasetWithPreselection
from statsformer.experiment.base import StatsformerConfig, Baseline, Experiment
from statsformer.baselines.data_driven import (
    LightGBMBaseline, XGBoostBaseline, RandomForestBaseline, AutoGluonBaseline
)
from statsformer.llm.with_preselection import GeneratedPriorWithPreselection
from statsformer.metalearning.base import ModelAndKwargs
from statsformer.metalearning.stacking import StackingType
from statsformer.models.kernel import WeightedKernelSVM
from statsformer.models.random_forest import RandomForest
from statsformer.models.xgboost import XGBoost
from statsformer.models.base import CVMetrics, ModelTask
from statsformer.utils import find_first_dir_with_filename

from statsformer.data.dataset import Dataset
from statsformer.llm.generated_prior import GeneratedPrior
from statsformer.prior import FeaturePriorSweepConfig, FunctionalForm


def get_lasso_kwargs(task, cv_k):
     return dict(
        task=task,
        lambda_min_ratio=0.01,
        lambda_path_size=100,
        default_folds_cv=cv_k
    )


def get_baselines(
    task: ModelTask,
    prior: GeneratedPrior,
    base_learners: list[ModelAndKwargs],
    cv_k: int=5
):
    return [
        Baseline(
            model=XGBoostBaseline(task=task),
            name="xgboost",
            display_name="XGBoost"
        ),
        Baseline(
            model=LassoBaseline(
                **get_lasso_kwargs(task, cv_k)
            ),
            name="lasso",
            display_name="LASSO"
        ),
        Baseline(
            model=LightGBMBaseline(
                task=task
            ),
            name="lightgbm",
            display_name="LightGBM"
        ),
        Baseline(
            model=NoLLMAblation(
                task=task,
                nfeat=len(prior.get_scores()),
                base_learners=base_learners,
                cv_k=cv_k,
            ),
            name="no_llm",
            display_name="Stacking"
        ),
        Baseline(
            model=LLMLassoBaseline(
                temperatures=[0, 1, 2],
                prior=prior,
                cv_k=cv_k,
                **get_lasso_kwargs(task, cv_k)
            ),
            name="llm_lasso",
            display_name="LLM-Lasso"
        ),
        Baseline(
            model=RandomForestBaseline(task=task),
            name="rand_forest",
            display_name="Random Forest"
        ),
        Baseline(
            model=AutoGluonBaseline(
                task=task,
                time_limit=120,
            ),
            name="autogluon",
            display_name="AutoGluon"
        ),
    ]

def main(
    trial_name: str,
    dataset_name: str,
    preselected: bool=False,
    prior_dir: str=None,
    rerun=False,
    cv_k: int=5,
    num_threads: int=8,
    seed=42,
    adversarial: bool=False
):
    if prior_dir is None:
        prior_dir = (
            f"data/generated_priors/{dataset_name}/RAG_OMIM/"
            "o3__temp_0__batch_40/"
            "default_prompt__default_system_prompt/initial_scores"
        ) # see if we have o3 scores

        if not os.path.exists(prior_dir):
            prior_dir = find_first_dir_with_filename(
                f"data/generated_priors/{dataset_name}/",
                "metadata.json"
            )

    if preselected:
        preselection = DatasetWithPreselection.from_dir(f"data/preselected_datasets/{dataset_name}")
        dataset = preselection.dataset
    else:
        dataset = Dataset.from_dir(f"data/datasets/{dataset_name}")
        preselection = None
    task = dataset.problem_type

    print(f"Using priors from: {prior_dir}")

    prior = GeneratedPrior.from_dir(prior_dir)

    if adversarial:
        prior = TransformedPrior(
            prior,
            transformation=AdversarialTransformation.INVERSE,
        )
    non_preselected_prior = prior
    if preselected:
        prior = GeneratedPriorWithPreselection(
            dataset, prior
        )

    models = [
        ModelAndKwargs(
            Lasso,
            get_lasso_kwargs(task, cv_k)
        ),
        ModelAndKwargs(
            XGBoost,
           dict(task=task)
        ),
        ModelAndKwargs(
            RandomForest,
            dict(task=task)
        ),
        ModelAndKwargs(
            WeightedKernelSVM,
            dict(
                task=task
            )
        )
    ]
    baselines = get_baselines(
        task, non_preselected_prior,
        base_learners=models, cv_k=cv_k
    )

    baseline_output_dir = f"data/experiments/baselines/{dataset_name}"

    if adversarial:
        baselines = [
            b for b in baselines if b.name == "no_llm"
        ]

    configs = [
        StatsformerConfig(
            models=models,
            name="statsformer",
            display_name="Statsformer",
            cv_k=cv_k,
            prior=prior,
            feature_prior_sweep=FeaturePriorSweepConfig(
                FunctionalForm.POWER,
                temperatures=[0, 1, 2],
                betas=[0.75, 1],
            ),
            preselection=preselection,
            cv_metric=CVMetrics.MCC,
        ),
    ]

    base_output_dir = f"data/experiments/with_baselines/{trial_name}/{dataset_name}"

    expt_kwargs = dict(
        dataset=dataset,
        base_output_dir=base_output_dir,
        baselines_output_dir=baseline_output_dir,
        baselines=baselines,
        statsformer_configs=configs,
        num_threads=num_threads,
        seed=seed
    )

    experiment = Experiment(**expt_kwargs)

    experiment.run_baselines()
    experiment.run_statsformer(rerun=rerun)

    # plot with baselines
    expt_kwargs["baselines"] = [
        b for b in baselines if b.name != "no_llm"
    ]
    experiment = Experiment(**expt_kwargs)

    plot_kwargs = dict()
    if task.is_classification():
        plot_filenames = [
            experiment.plot_accuracy(error_bars=True, **plot_kwargs),
            experiment.plot_auroc(error_bars=True, **plot_kwargs),
            experiment.plot_accuracy(error_bars=False, **plot_kwargs),
            experiment.plot_auroc(error_bars=False, **plot_kwargs)
        ]
    else:
        plot_filenames = [
            experiment.plot_mse(error_bars=True, **plot_kwargs),
            experiment.plot_mse(error_bars=False, **plot_kwargs)
        ]
    for fname in plot_filenames:
        print(f"Plot saved to: {fname}")

    # just plot us compared to no_llm
    expt_kwargs["baselines"] = [
        b for b in baselines if b.name == "no_llm"
    ]
    experiment = Experiment(**expt_kwargs)

    if task.is_classification():
        plot_filenames = [
            experiment.plot_accuracy(
                error_bars=True, custom_plot_name="statsformer_vs_no_llm_accuracy"),
            experiment.plot_auroc(
                error_bars=True, custom_plot_name="statsformer_vs_no_llm_auroc"),
            experiment.plot_accuracy(
                error_bars=False, custom_plot_name="statsformer_vs_no_llm_accuracy"),
            experiment.plot_auroc(
                error_bars=False, custom_plot_name="statsformer_vs_no_llm_auroc"),
            experiment.plot_win_ratio(
                our_method_name="statsformer",
                their_method_name="no_llm",
                metric_names=["auroc", "accuracy"],
                metric_display_names=["AUROC", "Accuracy"],
                save_filename="win_ratio"
            ),
        ]
    else:
        plot_filenames = [
            experiment.plot_mse(
                error_bars=True, custom_plot_name="statsformer_vs_no_llm_mse"),
            experiment.plot_mse(
                error_bars=False, custom_plot_name="statsformer_vs_no_llm_mse"),
            experiment.plot_win_ratio(
                our_method_name="statsformer",
                their_method_name="no_llm",
                metric_names=["mse"],
                metric_display_names=["MSE"],
                save_filename="win_ratio"
            ),
        ]
    for fname in plot_filenames:
        print(f"Plot saved to: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")

    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name",
    )
    parser.add_argument(
        "--preselected",
        action="store_true",
        help="Whether using preselection.",
    )
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="Directory containing prior information",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun even if results already exist",
    )
    parser.add_argument(
        "--cv_k",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of worker threads",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--trial_name", type=str,
        default="test",
        help="Determines the output directory for plots, etc."
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Invert scores, and then just plot stacking baseline and statsformer",
    )

    args = parser.parse_args()

    main(
        trial_name=args.trial_name,
        dataset_name=args.dataset_name,
        preselected=args.preselected,
        prior_dir=args.prior_dir,
        rerun=args.rerun,
        cv_k=args.cv_k,
        num_threads=args.num_threads,
        seed=args.seed,
        adversarial=args.adversarial
    )
