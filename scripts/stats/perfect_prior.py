# keep this import adelie because of weirdness with openmp and openblas config
import adelie

import matplotlib
matplotlib.use("Agg") 

import os
from statsformer.models.glm import Lasso

from statsformer.baselines.ablations import NoLLMAblation
from statsformer.metalearning.base import ModelAndKwargs
from statsformer.metalearning.stacking import OOFStacking
from statsformer.models.xgboost import XGBoost
from statsformer.models.random_forest import RandomForest
from statsformer.models.kernel import WeightedKernelSVM

from statsformer.models.base import ModelTask
from statsformer.prior import FeaturePriorSweepConfig, FunctionalForm

from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import pandas as pd

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def get_data(
    n, p, p_eff, imbalance,
    label_noise_level=0.1,
    nonlinear="linear",   # "linear", "tanh", "featurewise"
    seed=42
):
    np.random.seed(seed)

    means = np.random.uniform(-10, 10, size=p)
    stds = np.random.uniform(0.5, 5.0, size=p)
    data = np.random.randn(n, p) * stds + means

    true_coefs = np.random.randn(p) * 0.1
    coef_mag = np.random.uniform(0.5, 5, size=p_eff)
    coef_sign = np.random.choice([-1, 1], size=p_eff)
    true_coefs[:p_eff] = coef_mag * coef_sign
    np.random.shuffle(true_coefs)

    idx = np.where(true_coefs != 0)[0]

    # ---- signal construction ----
    if nonlinear == "linear":
        signal = data @ true_coefs

    elif nonlinear == "tanh":
        signal = np.tanh((data @ true_coefs) / np.std(data @ true_coefs))

    elif nonlinear == "featurewise":
        signal = np.zeros(n)
        for j in idx:
            signal += true_coefs[j] * np.tanh(data[:, j])
    else:
        raise ValueError(f"Unknown nonlinear mode: {nonlinear}")

    # ---- imbalance + noise ----
    intercept = np.quantile(signal, imbalance)
    signal -= intercept

    noise = np.std(signal) * label_noise_level
    signal += np.random.randn(n) * noise

    y = (signal > 0).astype(np.int32)
    return data, y, true_coefs



TASK = ModelTask.BINARY_CLASSIFICATION

def instantiate_models(true_coefs):
    true_coefs = np.abs(true_coefs)
    base_learners = [
        ModelAndKwargs(
            Lasso,
            dict(task=TASK, default_folds_cv=5),
        ),
        ModelAndKwargs(
            XGBoost,
            dict(task=TASK, feature_weight_colsample_bynode=0.1)
        ),
        ModelAndKwargs(
            RandomForest,
            dict(task=TASK)
        ),
        ModelAndKwargs(
            WeightedKernelSVM,
            dict(task=TASK)
        )
    ]
    statsformer = OOFStacking(k=5, task=TASK)

    for model in base_learners:
        statsformer.add_model(
            model, priors=FeaturePriorSweepConfig(
                functional_form=FunctionalForm.POWER,
                temperatures=[0, 1, 2]
            ),
            scores=true_coefs
        )

    baseline = NoLLMAblation(
        task=TASK,
        nfeat=len(true_coefs),
        base_learners=base_learners,
        cv_k=5,
    )

    return statsformer, baseline


def run_trial(
    n, p, p_eff, imbalance, nonlinear="linear", seed=42
):
    data, y, true_coefs = get_data(
        n*2, p, p_eff, imbalance,
        nonlinear=nonlinear, seed=seed
    )
    statsformer, baseline = instantiate_models(true_coefs)

    # do one stratified 50/50 train/test split
    X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=seed, shuffle=True, stratify=y)

    statsformer.fit(X_train, y_train, random_seed=seed)
    baseline.fit(X_train, y_train, random_seed=seed)

    return dict(
        ours=statsformer.eval(X_test, y_test),
        theirs=baseline.eval(X_test, y_test)
    )


BLUE = "#3B82F6"   # nice modern blue
RED  = "#EF4444"   # nice modern red

def signed_histplot(x, title, xlabel, save_path=None):
    plt.figure(figsize=(10, 5))

    neg = x[x < 0]
    pos = x[x >= 0]

    # Use shared bins so the histograms line up
    bins = np.histogram_bin_edges(x, bins="auto")
    bin_widths = [bins[i+1] - bins[i] for i in range(len(bins)-1)]
    average_bin_width = np.mean(bin_widths)
    pos_bins = np.arange(0, pos.max() + average_bin_width, average_bin_width)

    sns.histplot(
        pos,
        bins=pos_bins,
        color=BLUE,
        stat="count",
        alpha=0.6,
        label="Positive"
    )

    if len(neg) > 0:
        neg_bins = -np.flip(np.arange(0, -neg.min() + average_bin_width, average_bin_width))
        sns.histplot(
            neg,
            bins=neg_bins,
            color=RED,
            stat="count",
            alpha=0.6,
            label="Negative"
        )

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def run_trials(
    n, p, p_eff, imbalance, nonlinear="linear",
    n_trials=10, base_seed=42
):
    results = {
        "our_acc": [],
        "their_acc": [],
        "our_auroc": [],
        "their_auroc": []
    }
    for _ in tqdm(range(n_trials)):
        res = run_trial(
            n, p, p_eff, imbalance,
            nonlinear=nonlinear,
            seed=base_seed
        )
        base_seed += 1

        ours, theirs = res["ours"], res["theirs"]
        results["our_acc"].append(ours.accuracy)
        results["our_auroc"].append(ours.auroc)
        results["their_acc"].append(theirs.accuracy)
        results["their_auroc"].append(theirs.auroc)
    return pd.DataFrame(results)


def main(
    save_dir,
    n: int, # number of points
    p: int, # number of features
    p_eff: int, # number of meaningful features
    imbalance: float=0.2, # percent of positive samples (from 0-1)
    nonlinear="linear",
    n_trials: int=20
):
    save_file = Path(save_dir)
    if not os.path.exists(save_file / "results.csv"):
        df = run_trials(
            n, p, p_eff, imbalance,
            nonlinear=nonlinear,
            n_trials=n_trials
        )

        df["acc_diff"] = df["our_acc"] - df["their_acc"]
        df["auroc_diff"] = df["our_auroc"] - df["their_auroc"]

        save_file.mkdir(exist_ok=True, parents=True)

        df.to_csv(save_file / "results.csv", index=False)
    else:
        df = pd.read_csv(save_file / "results.csv")
    signed_histplot(
        df["acc_diff"],
        title="Accuracy Difference (Our Method - Baseline)",
        xlabel="Accuracy Difference",
        save_path=save_file / "accuracy_difference.png"
    )

    signed_histplot(
        df["auroc_diff"],
        title="AUROC Difference (Our Method - Baseline)",
        xlabel="AUROC Difference",
        save_path=save_file / "auroc_difference.png"
    )
    print("done")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run synthetic data perfect prior experiments"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save results"
    )

    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of points"
    )

    parser.add_argument(
        "--p",
        type=int,
        required=True,
        help="Number of features"
    )

    parser.add_argument(
        "--p-eff",
        type=int,
        required=True,
        help="Number of meaningful features"
    )

    parser.add_argument(
        "--imbalance",
        type=float,
        default=0.2,
        help="Fraction of positive samples (0–1)"
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials"
    )

    parser.add_argument(
        "--nonlinear",
        type=str,
        default="linear",
        choices=["linear", "tanh", "featurewise"],
        help="Type of nonlinearity: linear, tanh, featurewise"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        save_dir=args.save_dir,
        n=args.n,
        p=args.p,
        p_eff=args.p_eff,
        imbalance=args.imbalance,
        n_trials=args.n_trials,
        nonlinear=args.nonlinear
    )
