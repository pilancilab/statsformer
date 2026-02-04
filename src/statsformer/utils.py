from dataclasses import asdict
from enum import Enum
import json
import os
from pathlib import Path
from scipy.special import logit
import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold


def dataclass_to_json(dclass, filename: str, indent=2):
    """
    Write a dataclass to a JSON file on disk.
    """
    Path(filename).write_text(json.dumps(asdict(dclass), indent=indent))


def plot_error_bars(
    ax,
    x: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    color,
    width=0.02,
    x_offset=0
):
    """
    Plots vertical error bars at positions x with given upper and lower bounds.
    """
    bar_width = width * (np.max(x) - np.min(x))
    x += x_offset * (np.max(x) - np.min(x))
    ax.vlines(x, lower, upper, colors=color)
    ax.hlines(upper, x - bar_width, x + bar_width, colors=color)
    ax.hlines(lower, x - bar_width, x + bar_width, colors=color)



def plot_format_generator(colors: list[str]):
    """
    Generator that yields (color, marker) tuples for plotting.
    """
    marker_styles = ["o", "v", "s", "X", "d"]
    idx = 0

    while True:
        if idx >= len(marker_styles) * len(colors):
            idx = 0
        yield colors[idx % len(colors)], marker_styles[idx // len(colors)]
        idx += 1


def clipped_logit(p: np.ndarray):
    """
    Computes the logit of probabilities p, clipping to avoid numerical issues.
    """
    p = np.clip(
         p, 1e-8, 1 - 1e-8
    )
    return logit(p)


def get_cross_validator(
    cv: int | BaseCrossValidator,
    y: np.ndarray,
    is_classification: bool,
    oversample: bool=False,
    seed: int=42
) -> BaseCrossValidator:
        """
        Gets the proper cross-validation splitter based on the task type
        (e.g., stratified for classification)
        """
        if isinstance(cv, BaseCrossValidator):
            return cv

        n_folds = min(cv, max(
            len(y) // 10, 2
        ))

        # find the count of the least frequent class
        if is_classification:
            min_class_count = np.min(
                np.bincount(y.astype(int).ravel())
            )
            n_folds = min(n_folds, min_class_count)
            if oversample:
                return OversampledStratifiedKFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=seed
                )
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            return KFold(n_splits=n_folds, shuffle=True, random_state=seed)


def get_oversample_indices(
    y: np.ndarray,
    max_oversample_factor: int=2,
    random_seed: int=42
):
    indices_per_class = {
        c: np.where(y == c)[0].tolist()
        for c in np.unique(y)
    }
    class_counts = {
        c: len(indices_per_class[c]) for c in indices_per_class
    }
    max_count = max(class_counts.values())
    oversample_rate = {
        c: min(max_oversample_factor, max_count / class_counts[c]) \
            for c in indices_per_class
    }
    
    desired_per_class = {
        c: int(class_counts[c] * oversample_rate[c]) for c in class_counts
    }

    rng = np.random.default_rng(random_seed)

    oversampled_idxs = []
    for c in indices_per_class:
        ic = indices_per_class[c]
        oversampled_idxs += list(ic) + list(rng.choice(
            ic, size=desired_per_class[c] - len(ic),
            replace=True
        ))

    return oversampled_idxs


class OversampledStratifiedKFold(StratifiedKFold):
    def __init__(
        self,
        max_oversample_factor: int=2,
        random_state: int=42,
        **kwargs
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.max_oversample_factor = max_oversample_factor
        self.random_state = random_state

    def split(self, X, y, groups = None):
        splits = super().split(X, y, groups)
        for i, split in enumerate(splits):
            train, test = split
            train_idxs = [train[i] for i in get_oversample_indices(
                y[train], self.max_oversample_factor,
                random_seed=self.random_state + i
            )]
            test_idxs = [test[i] for i in get_oversample_indices(
                y[test], self.max_oversample_factor,
                random_seed=self.random_state + i
            )]
            yield train_idxs, test_idxs


def find_first_dir_with_filename(
    base: str, search_file: str
):
    stack = [f"{base}/{d}" for d in os.listdir(base)]
    valid_dir_found = False
    while len(stack) > 0 and not valid_dir_found:
        curr_dir = stack.pop()
        if os.path.exists(f"{curr_dir}/{search_file}"):
            valid_dir_found = True
            break # we found a valid prior dir
        subdirs = os.listdir(curr_dir)
        stack.extend([f"{curr_dir}/{d}" for d in subdirs])
    if not valid_dir_found:
        raise ValueError(f"No valid directory found with {search_file}")
    return curr_dir