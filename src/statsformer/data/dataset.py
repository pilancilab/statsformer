from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from statsformer.models.base import ModelTask


# For subsampling training data (i.e., plotting training data ratio vs
# accuracy)
DEFAULT_TRAIN_RATIOS = list([
    x / 10 for x in range(1, 9)
])


@dataclass
class Split:
    """
    Given a global dataset, contains the indices for a train/test split.
    """
    train_idxs: list[int]
    test_idxs: list[int]


@dataclass
class SplitsPerRatio:
    """
    For a given subsampling ratio, contains multiple train/test splits, results
    for which can be averaged.
    """
    train_ratio: float
    splits: list[Split]

    def save(self, dirname: str | Path):
        """
        Save the splits to disk.
        """
        subdir = Path(dirname) / f"ratio_{self.train_ratio}_splits/"
        out_file_train = subdir / "train_idxs.npy"
        out_file_test = subdir / "test_idxs.npy"
        subdir.mkdir(exist_ok=True, parents=True)

        train_idxs = np.array(
            [x.train_idxs for x in self.splits]
        )
        test_idxs = np.array(
            [x.test_idxs for x in self.splits]
        )
        np.save(out_file_train, train_idxs)
        np.save(out_file_test, test_idxs)
    
    @classmethod
    def from_dir(cls, dirname: str | Path):
        """
        Load splits from disk.
        """
        dirname = Path(dirname)
        train_ratio = float(re.match(r".*/ratio_(.*)_splits.*", str(dirname)).group(1))

        train_idxs = np.load(dirname / "train_idxs.npy")
        test_idxs = np.load(dirname / "test_idxs.npy")

        return cls(
            train_ratio=train_ratio,
            splits=[
                Split(
                    train_idxs=list(train_idxs[i, :]),
                    test_idxs=list(test_idxs[i, :])
                ) for i in range(train_idxs.shape[0])
            ]
        )


@dataclass
class TrainTest:
    """
    Contains train/test split data as numpy arrays.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_ratio: float


@dataclass
class TrainTestPandas:
    """
    Contains train/test split data as pandas DataFrames/Series.
    """
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_ratio: float


@dataclass
class Dataset:
    """
    Stores a dataset along with precomputed train/test splits in a standardized
    format.
    """
    X: pd.DataFrame # full data
    y: pd.Series # full labels
    problem_type: ModelTask # e.g., classification, regression
    class_labels: list[str] # e.g., ["healthy", "diseased"]
    display_name: str # for plotting
    splits: list[SplitsPerRatio]
    save_dir: str

    def get_split(self, ratio_idx: int, split_idx: int):
        split = self.splits[ratio_idx].splits[split_idx]
        train_ratio = self.splits[ratio_idx].train_ratio

        X = self.X.to_numpy()
        y = self.y.to_numpy()
        return TrainTest(
            X_train=X[split.train_idxs, :],
            y_train=y[split.train_idxs],
            X_test=X[split.test_idxs],
            y_test=y[split.test_idxs],
            train_ratio=train_ratio
        )

    def get_split_pandas(
        self, ratio_idx: int, split_idx: int
    ) -> TrainTestPandas:
        split = self.splits[ratio_idx].splits[split_idx]
        train_ratio = self.splits[ratio_idx].train_ratio

        X = self.X
        y = self.y
        return TrainTestPandas(
            X_train=X.iloc[split.train_idxs],
            y_train=y.iloc[split.train_idxs],
            X_test=X.iloc[split.test_idxs],
            y_test=y.iloc[split.test_idxs],
            train_ratio=train_ratio
        )
    
    def feature_names(self) -> list[str]:
        return self.X.columns.tolist()
    
    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape
    
    def __len__(self):
        return len(self.y)
    
    def num_ratios(self):
        return len(self.splits)
    
    def num_splits(self):
        return len(self.splits[0].splits)

    @classmethod
    def from_Xy(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: ModelTask,
        display_name: str,
        save_dir: str | Path,
        train_ratios: list[float]=DEFAULT_TRAIN_RATIOS,
        test_ratio: float=0.2,
        num_splits: int=10,
        seed: int=42,
        crop_dataset_to_size: int | None=None,
    ) -> "Dataset":
        """
        Create a Dataset from feature matrix X and label vector y, along with
        precomputed train/test splits at various training data ratios.

        Saves the dataset to disk at `save_dir`, so that it can be reloaded later.
        """
        if problem_type.is_classification():
            class_labels = list(y.unique())
            # Map labels to integers 0, 1, ...
            replace_dict = {label: i for (i, label) in enumerate(class_labels)}
            y = y.map(replace_dict).astype(int)
        else:
            # No label mapping for regression
            class_labels = []
        y = y.rename("label")


        if problem_type.is_classification():
            min_class_count = y.value_counts().min()
            # Every train split needs at least 2 training data points from
            # each class
            min_ratio_required = 4 / min_class_count
        
            train_ratios = [r for r in train_ratios if r >= min_ratio_required]
            test_ratio = max(test_ratio, min_ratio_required)          

        splits: list[SplitsPerRatio] = []
        train_ratios = [r for r in train_ratios if (r > 0 and r + test_ratio <= 1)]
        train_ratios = sorted(train_ratios)

        assert test_ratio + max(train_ratios) <= 1

        # If cropping the dataset, determine the ratio to keep
        keep_ratio = 1 if crop_dataset_to_size is None else \
            min(crop_dataset_to_size / len(y), 1)

        # For each training data ratio, compute train/test splits
        for train_ratio in train_ratios:
            splitter = _get_splitter(
                problem_type=problem_type,
                num_splits=num_splits,
                test_size=test_ratio * keep_ratio,
                train_size=train_ratio * keep_ratio,
                seed=seed
            )
            splits.append(SplitsPerRatio(
                train_ratio=train_ratio,
                splits=[
                    Split(train, test) for (train, test) in splitter.split(X, y)
                ]
            ))
        
        # If cropping the dataset, remap indices to only those used
        # so that we don't store unused data
        used_idxs = []
        for splits_per_ratio in splits:
            for split in splits_per_ratio.splits:
                used_idxs += list(split.train_idxs) + list(split.test_idxs)
        used_idxs = set(used_idxs)
        if len(used_idxs) < len(y):
            used_idxs = sorted(list(used_idxs))
            mapping = {idx: i for (i, idx) in enumerate(used_idxs)}
            for splits_per_ratio in splits:
                for split in splits_per_ratio.splits:
                    split.train_idxs = [mapping[i] for i in split.train_idxs]
                    split.test_idxs = [mapping[i] for i in split.test_idxs]

            X = X.iloc[used_idxs].reset_index(drop=True)
            y = y[used_idxs].reset_index(drop=True)
        
        # Build and save dataset
        dataset = cls(
            X=X,
            y=y,
            class_labels=class_labels,
            display_name=display_name,
            splits=splits,
            problem_type=problem_type,
            save_dir=save_dir,
        )
        dataset.save()
        return dataset
    
    @classmethod
    def from_dir(cls, dir: str | Path):
        """
        Load a Dataset from disk.
        """
        dir = Path(dir)
        X = pd.read_csv(dir / "X.csv")
        y = pd.read_csv(dir / "y.csv")

        metadata = json.loads((dir / "metadata.json").read_text())
        class_labels = metadata["labels"]
        problem_type = ModelTask(metadata["problem_type"])
        display_name = metadata["display_name"]
        
        splits = [
            SplitsPerRatio.from_dir(dirname) for dirname in (dir / "splits").glob("*/")
        ]
        splits.sort(key=lambda split: split.train_ratio)

        return cls(
            X=X,
            y=y,
            class_labels=class_labels,
            display_name=display_name,
            splits=splits,
            problem_type=problem_type,
            save_dir=dir
        )
    
    def save(self):
        """
        Save the Dataset to disk.
        """
        print(f"[INFO] Saving dataset to {self.save_dir}")
        dir = Path(self.save_dir)
        # Remove data if it was already there
        if os.path.exists(dir):
            shutil.rmtree(dir)
        dir.mkdir(exist_ok=True, parents=True)
        self.X.to_csv(dir / "X.csv", index=False)
        self.y.to_csv(dir / "y.csv", index=False)

        for splits in self.splits:
            splits.save(dir / "splits")

        (dir / "metadata.json").write_text(json.dumps(
            {
                "labels": self.class_labels,
                "problem_type": self.problem_type.value,
                "display_name": self.display_name
            }, indent=2
        ))


def _get_splitter(
    problem_type: ModelTask,
    num_splits: int,
    test_size: float | None=None,
    train_size: float | None=None,
    seed: int=42
):
    """
    Returns a train/test splitter: stratified for classification tasks,
    standard for regression tasks.
    """
    if problem_type.is_classification():
        return StratifiedShuffleSplit(
            n_splits=num_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=seed
        )
    else:
        return ShuffleSplit(
            n_splits=num_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=seed
        )