from dataclasses import asdict, dataclass
import json
from pathlib import Path

from statsformer.baselines.feature_selection import FeatureSelector
from statsformer.data.dataset import Dataset, TrainTest


@dataclass
class PreselectedFeatures:
    """
    Preselected features for a single train ratio, across all splits.
    """
    features_per_split: list[list[str]]
    train_ratio: float


@dataclass
class DatasetWithPreselection:
    dataset: Dataset
    preselected_features: list[PreselectedFeatures]
    save_dir: str | Path

    def get_features(self, ratio_idx: int, split_idx: int):
        return self.preselected_features[ratio_idx].features_per_split[split_idx]

    def get_split(self, ratio_idx: int, split_idx: int):
        data = self.dataset.get_split_pandas(ratio_idx, split_idx)
        features = self.get_features(ratio_idx, split_idx)
        return TrainTest(
            X_train=data.X_train[features].to_numpy(),
            y_train=data.y_train.to_numpy(),
            X_test=data.X_test[features].to_numpy(),
            y_test=data.y_test.to_numpy(),
            train_ratio=data.train_ratio
        )

    @property
    def problem_type(self):
        return self.dataset.problem_type
    
    @property
    def display_name(self):
        return self.dataset.display_name

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        save_dir: str | Path,
        num_features_per_split: int,
        preselector: FeatureSelector
    ) -> "DatasetWithPreselection":
        """
        Performs feature preselection on the given dataset using the provided
        feature selector. For each train/test split in the dataset, it selects
        the specified number of features and saves the resulting preselected
        features along with the dataset information to the specified directory.
        """
        preselected_features = []
        for ratio_idx in range(dataset.num_ratios()):
            feat_for_ratio = []
            for split_idx in range(dataset.num_splits()):
                train_test = dataset.get_split(ratio_idx, split_idx)
                feat_for_ratio.append(
                    [dataset.feature_names()[idx] for idx in \
                        preselector.select_features(
                            X=train_test.X_train,
                            y=train_test.y_train,
                            num_features=num_features_per_split,
                            task=dataset.problem_type
                        )
                    ]
                )
                train_ratio = train_test.train_ratio
            preselected_features.append(PreselectedFeatures(
                train_ratio=train_ratio,
                features_per_split=feat_for_ratio
            ))
        out = cls(
            dataset=dataset,
            preselected_features=preselected_features,
            save_dir=save_dir
        )
        out.save()
        return out
    
    def save(self):
        """
        Save the DatasetWithPreselection to disk.
        """
        print(f"[INFO] Saving features to {self.save_dir}")
        dir = Path(self.save_dir)
        dir.mkdir(exist_ok=True, parents=True)
        (dir / "datadir.txt").write_text(str(self.dataset.save_dir))
        (dir / "features_per_split.json").write_text(json.dumps([
            asdict(f) for f in self.preselected_features
        ], indent=2))
    
    def num_splits(self):
        return len(self.preselected_features[0].features_per_split)
    
    def num_ratios(self):
        return len(self.preselected_features)
    
    @classmethod
    def from_dir(
        cls,
        dir: str | Path,
    ) -> "DatasetWithPreselection":
        """
        Load a DatasetWithPreselection from disk.
        """
        dir = Path(dir)
        dataset = Dataset.from_dir((dir / "datadir.txt").read_text())
        features = [
            PreselectedFeatures(**f) for f in json.loads(
                (dir / "features_per_split.json").read_text()
            )
        ]
        return cls(
            dataset=dataset,
            preselected_features=features,
            save_dir=dir
        )

def load_dataset_or_preselected(dir: str | Path) -> Dataset | DatasetWithPreselection:
    if (Path(dir) / "features_per_split.json").exists():
        return DatasetWithPreselection.from_dir(dir)
    return Dataset.from_dir(dir)