from copy import deepcopy
from statsformer.data.dataset import Dataset
from pathlib import Path
import numpy as np


def permute_feature_names(
    dataset: Dataset,
    save_dir: str=None,
    seed: int=42
) -> Dataset:
    feature_names = dataset.feature_names()
    np.random.seed(seed)

    original_dataset = dataset
    dataset = deepcopy(dataset)
    new_feature_names = np.random.permutation(feature_names)
    dataset.X.rename(
        {old: new for old, new in zip(feature_names, new_feature_names)},
        axis=1, inplace=True
    )
    if save_dir is None:
        original_dirname = Path(dataset.save_dir).name
        save_dir = Path(dataset.save_dir).parent / f"{original_dirname}_permuted_features"
    dataset.save_dir = save_dir

    assert original_dataset.feature_names() != dataset.feature_names()

    dataset.save()
    return dataset

