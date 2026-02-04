from anyio import Path
import openml
from pandas import CategoricalDtype

from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS, Dataset
from statsformer.models.base import ModelTask


def build_internet_ads_dataset(
    data_output_dir: str | Path,
    num_splits: int=10,
    train_ratios: list[float]=DEFAULT_TRAIN_RATIOS,
    seed: int=42,
    max_dataset_size: int | None=1000,
):
    dataset = openml.datasets.get_dataset(40978)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    for key in ["height", "width", "aratio"]:
        X.loc[X[key] == 0, key] = X.loc[X[key] != 0, key].mean().__round__(4)
    
    for col in X.columns:
        if isinstance(X[col].dtype, CategoricalDtype) :
            X[col] = X[col].astype(float)
    
    return Dataset.from_Xy(
        X=X,
        y=y,
        save_dir=data_output_dir,
        display_name="Internet Ads",
        problem_type=ModelTask.BINARY_CLASSIFICATION,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
        crop_dataset_to_size=max_dataset_size
    )