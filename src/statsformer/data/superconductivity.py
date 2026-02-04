from pathlib import Path

from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS, Dataset
from statsformer.data.from_csv import build_superconductivity_dataset as _build_superconductivity_dataset


def build_superconductivity_dataset(
    csv_path: str | Path,
    data_output_dir: str | Path,
    num_splits: int = 10,
    train_ratios: list[float] = DEFAULT_TRAIN_RATIOS,
    seed: int = 42,
    max_dataset_size: int | None = None,
):
    return _build_superconductivity_dataset(
        csv_path=csv_path,
        data_output_dir=data_output_dir,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
        max_dataset_size=max_dataset_size
    )
