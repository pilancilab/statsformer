from pathlib import Path
import rdata
import pandas as pd

from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS, Dataset
from statsformer.models.base import ModelTask


def build_etp_dataset(
    rdata_path: str | Path,
    data_output_dir: str | Path,
    num_splits: int=10,
    train_ratios: list[float]=DEFAULT_TRAIN_RATIOS,
    seed: int=42,
):
    X, y = read_dataset(
        rdata_path, 'ETP',
        {1: "ETP", 0: "notETP"}
    )
    return Dataset.from_Xy(
        X=X,
        y=y,
        save_dir=data_output_dir,
        display_name="ETP",
        problem_type=ModelTask.BINARY_CLASSIFICATION,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
    )


def read_dataset(
    path, name, category_mapping,
    override_gene_names=None
):
    genenames_key = f'genenames{name}' \
        if override_gene_names is None else override_gene_names
    converted = rdata.read_rds(path)
    X = pd.DataFrame(
        converted[f'data{name}'][f'xall{name}'].to_numpy(),
        columns=converted[f'data{name}'][genenames_key]
    )
    y = pd.Series(converted[f'data{name}'][f'yall{name}'])
    y = y.map(category_mapping).astype(str)
    return X, y