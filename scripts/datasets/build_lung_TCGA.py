from pathlib import Path
import argparse

import pandas as pd

from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS, Dataset
from statsformer.models.base import ModelTask


def main(
    data_dir: str | Path="data",
    num_threads: int=20,
    num_splits: int=10,
    train_ratios: list[float]=DEFAULT_TRAIN_RATIOS,
    seed: int=42,
):
    data_dir = Path(data_dir)
    
    vst_mat_filter = pd.read_csv("data/csv/lung_TCGA/vst_mat_filter.csv")
    metadata = pd.read_csv("data/csv/lung_TCGA/metadata.csv")

    return Dataset.from_Xy(
        X=vst_mat_filter,
        y=metadata["condition"],
        save_dir=data_dir / "datasets" / "lung_TCGA",
        display_name="LUAD vs. LUSC",
        problem_type=ModelTask.BINARY_CLASSIFICATION,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Lung TCGA dataset with optional configuration parameters."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for dataset input/output (default: 'data').",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=20,
        help="Number of threads to use for processing (default: 20).",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=10,
        help="Number of splits for dataset partitioning (default: 10).",
    )
    parser.add_argument(
        "--train_ratios",
        type=float,
        nargs='+',
        default=DEFAULT_TRAIN_RATIOS,
        help="Fraction of data to use for testing (default: [0.1, 0.2, ..., 0.8]).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        num_threads=args.num_threads,
        num_splits=args.num_splits,
        train_ratios=args.train_ratios,
        seed=args.seed,
    )
