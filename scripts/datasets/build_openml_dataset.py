
import argparse
from pathlib import Path

from enum import Enum

from statsformer.data.GEMLeR import build_ova_breast_cancer_dataset
from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS
from statsformer.data.internet_ads import build_internet_ads_dataset
from statsformer.data.nomao import build_nomao_dataset
from statsformer.data.credit_g import build_credit_g_dataset


class Datasets(Enum):
    NOMAO = "nomao"
    INTERNET_ADS = "internet_ads"
    BREAST_CANCER = "breast_cancer"
    CREDIT_G = "credit_g"


DATASET_TO_FUNCTION = {
    Datasets.NOMAO: build_nomao_dataset,
    Datasets.INTERNET_ADS: build_internet_ads_dataset,
    Datasets.BREAST_CANCER: build_ova_breast_cancer_dataset,
    Datasets.CREDIT_G: build_credit_g_dataset,
}


def main(
    dataset: Datasets,
    data_dir: str | Path="data",
    num_splits: int=10,
    train_ratios: list[float]=DEFAULT_TRAIN_RATIOS,
    max_dataset_size: int | None=1000,
    seed: int=42,
):
    data_dir = Path(data_dir)
    DATASET_TO_FUNCTION[dataset](
        data_output_dir=data_dir / "datasets" / dataset.value,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
        max_dataset_size=max_dataset_size
    )


if __name__ == "__main__":
    datasets = ", ".join([x for x in Datasets._value2member_map_])
    parser = argparse.ArgumentParser(
        description=f"Build an OpenML dataset (one of: {datasets}) with optional configuration parameters."
    )
    
    parser.add_argument(
        "--dataset",
        type=Datasets,
        required=True,
        help=f"Dataset to build (one of: {datasets}).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for dataset input/output (default: 'data').",
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
    parser.add_argument(
        "--max_dataset_size",
        type=int,
        default=1000,
        help="Maximum size to crop the dataset to (default: 1000).",
    )

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        data_dir=args.data_dir,
        num_splits=args.num_splits,
        train_ratios=args.train_ratios,
        seed=args.seed,
        max_dataset_size=args.max_dataset_size
    )
