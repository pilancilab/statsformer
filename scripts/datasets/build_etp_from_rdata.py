
import argparse
from pathlib import Path

from enum import Enum

from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS
from statsformer.data.from_rdata import build_etp_dataset


def main(
    data_dir: Path=Path("data"),
    rdata_dir: Path=Path("data/rdata"),
    num_splits: int=10,
    train_ratios: list[float]=DEFAULT_TRAIN_RATIOS,
    seed: int=42,
):
    build_etp_dataset(
        rdata_path=rdata_dir / "ETP.RData",
        data_output_dir=data_dir / "datasets" / "ETP",
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Build ETP dataset from RData with optional configuration parameters."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for dataset input/output (default: 'data').",
    )
    parser.add_argument(
        "--rdata_dir",
        type=str,
        default="data/rdata",
        help="Base directory for rdata input files (default: 'data/rdata').",
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
        rdata_dir=Path(args.rdata_dir),
        data_dir=Path(args.data_dir),
        num_splits=args.num_splits,
        train_ratios=args.train_ratios,
        seed=args.seed,
    )
