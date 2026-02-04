from pathlib import Path

from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS, Dataset
from statsformer.data.from_csv import build_bank_marketing_dataset as _build_bank_marketing_dataset


def build_bank_marketing_dataset(
    csv_path: str | Path,
    data_output_dir: str | Path,
    num_splits: int = 10,
    train_ratios: list[float] = DEFAULT_TRAIN_RATIOS,
    seed: int = 42,
    max_dataset_size: int | None = None,
):
    """
    Build the Bank Marketing dataset from CSV file.
    
    The dataset is related to direct marketing campaigns (phone calls) of a 
    Portuguese banking institution. The classification goal is to predict if 
    the client will subscribe a term deposit.
    
    Args:
        csv_path: Path to CSV file containing the dataset
        data_output_dir: Where to save the processed dataset
        num_splits: Number of train/test splits per ratio
        train_ratios: List of training data ratios
        seed: Random seed for reproducibility
        max_dataset_size: Optional max number of samples (crops if larger)
    """
    return _build_bank_marketing_dataset(
        csv_path=csv_path,
        data_output_dir=data_output_dir,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
        max_dataset_size=max_dataset_size
    )
