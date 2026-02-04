from pathlib import Path
import pandas as pd
from pandas import CategoricalDtype

from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS, Dataset
from statsformer.models.base import ModelTask


def build_bank_marketing_dataset(
    csv_path: str | Path,
    data_output_dir: str | Path,
    num_splits: int = 10,
    train_ratios: list[float] = DEFAULT_TRAIN_RATIOS,
    seed: int = 42,
    max_dataset_size: int | None = None,
):
    data = pd.read_csv(csv_path, sep=';')
    y = data['y']
    X = data.drop(columns=['y'])
    
    for col in X.columns:
        if isinstance(X[col].dtype, CategoricalDtype):
            X[col] = pd.Categorical(X[col]).codes
        elif X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    y = y.astype(str)
    y = y.map({'yes': 'subscribed', 'no': 'not_subscribed'})
    
    return Dataset.from_Xy(
        X=X,
        y=y,
        save_dir=data_output_dir,
        display_name="Bank Marketing",
        problem_type=ModelTask.BINARY_CLASSIFICATION,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
        crop_dataset_to_size=max_dataset_size
    )


def build_superconductivity_dataset(
    csv_path: str | Path,
    data_output_dir: str | Path,
    num_splits: int = 10,
    train_ratios: list[float] = DEFAULT_TRAIN_RATIOS,
    seed: int = 42,
    max_dataset_size: int | None = None,
):
    data = pd.read_csv(csv_path)
    
    # Strip whitespace from column names if present
    data.columns = data.columns.str.strip()
    
    # Target column is 'critical_temp'
    y = data['critical_temp']
    
    # Drop the target column to get features
    X = data.drop(columns=['critical_temp'])
    
    # Convert categorical and object columns to numeric codes
    for col in X.columns:
        if isinstance(X[col].dtype, CategoricalDtype):
            X[col] = pd.Categorical(X[col]).codes
        elif X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    # For regression, keep y as numeric (no string conversion needed)
    # Dataset.from_Xy will handle regression appropriately
    
    return Dataset.from_Xy(
        X=X,
        y=y,
        save_dir=data_output_dir,
        display_name="Superconductivity",
        problem_type=ModelTask.REGRESSION,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
        crop_dataset_to_size=max_dataset_size
    )


def read_csv_dataset(
    csv_path: str | Path,
    label_column: str = 'y',
    label_mapping: dict | None = None,
):
    """
    Generic helper function to read a CSV dataset.
    
    Args:
        csv_path: Path to CSV file
        label_column: Name of the label column
        label_mapping: Optional mapping to convert labels to strings
    
    Returns:
        X: DataFrame with features
        y: Series with labels
    """
    data = pd.read_csv(csv_path)
    y = data[label_column]
    X = data.drop(columns=[label_column])
    
    for col in X.columns:
        if isinstance(X[col].dtype, CategoricalDtype):
            X[col] = pd.Categorical(X[col]).codes
        elif X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    if label_mapping:
        y = y.astype(str)
        y = y.map(label_mapping)
    else:
        y = y.astype(str)
    
    return X, y
