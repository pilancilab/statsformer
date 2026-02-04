

import openml
from pyparsing import Path
from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS, Dataset
import requests
import pandas as pd
from io import StringIO

from statsformer.models.base import ModelTask


GENE_MAPPING_LINK = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?mode=raw&is_datatable=true&acc=GPL570&id=55999&db=GeoDb_blob143"


def build_ova_breast_cancer_dataset(**kwargs):
    return build_GEMLeR_dataset(
        display_name="OVA Breast Cancer vs. Other",
        openml_id=1128,
        **kwargs
    )


def get_gene_mapping() -> dict[str, str]:
    response = requests.get(GENE_MAPPING_LINK)
    data = "\n".join([line for line in response.text.split("\n") if not line.startswith("#")])
    X_platform = pd.read_csv(StringIO(data), sep="\t", usecols=["ID", "Gene Symbol"])
    X_platform["Gene Symbol"] = X_platform["Gene Symbol"].str.split(" /// ").str[0]  # take first symbol if multiple

    # drop rows where ID starts with "AFFX-"
    X_platform = X_platform[~X_platform["ID"].str.startswith("AFFX-")]
    X_platform = X_platform.drop(X_platform[X_platform["Gene Symbol"].isna()].index)
    return {row["ID"]: row["Gene Symbol"] for _, row in X_platform.iterrows()}


def build_GEMLeR_dataset(
    data_output_dir: str | Path,
    display_name: str,
    openml_id: int,
    num_splits: int=10,
    train_ratios: list[float]=DEFAULT_TRAIN_RATIOS,
    seed: int=42,
    max_dataset_size: int | None=None
):
    """
    [MAIN METHOD] Builds a GEMLeR [1] dataset from OpenML (GEMLeR is a specific
    collection of gene expression datasets).

    [1] Stiglic, G., & Kokol, P. (2010). Stability of Ranked Gene Lists in Large 
    icroarray Analysis Studies. Journal of biomedicine biotechnology, 2010, 616358.
    """

    mapping = get_gene_mapping()
    dataset = openml.datasets.get_dataset(openml_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = X.loc[:, X.columns.isin(mapping.keys())]
    X.rename(columns=mapping, inplace=True)
    X = X.T.groupby(X.columns).mean().T

    # get top 1000 most variable genes
    gene_variances = X.var(axis=0)
    top_1000_genes = gene_variances.sort_values(ascending=False).index[:1000]
    X = X[top_1000_genes]

    y = y.astype(str)

    return Dataset.from_Xy(
        X=X,
        y=y,
        save_dir=data_output_dir,
        display_name=display_name,
        problem_type=ModelTask.BINARY_CLASSIFICATION,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
        crop_dataset_to_size=max_dataset_size
    )
