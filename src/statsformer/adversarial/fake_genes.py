from copy import deepcopy
import os
import random

import requests
import xml.etree.ElementTree as Tree
import secrets
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from statsformer.data.dataset import Dataset


def replace_genenames(
    dataset: Dataset,
    save_dir: str=None,
    replace_ratio: float=0.5,
    replace_top: bool=False,
    seed: int=42,
):
    np.random.seed(seed)
    random.seed(seed)
    original_dataset = dataset
    dataset = deepcopy(dataset)
    
    num_replace = int(len(dataset.y) * replace_ratio)
    if replace_top:
        df = dataset.X
        # find the highest_variance feature indices
        variances = df.var(axis=0)
        sorted_variances = variances.sort_values(ascending=False)
        feature_names_to_replace = sorted_variances.index[:num_replace].tolist()
    else:
        feature_names_to_replace = np.random.choice(
            dataset.feature_names(),
            size=num_replace,
            replace=False
        ).tolist()
    fake_genes = get_fake_gene_names(
        n=num_replace
    )
    dataset.X.rename(
        {old: new for (old, new) in zip(feature_names_to_replace, fake_genes)},
        axis=1, inplace=True
    )

    if save_dir is None:
        original_dirname = Path(dataset.save_dir).name
        save_dir = Path(dataset.save_dir).parent / f"{original_dirname}_{num_replace}_fake_genes"
    dataset.save_dir = save_dir

    assert original_dataset.feature_names() != dataset.feature_names()

    dataset.save()
    return dataset


def get_fake_gene_names(n, min_len=4, max_len=6):
    """
    Produces `n` fake genenames (random alphanumeric strings that are not valid
    OMIM gene names).

    Parameters:
    - `n`: number of fake genenames to generate.
    - `min_len`: minimum genename length.
    - `max_len`: maximum genename length.
    """
    genes = set()
    while len(genes) < n:
        lengths = np.random.randint(min_len, max_len+1, size=n - len(genes))
        fake_data = ''.join(
            secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            for _ in range(sum(lengths))
        )
        
        start = 0
        for ell in lengths:
            gene = fake_data[start:start+ell]
            if get_mim_number(gene, quiet=True) is None:
                genes.add(gene)
            start += ell
    return list(genes)


def get_mim_number(gene, quiet=False): 
    """
    Query OMIM API to fetch the mimNumber for a given gene or phenotype.
    Args:
        gene (str): hgnc gene name.
    """
    load_dotenv()
    api_key = os.environ.get("OMIM_KEY", None)
    assert api_key is not None, "Need OMIM_KEY to be in .env"

    base_url = "https://api.omim.org/api/entry/search"
    search_query = f"{gene}" 
    params = {
        "start": 0,
        "sort": "score desc",
        "limit": 1,
        "apiKey": api_key,
        "format": "xml",  # Ensure the response is in XML format
    }

    # Manually append the 'search' query to the URL
    full_url = f"{base_url}?search={search_query}"

    try:
        # Send the HTTP GET request
        response = requests.get(full_url, params=params)
        response.raise_for_status()

        # Parse the XML response
        root = Tree.fromstring(response.text)

        # Locate the mimNumber in the response
        mim_number_element = root.find(".//mimNumber")
        if mim_number_element is not None:
            return mim_number_element.text
        else:
            if not quiet:
                print(f"No mimNumber found for gene/phenotype: {gene}")
            return None
    except requests.exceptions.RequestException as e:
        if not quiet:
            print(f"Error fetching mimNumber for gene/phenotype {gene}: {e}")
        return None