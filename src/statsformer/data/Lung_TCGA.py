import requests
import pandas as pd
import io
from tqdm import tqdm
from multiprocessing import Pool
import gc
from pydeseq2.dds import DeseqDataSet
import mygene
from pathlib import Path

from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS, Dataset
from statsformer.models.base import ModelTask


METADATA_CSV = "metadata.csv"


def build_lung_TCGA_dataset(
    checkpointing_dir: str | Path,
    data_output_dir: str | Path,
    num_threads: int=20,
    num_splits: int=10,
    train_ratios: list[float]=DEFAULT_TRAIN_RATIOS,
    seed: int=42,
):
    checkpointing_dir = Path(checkpointing_dir)
    checkpointing_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = Path(data_output_dir)

    vst_mat_filter = _get_vst_mat_filter(
        checkpointing_dir, num_threads
    )
    metadata = pd.read_csv(checkpointing_dir / METADATA_CSV)

    return Dataset.from_Xy(
        X=vst_mat_filter,
        y=metadata["condition"],
        save_dir=data_output_dir,
        display_name="LUAD vs. LUSC",
        problem_type=ModelTask.BINARY_CLASSIFICATION,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed
    )


def _get_vst_mat_filter(
    checkpointing_dir: Path,
    num_threads: int
):
    vst_mat_filename = checkpointing_dir / "vst_mat.csv"
    vst_mat_filter_filename = checkpointing_dir / "vst_mat_filter.csv"
    protein_coding_filename = checkpointing_dir / "protein_coding.csv"

    if vst_mat_filter_filename.exists():
        print(f"[INFO] Loading filtered VST matrix")
        vst_mat_filter = pd.read_csv(vst_mat_filter_filename)
    else:
        counts_filtered = _get_filtered_counts(checkpointing_dir, num_threads)
        print(f"[INFO] Building Deseq2")
        metadata = pd.read_csv(checkpointing_dir / METADATA_CSV)
        dds = DeseqDataSet(
            counts=counts_filtered,
            metadata=metadata,
            design="condition"
        )
        dds.deseq2()
        # variance stabilizing transformation
        print(f"[INFO] Building VST matrix")
        dds.vst()

        vst_mat = pd.DataFrame(dds.layers['vst_counts'], columns=counts_filtered.columns)

        print(f"[INFO] Getting gene info")
        mg = mygene.MyGeneInfo()
        query = list(vst_mat.columns.unique())
        gene_info = mg.querymany(query, scopes="ensembl.gene", 
                                fields=["symbol", "type_of_gene"], species="human")

        gene_info_df = pd.DataFrame(gene_info)[["query", "symbol", "type_of_gene"]]
        gene_info_df = gene_info_df.rename(columns={"query": "ensembl_gene_id"})

        protein_coding = gene_info_df[gene_info_df["type_of_gene"] == "protein-coding"]
        vst_mat = vst_mat[protein_coding["ensembl_gene_id"]]
        vst_mat.to_csv(vst_mat_filename, index=False)
        protein_coding.to_csv(protein_coding_filename, index=False)

        # Filter matrix
        print(f"[INFO] Filtering VST matrix")
        gene_variances = vst_mat.var(axis=0)
        top_1000_genes = gene_variances.sort_values(ascending=False).index[:1000]

        vst_mat_filter = vst_mat[top_1000_genes]

        # --- Map Ensembl IDs → gene names ---
        id_to_name = dict(zip(protein_coding["ensembl_gene_id"],
                            protein_coding["symbol"]))
        gene_names = [id_to_name.get(g, g) for g in top_1000_genes]
        vst_mat_filter.columns = gene_names
        vst_mat_filter.to_csv(vst_mat_filter_filename, index=False)

    return vst_mat_filter


def _get_filtered_counts(
    checkpointing_dir: Path,
    num_threads: int
):
    counts_filt_file = checkpointing_dir / "counts_filtered.csv"
    metadata_df_file = checkpointing_dir / METADATA_CSV

    if counts_filt_file.exists():
        print(f"[INFO] Loading filtered counts")
        counts_filtered = pd.read_csv(counts_filt_file)
    else:
        df_LUAD = _get_LUAD_df(checkpointing_dir, num_threads)
        df_LUSC = _get_LUSC_df(checkpointing_dir, num_threads)
        print(f"[INFO] Building filtered counts")
    
        counts_combined = pd.concat(
            (df_LUAD, df_LUSC), axis=0
        ).round().astype(int).T
        keep = counts_combined.sum(axis=1) >= 10
        counts_filtered = counts_combined.loc[keep]
        counts_filtered.index = counts_filtered.index.str.replace(r"\..*$", "", regex=True)
        counts_filtered = counts_filtered.groupby(counts_filtered.index).sum().T
        counts_filtered = counts_filtered.reset_index(drop=True)

        metadata = pd.DataFrame({
            "condition": ["LUAD"] * len(df_LUAD) + ["LUSC"] * len(df_LUSC)
        })
        metadata.to_csv(metadata_df_file, index=False)
        counts_filtered.to_csv(counts_filt_file, index=False)

        del counts_combined
        del df_LUAD
        del df_LUSC
        gc.collect()
    
    return counts_filtered


def _get_LUAD_df(
    checkpointing_dir: Path,
    num_threads: int
):
    df_LUAD_file = checkpointing_dir / "df_LUAD.csv"
    if df_LUAD_file.exists():
        print(f"[INFO] Loading LUAD Dataframe")
        df_LUAD = pd.read_csv(df_LUAD_file)
    else:
        print(f"[INFO] Collecting LUAD data")
        df_LUAD = _get_data("LUAD", num_threads)
        df_LUAD.to_csv(df_LUAD_file, index=False)
    return df_LUAD


def _get_LUSC_df(
    checkpointing_dir: Path,
    num_threads: int
):
    df_LUSC_file = checkpointing_dir / "df_LUSC.csv"
    if df_LUSC_file.exists():
        print(f"[INFO] Loading LUSC Dataframe")
        df_LUSC = pd.read_csv(df_LUSC_file)
    else:
        print(f"[INFO] Collecting LUSC data")
        df_LUSC = _get_data("LUSC", num_threads)
        df_LUSC.to_csv(df_LUSC_file, index=False)
    return df_LUSC


def _get_expression_json(file_id):
    data_endpt = f"https://api.gdc.cancer.gov/data/{file_id}"
    response = requests.get(data_endpt, headers = {"Content-Type": "application/json"})

    df = pd.read_csv(io.StringIO(response.content.decode()), skiprows=1, delimiter='\t')
    df = df.loc[4:].reset_index(drop=True)
    df = df[["gene_id", "unstranded"]].transpose()
    df = df.rename(columns=df.iloc[0])

    return {
        **df.loc["unstranded"].to_dict(),
        "file_id": file_id
    }


def _get_data(condition, n_threads=20):
    files_endpt = "https://api.gdc.cancer.gov/files"

    # This set of filters is nested under an 'and' operator.
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": f"TCGA-{condition}"}},
            {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
        ],
    }

    fields = [
        'id',
        'file_id',
        'data_format',
        'cases.submitter_id',
        "cases.samples.sample_type"
        
    ]
    fields = ",".join(fields)

    # A POST is used, so the filter parameters can be passed directly as a Dict object.
    params = {
        "fields": fields,
        "filters": filters,
        "format": "TSV",
        "size": 600
    }

    # The parameters are passed to 'json' rather than 'params' in this case
    response = requests.post(files_endpt, headers = {"Content-Type": "application/json"}, json = params)
    df = pd.read_csv(io.StringIO(response.content.decode()), delimiter='\t')

    # primary tumors only
    df = df[df['cases.0.samples.0.sample_type'] == 'Primary Tumor']
    df = df.rename(columns={'cases.0.submitter_id': 'patient_id'})
    df = df[["patient_id", "file_id"]]
    df = df.reset_index(drop=True)

    file_ids = df["file_id"]

    with Pool(n_threads) as p:
        gene_data = list(tqdm(p.imap(
            _get_expression_json, file_ids
        ), total=len(file_ids)))
    
    df2 = pd.DataFrame(gene_data)
    df_merged = df2.join(df.set_index("file_id"), on="file_id")

    return (
        df_merged
        .groupby("patient_id", as_index=True)
        .mean(numeric_only=True)
    )