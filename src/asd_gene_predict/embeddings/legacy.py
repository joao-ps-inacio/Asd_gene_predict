"""Import of the embeddings computed during the thesis.

Two layouts exist in the thesis repository:

- sequence embeddings (ProtT5): each vector stored as a string inside a CSV cell
  (``"[0.1, -0.2, ...]"``), with columns ``0`` symbol, ``1`` Ensembl gene, ``2`` Ensembl
  protein, ``3`` vector, ``4`` label;
- graph embeddings (DeepWalk): one numeric column per dimension (``0`` … ``499``) plus
  ``ensb_gene_id``.

This module parses them once into the Parquet layout of :mod:`embeddings.io`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asd_gene_predict.embeddings.io import to_frame

LEGACY_META = {
    "prott5": {
        "model": "Rostlab/prot_t5_xl_half_uniref50-enc",
        "pooling": "mean",
        "sequence": "Ensembl canonical protein",
        "origin": "a59490/Tese_ASD_Gene_Pred@c3e7610 (03_ML/sequence/Protein)",
    },
    "deepwalk": {
        "model": "DeepWalk (GRAPE) on the STRING protein-interaction network",
        "dimensions": 500,
        "origin": "a59490/Tese_ASD_Gene_Pred@c3e7610 (04_Validation/01_Ranked_list/graph)",
    },
}


def parse_vector(text: str) -> np.ndarray:
    return np.fromstring(text.strip().strip("[]"), sep=",", dtype=np.float32)


def read_legacy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["1", "3"], dtype=str)
    vectors = np.stack(df["3"].map(parse_vector).to_numpy())
    if np.isnan(vectors).any():
        raise ValueError(f"{path}: vetores com NaN")
    frame = to_frame(df["1"].str.strip(), vectors)
    return frame.drop_duplicates("ensembl_gene_id").reset_index(drop=True)


def read_legacy_graph(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    dims = sorted((c for c in df.columns if c.isdigit()), key=int)
    if not dims:
        raise ValueError(f"{path}: sem colunas de dimensões numéricas")
    vectors = df[dims].to_numpy(dtype=np.float32)
    if np.isnan(vectors).any():
        raise ValueError(f"{path}: vetores com NaN")
    frame = to_frame(df["ensb_gene_id"].astype(str).str.strip(), vectors)
    return frame.drop_duplicates("ensembl_gene_id").reset_index(drop=True)
