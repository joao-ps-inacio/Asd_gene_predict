"""Reading and writing embedding tables.

An embedding table is a Parquet file with one row per gene: an ``ensembl_gene_id`` column
followed by the vector as float32 columns ``f0 … fN``. Model and pooling metadata are stored in
the Parquet schema metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from asd_gene_predict.paths import PROCESSED

ID = "ensembl_gene_id"


def embedding_path(name: str) -> Path:
    """``protein_prott5`` → ``data/processed/emb_protein_prott5.parquet``."""
    return PROCESSED / f"emb_{name}.parquet"


def to_frame(ids, vectors: np.ndarray) -> pd.DataFrame:
    vectors = np.asarray(vectors, dtype=np.float32)
    cols = [f"f{i}" for i in range(vectors.shape[1])]
    df = pd.DataFrame(vectors, columns=cols)
    df.insert(0, ID, list(ids))
    return df


def save_embeddings(df: pd.DataFrame, path: Path, metadata: dict | None = None) -> Path:
    if df[ID].duplicated().any():
        raise ValueError("embeddings: genes duplicados")
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"asd"] = json.dumps(metadata or {}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path)
    return path


def load_embeddings(path: Path) -> pd.DataFrame:
    """Return a float32 matrix indexed by ``ensembl_gene_id``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Gerar primeiro os embeddings (`asd embed`).")
    df = pd.read_parquet(path).set_index(ID)
    return df.astype(np.float32)


def read_metadata(path: Path) -> dict:
    meta = pq.read_schema(path).metadata or {}
    return json.loads(meta.get(b"asd", b"{}"))
