"""Gene identifier map shared by every pipeline stage.

One row per Ensembl gene, joining:

- **HGNC** (backbone): approved symbol, HGNC ID, Entrez ID, locus group;
- **MANE Select**: the canonical transcript/protein (used for sequence embeddings);
- **STRING v12**: the protein node ID in the PPI graph (used for graph embeddings).

Keyed by unversioned Ensembl gene ID (``ENSG00000000003``). Transcript and protein IDs keep
their version, because the version determines the exact sequence.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from asd_gene_predict.paths import PROCESSED

log = logging.getLogger(__name__)

GENE_MAP_PATH = PROCESSED / "gene_map.parquet"

COLUMNS = [
    "ensembl_gene_id",
    "symbol",
    "hgnc_id",
    "entrez_id",
    "locus_group",
    "mane_status",
    "mane_ensembl_transcript",
    "mane_ensembl_protein",
    "mane_refseq_transcript",
    "mane_refseq_protein",
    "string_id",
]
ID_COLUMNS = ("ensembl_gene_id", "symbol", "hgnc_id", "entrez_id", "string_id")

_ENSG = re.compile(r"^ENSG\d{11}$")


def strip_version(ids: pd.Series) -> pd.Series:
    """``ENSG00000121410.12`` → ``ENSG00000121410``."""
    return ids.str.split(".", n=1).str[0]


# --------------------------------------------------------------------------- readers


def read_hgnc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if "status" in df:
        df = df[df["status"] == "Approved"]
    df = df.dropna(subset=["ensembl_gene_id"])
    out = pd.DataFrame(
        {
            "ensembl_gene_id": strip_version(df["ensembl_gene_id"].str.strip()),
            "symbol": df["symbol"],
            "hgnc_id": df["hgnc_id"],
            "entrez_id": pd.to_numeric(df["entrez_id"], errors="coerce").astype("Int64"),
            "locus_group": df["locus_group"],
        }
    )
    dup = out["ensembl_gene_id"].duplicated(keep="first")
    if dup.any():
        log.warning(
            "HGNC: %d IDs Ensembl duplicados, mantida a primeira linha: %s",
            dup.sum(),
            out.loc[dup, "ensembl_gene_id"].tolist()[:10],
        )
        out = out[~dup]
    return out.reset_index(drop=True)


def read_mane(path: Path) -> pd.DataFrame:
    """MANE summary → one row per gene, preferring ``MANE Select`` over ``MANE Plus Clinical``."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.dropna(subset=["Ensembl_Gene"])
    df = df.assign(
        ensembl_gene_id=strip_version(df["Ensembl_Gene"]),
        _is_select=df["MANE_status"].eq("MANE Select"),
    ).sort_values(["ensembl_gene_id", "_is_select"], ascending=[True, False])
    df = df.drop_duplicates("ensembl_gene_id", keep="first")
    return df.rename(
        columns={
            "MANE_status": "mane_status",
            "Ensembl_nuc": "mane_ensembl_transcript",
            "Ensembl_prot": "mane_ensembl_protein",
            "RefSeq_nuc": "mane_refseq_transcript",
            "RefSeq_prot": "mane_refseq_protein",
        }
    )[
        [
            "ensembl_gene_id",
            "mane_status",
            "mane_ensembl_transcript",
            "mane_ensembl_protein",
            "mane_refseq_transcript",
            "mane_refseq_protein",
        ]
    ].reset_index(drop=True)


def read_string_aliases(path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """STRING aliases → ``ensembl_gene_id | string_id``.

    Keeps aliases that are Ensembl gene IDs. When a gene maps to several STRING proteins, the
    one supported by the most alias sources wins (ties broken alphabetically), so the choice is
    deterministic.
    """
    parts = []
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        names=["string_id", "alias", "source"],
        comment=None,
        header=0,
    )
    for chunk in reader:
        alias = strip_version(chunk["alias"].str.strip())
        keep = alias.str.match(_ENSG.pattern, na=False)
        parts.append(
            pd.DataFrame(
                {"ensembl_gene_id": alias[keep], "string_id": chunk.loc[keep, "string_id"]}
            )
        )
    pairs = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["ensembl_gene_id", "string_id"])
    )
    counts = (
        pairs.groupby(["ensembl_gene_id", "string_id"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["ensembl_gene_id", "n", "string_id"], ascending=[True, False, True])
    )
    multi = counts["ensembl_gene_id"].duplicated().sum()
    if multi:
        log.info("STRING: %d genes com mais de uma proteína; escolhida a mais suportada.", multi)
    return counts.drop_duplicates("ensembl_gene_id")[["ensembl_gene_id", "string_id"]].reset_index(
        drop=True
    )


# --------------------------------------------------------------------------- build / validate


def build_gene_map(hgnc: Path, mane: Path, string_aliases: Path) -> pd.DataFrame:
    gmap = (
        read_hgnc(hgnc)
        .merge(read_mane(mane), on="ensembl_gene_id", how="left")
        .merge(read_string_aliases(string_aliases), on="ensembl_gene_id", how="left")
    )
    # a STRING protein must map to a single gene; drop ambiguous links rather than guess
    shared = gmap["string_id"].notna() & gmap["string_id"].duplicated(keep=False)
    if shared.any():
        log.warning("STRING: %d genes partilham proteína STRING; ligação removida.", shared.sum())
        gmap.loc[shared, "string_id"] = pd.NA
    gmap = gmap[COLUMNS].sort_values("ensembl_gene_id").reset_index(drop=True)
    validate_gene_map(gmap)
    return gmap


def validate_gene_map(gmap: pd.DataFrame) -> None:
    missing = set(COLUMNS) - set(gmap.columns)
    if missing:
        raise ValueError(f"gene_map sem colunas: {sorted(missing)}")
    if not gmap["ensembl_gene_id"].str.match(_ENSG.pattern).all():
        bad = gmap.loc[~gmap["ensembl_gene_id"].str.match(_ENSG.pattern), "ensembl_gene_id"]
        raise ValueError(f"IDs Ensembl inválidos: {bad.tolist()[:10]}")
    for col in ("ensembl_gene_id", "symbol", "hgnc_id", "string_id"):
        dup = gmap[col].dropna().duplicated()
        if dup.any():
            raise ValueError(
                f"gene_map: valores duplicados em {col}: {gmap[col].dropna()[dup].tolist()[:10]}"
            )


def summarize(gmap: pd.DataFrame) -> dict[str, int]:
    return {
        "genes": len(gmap),
        "protein_coding": int(gmap["locus_group"].eq("protein-coding gene").sum()),
        "with_entrez": int(gmap["entrez_id"].notna().sum()),
        "with_mane": int(gmap["mane_ensembl_protein"].notna().sum()),
        "with_string": int(gmap["string_id"].notna().sum()),
    }


def save_gene_map(gmap: pd.DataFrame, path: Path = GENE_MAP_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gmap.to_parquet(path, index=False)
    return path


def load_gene_map(path: Path = GENE_MAP_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Correr primeiro `asd gene-map`.")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- lookups


def map_ids(ids: Iterable, source: str, target: str, gmap: pd.DataFrame | None = None) -> pd.Series:
    """Translate ``ids`` from one identifier column to another.

    Returns a Series aligned with ``ids`` (index = input IDs); unmapped IDs are ``<NA>``.
    Example: ``map_ids([2, 9], "entrez_id", "ensembl_gene_id")``.
    """
    if source not in ID_COLUMNS or target not in ID_COLUMNS:
        raise ValueError(f"source/target têm de estar em {ID_COLUMNS}")
    gmap = load_gene_map() if gmap is None else gmap
    lookup = gmap.dropna(subset=[source]).set_index(source)[target]
    ids = pd.Index(list(ids))
    if source == "entrez_id":
        ids = pd.Index(pd.to_numeric(ids, errors="coerce")).astype("Int64")
    return pd.Series(lookup.reindex(ids).to_numpy(), index=ids, name=target)
