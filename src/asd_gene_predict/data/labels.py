"""Positive (SFARI) and negative (Krishnan et al. 2016) gene labels.

Reproduces the thesis rules:

- positives are SFARI genes; a gene in both lists is a positive (SFARI wins);
- negatives are the Krishnan genes that are not in SFARI;
- the training sets (``cat_1``, ``cat_1_sd``, ...) are all negatives plus the SFARI genes with
  ``score <= max_score`` and, when ``with_unscored``, the SFARI genes without a score.

Output schema (``labels.parquet``), one row per gene:
``ensembl_gene_id | symbol | label | sfari_score | syndromic | source``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from asd_gene_predict.paths import CONFIGS, PROCESSED

log = logging.getLogger(__name__)

LABELS_PATH = PROCESSED / "labels.parquet"
EXCLUSIONS_FILE = CONFIGS / "exclusions.yaml"
COLUMNS = ["ensembl_gene_id", "symbol", "label", "sfari_score", "syndromic", "source"]


@dataclass(frozen=True)
class PositiveSet:
    name: str
    max_score: int
    with_unscored: bool = False


def read_sfari(path: Path, gene_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """SFARI export → ``ensembl_gene_id | symbol | sfari_score | syndromic``.

    Rows without an Ensembl ID are dropped, as in the thesis, unless ``gene_map`` is given,
    in which case they are recovered by HGNC symbol.
    """
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "ensembl_gene_id": df["ensembl-id"].str.strip(),
            "symbol": df["gene-symbol"].str.strip(),
            "sfari_score": pd.to_numeric(df["gene-score"], errors="coerce").astype("Int64"),
            "syndromic": df["syndromic"].fillna(0).astype(bool),
        }
    )
    missing = out["ensembl_gene_id"].isna()
    if missing.any() and gene_map is not None:
        by_symbol = gene_map.dropna(subset=["symbol"]).set_index("symbol")["ensembl_gene_id"]
        out.loc[missing, "ensembl_gene_id"] = out.loc[missing, "symbol"].map(by_symbol)
        recovered = out.loc[missing, "ensembl_gene_id"].notna().sum()
        log.info(
            "SFARI: %d de %d genes sem Ensembl recuperados pelo símbolo.", recovered, missing.sum()
        )
        missing = out["ensembl_gene_id"].isna()
    if missing.any():
        log.warning(
            "SFARI: %d genes sem ID Ensembl ignorados: %s",
            missing.sum(),
            out.loc[missing, "symbol"].tolist(),
        )
    out = out[~missing]
    dup = out["ensembl_gene_id"].duplicated()
    if dup.any():
        log.warning("SFARI: %d IDs Ensembl duplicados; mantida a primeira linha.", dup.sum())
        out = out[~dup]
    return out.reset_index(drop=True)


def read_krishnan(path: Path) -> pd.Series:
    """Krishnan negatives → Series of Ensembl gene IDs (blank rows dropped)."""
    ids = pd.read_csv(path).iloc[:, 0].dropna().astype(str).str.strip()
    ids = ids[ids != ""].drop_duplicates()
    return ids.reset_index(drop=True).rename("ensembl_gene_id")


def load_exclusions(path: Path = EXCLUSIONS_FILE) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        entries = (yaml.safe_load(f) or {}).get("exclusions") or []
    return {e["ensembl_gene_id"]: e.get("reason", "") for e in entries}


def build_labels(
    sfari: pd.DataFrame, negatives: pd.Series, exclusions: dict[str, str] | None = None
) -> pd.DataFrame:
    pos = sfari.assign(label=1, source="sfari")
    overlap = negatives.isin(pos["ensembl_gene_id"])
    if overlap.any():
        log.info("%d genes estão no SFARI e no Krishnan; ficam como positivos.", overlap.sum())
    neg = pd.DataFrame(
        {
            "ensembl_gene_id": negatives[~overlap],
            "symbol": pd.NA,
            "sfari_score": pd.array([pd.NA] * int((~overlap).sum()), dtype="Int64"),
            "syndromic": False,
            "label": 0,
            "source": "krishnan_2016",
        }
    )
    labels = pd.concat([pos, neg], ignore_index=True)[COLUMNS]

    excluded = labels["ensembl_gene_id"].isin((exclusions or {}).keys())
    if excluded.any():
        log.info("Excluídos %d genes (configs/exclusions.yaml).", excluded.sum())
        labels = labels[~excluded]

    labels = labels.astype({"label": "int8", "symbol": "string"})
    labels = labels.sort_values(["label", "ensembl_gene_id"], ascending=[False, True])
    labels = labels.reset_index(drop=True)
    validate_labels(labels)
    return labels


def validate_labels(labels: pd.DataFrame) -> None:
    if list(labels.columns) != COLUMNS:
        raise ValueError(f"labels: colunas {list(labels.columns)} != {COLUMNS}")
    if labels["ensembl_gene_id"].duplicated().any():
        raise ValueError("labels: genes duplicados (um gene não pode ser positivo e negativo)")
    if not labels["label"].isin([0, 1]).all():
        raise ValueError("labels: valores de label fora de {0, 1}")
    if labels.loc[labels["label"] == 0, "sfari_score"].notna().any():
        raise ValueError("labels: negativos com score SFARI")


def positive_sets(config: dict) -> list[PositiveSet]:
    return [PositiveSet(name, **spec) for name, spec in config["labels"]["positive_sets"].items()]


def select_set(labels: pd.DataFrame, pset: PositiveSet) -> pd.DataFrame:
    """All negatives plus the positives that belong to ``pset``."""
    score = labels["sfari_score"]
    in_set = score.le(pset.max_score).fillna(False)
    if pset.with_unscored:
        in_set |= score.isna() & labels["label"].eq(1)
    keep = labels["label"].eq(0) | in_set
    return labels[keep].reset_index(drop=True)


def summarize(labels: pd.DataFrame, sets: list[PositiveSet]) -> pd.DataFrame:
    rows = []
    for pset in sets:
        sub = select_set(labels, pset)
        rows.append(
            {
                "set": pset.name,
                "positives": int(sub["label"].sum()),
                "negatives": int((sub["label"] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def save_labels(labels: pd.DataFrame, path: Path = LABELS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, index=False)
    return path


def load_labels(path: Path = LABELS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Correr primeiro `asd labels`.")
    return pd.read_parquet(path)
