"""Temporal validation: did a ranking made with an old SFARI release anticipate the genes that
SFARI added later?

The candidate pool is every ranked gene that was *not* labelled when the model was built
(not in the old SFARI release, not a training negative). Genes added to SFARI afterwards are
the positives. A useful ranking puts them near the top; a random one spreads them uniformly.

Baselines are scored on the same pool, so the comparison is fair:
- protein length: long genes accumulate more de novo variants and get discovered more often;
- LOEUF (gnomAD): genes intolerant to loss of function are classic disease-gene candidates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import average_precision_score, roc_auc_score

ID = "ensembl_gene_id"
TOPS = (0.01, 0.05, 0.10)


# --------------------------------------------------------------------------- inputs


def new_genes(old_sfari: pd.DataFrame, new_sfari: pd.DataFrame) -> pd.DataFrame:
    """Genes in ``new_sfari`` absent from ``old_sfari`` (by Ensembl ID and by symbol).

    Both frames use the layout of :func:`asd_gene_predict.data.labels.read_sfari`.
    """
    seen = new_sfari[ID].isin(old_sfari[ID]) | new_sfari["symbol"].isin(old_sfari["symbol"])
    return new_sfari[~seen].reset_index(drop=True)


def promoted_genes(old_sfari: pd.DataFrame, new_sfari: pd.DataFrame) -> pd.DataFrame:
    """Genes whose SFARI score improved (lower number = stronger evidence)."""
    both = old_sfari.merge(new_sfari, on=ID, suffixes=("_old", "_new"))
    better = both["sfari_score_new"].lt(both["sfari_score_old"]).fillna(False)
    return both[better].reset_index(drop=True)


def read_legacy_ranking(path: Path) -> pd.Series:
    """Thesis ranked list → probability of being an ASD gene, indexed by Ensembl gene ID."""
    df = pd.read_csv(path, usecols=["Ensembl_ID", "Probability_Class_1"])
    df = df.drop_duplicates("Ensembl_ID")
    return df.set_index("Ensembl_ID")["Probability_Class_1"].rename_axis(ID)


def protein_length(path: Path) -> pd.Series:
    """Longest protein per gene, from the thesis peptide table."""
    df = pd.read_csv(path, usecols=["Gene stable ID", "protein_seq"]).dropna()
    lengths = df["protein_seq"].str.len().groupby(df["Gene stable ID"]).max()
    return lengths.rename_axis(ID).rename("protein_length").astype(float)


def loeuf_score(path: Path) -> pd.Series:
    """gnomAD constraint → ``-LOEUF`` per gene (higher = more intolerant), canonical
    transcript only."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    loeuf_col = next(c for c in ("lof.oe_ci.upper", "oe_lof_upper") if c in df.columns)
    gene_col = next(c for c in ("gene_id", "gene_ids") if c in df.columns)
    for flag in ("mane_select", "canonical"):
        if flag in df.columns:
            df = df[df[flag].astype(str).str.lower() == "true"]
            break
    df = df[df[gene_col].astype(str).str.startswith("ENSG")].dropna(subset=[loeuf_col])
    out = -df.groupby(gene_col)[loeuf_col].min()
    return out.rename_axis(ID).rename("neg_loeuf")


# --------------------------------------------------------------------------- evaluation


def evaluate_ranking(scores: pd.Series, positives: Iterable[str], tops=TOPS) -> dict:
    """How well ``scores`` (higher = more likely ASD gene) put ``positives`` at the top."""
    scores = scores.dropna()
    y = scores.index.isin(set(positives)).astype(int)
    n, k = len(scores), int(y.sum())
    if k == 0 or k == n:
        raise ValueError("O pool tem de ter genes novos e genes que não o são.")
    pct = scores.rank(pct=True, method="average").to_numpy()  # 1.0 = top
    out = {
        "n_pool": n,
        "n_new": k,
        "auc": roc_auc_score(y, scores),
        "auprc": average_precision_score(y, scores),
        "auprc_random": k / n,
        "median_percentile": float(np.median(pct[y == 1])),
    }
    for top in tops:
        in_top = pct > 1 - top
        hits, m = int(y[in_top].sum()), int(in_top.sum())
        tag = f"top{round(top * 100)}"
        out[f"{tag}_hits"] = hits
        out[f"{tag}_frac"] = hits / k
        out[f"{tag}_enrichment"] = (hits / k) / (m / n)
        out[f"{tag}_p"] = float(hypergeom.sf(hits - 1, n, k, m))
    return out


def stratified_auc(
    scores: pd.Series, confounder: pd.Series, positives: Iterable[str], bins: int = 5
) -> float:
    """AUC computed inside quantile bins of ``confounder`` (e.g. protein length) and averaged,
    weighted by the number of positives per bin. It answers: among genes of similar length,
    does the score still rank the new genes higher?"""
    df = pd.DataFrame({"s": scores, "c": confounder}).dropna()
    df["y"] = df.index.isin(set(positives)).astype(int)
    df["bin"] = pd.qcut(df["c"].rank(method="first"), bins, labels=False)
    aucs, weights = [], []
    for _, g in df.groupby("bin"):
        if 0 < g["y"].sum() < len(g):
            aucs.append(roc_auc_score(g["y"], g["s"]))
            weights.append(g["y"].sum())
    return float(np.average(aucs, weights=weights))


def bootstrap_auc_diff(
    a: pd.Series, b: pd.Series, positives: Iterable[str], n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """AUC(a) - AUC(b) on the same genes, with a 95% bootstrap interval (stratified by class)."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    y = df.index.isin(set(positives)).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    va, vb = df["a"].to_numpy(), df["b"].to_numpy()
    diffs = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        diffs.append(roc_auc_score(y[idx], va[idx]) - roc_auc_score(y[idx], vb[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(roc_auc_score(y, va) - roc_auc_score(y, vb)), float(lo), float(hi)


def compare(
    scores: Mapping[str, pd.Series],
    positives: Iterable[str],
    exclude: Iterable[str],
    shared_pool: bool = True,
    confounder: pd.Series | None = None,
) -> pd.DataFrame:
    """Evaluate several scores. With ``shared_pool`` every score is evaluated on the genes
    that all of them cover, so differences cannot come from different gene sets. With a
    ``confounder``, also report the AUC within bins of it (``auc_adjusted``)."""
    exclude, positives = set(exclude), set(positives)
    cleaned = {name: s.dropna()[~s.dropna().index.isin(exclude)] for name, s in scores.items()}
    if shared_pool:
        common = set.intersection(*(set(s.index) for s in cleaned.values()))
        cleaned = {name: s[s.index.isin(common)] for name, s in cleaned.items()}
    rows = {name: evaluate_ranking(s, positives) for name, s in cleaned.items()}
    if confounder is not None:
        for name, s in cleaned.items():
            rows[name]["auc_adjusted"] = stratified_auc(s, confounder, positives)
    return pd.DataFrame(rows).T.rename_axis("score")


def top_hits(scores: pd.Series, positives: Iterable[str], exclude: Iterable[str], n: int = 50):
    """New genes among the ``n`` best-ranked unlabelled genes."""
    pool = scores.dropna()[~scores.dropna().index.isin(set(exclude))]
    top = pool.sort_values(ascending=False).head(n)
    return [g for g in top.index if g in set(positives)]
