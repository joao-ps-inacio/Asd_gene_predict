"""Fair comparison of feature sources (protein, graph, both).

Fairness rules:

- every source is evaluated on the **same genes** (the intersection of the embeddings);
- the folds depend only on the genes and the seed, so every source sees the **same folds**;
- the same models, grids and inner tuning are used for every source.

Differences between two sources are tested per fold, with the corrected resampled t-test of
Nadeau & Bengio (2003). Folds from repeated cross-validation share training data, so a plain
t-test over folds is over-confident; the correction inflates the variance by the train/test
overlap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from asd_gene_predict.data.labels import PositiveSet
from asd_gene_predict.models.evaluate import cross_validate

KEY = ["model", "repeat", "fold"]


def feature_sets(
    embeddings: Mapping[str, pd.DataFrame], fusion: bool = True
) -> dict[str, pd.DataFrame]:
    """Restrict every embedding to the shared genes; optionally add their concatenation."""
    common = sorted(set.intersection(*(set(df.index) for df in embeddings.values())))
    out = {name: df.loc[common] for name, df in embeddings.items()}
    if fusion and len(out) > 1:
        out["+".join(out)] = pd.concat(
            [df.add_prefix(f"{name}:") for name, df in out.items()], axis=1
        )
    return out


def compare_sources(
    sets: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    positive_set: PositiveSet,
    models: Iterable[str],
    **cv_kwargs,
) -> pd.DataFrame:
    """Cross-validate every feature set with identical settings; one row per fold."""
    runs = []
    for name, X in sets.items():
        res = cross_validate(X, labels, [positive_set], list(models), **cv_kwargs)
        runs.append(res.assign(features=name))
    return pd.concat(runs, ignore_index=True)


def corrected_ttest(diffs: np.ndarray, test_train_ratio: float) -> dict:
    """Nadeau-Bengio corrected resampled t-test on per-fold differences."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean, var = diffs.mean(), diffs.var(ddof=1)
    se = np.sqrt((1 / n + test_train_ratio) * var)
    if se == 0:
        return {"diff": mean, "ci_low": mean, "ci_high": mean, "p": 0.0 if mean else 1.0, "n": n}
    t = mean / se
    half = stats.t.ppf(0.975, n - 1) * se
    return {
        "diff": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "p": float(2 * stats.t.sf(abs(t), n - 1)),
        "n": n,
    }


def paired_comparison(
    results: pd.DataFrame, a: str, b: str, metrics: Iterable[str] = ("roc_auc", "average_precision")
) -> pd.DataFrame:
    """``a - b`` per model and metric, paired on (model, repeat, fold)."""
    left = results[results.features == a].set_index(KEY)
    right = results[results.features == b].set_index(KEY)
    joined = left.join(right, lsuffix="_a", rsuffix="_b", how="inner")
    ratio = float((joined["n_test_a"] / (joined["n_train_a"])).mean())
    rows = []
    for model, g in joined.groupby(level="model"):
        for m in metrics:
            res = corrected_ttest((g[f"{m}_a"] - g[f"{m}_b"]).to_numpy(), ratio)
            rows.append({"model": model, "metric": m, "a": a, "b": b, **res})
    return pd.DataFrame(rows)


def cross_category(
    sets: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    model: str = "lr",
    train_score: int = 1,
    test_scores: tuple[int, ...] = (2, 3),
    n_splits: int = 5,
    n_repeats: int = 5,
    inner_splits: int = 5,
    scoring: str = "average_precision",
    seed: int = 42,
) -> pd.DataFrame:
    """Train on SFARI category ``train_score`` genes and test on categories ``test_scores``.

    Negatives are split into folds; positives of the test categories are never used for
    training. Every feature set sees the same splits. One row per (features, repeat, fold).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

    from asd_gene_predict.models.evaluate import continuous_scores
    from asd_gene_predict.models.registry import build_pipeline, pipeline_grid

    genes = set.intersection(*(set(df.index) for df in sets.values()))
    lab = labels[labels.ensembl_gene_id.isin(genes)].set_index("ensembl_gene_id")
    pos_train = lab.index[(lab.label == 1) & (lab.sfari_score == train_score)]
    pos_test = lab.index[(lab.label == 1) & lab.sfari_score.isin(test_scores)]
    neg = lab.index[lab.label == 0]
    rows = []
    for r in range(n_repeats):
        for f, (tr, te) in enumerate(
            KFold(n_splits, shuffle=True, random_state=seed + r).split(neg)
        ):
            train, test = list(pos_train) + list(neg[tr]), list(pos_test) + list(neg[te])
            y_tr = np.r_[np.ones(len(pos_train)), np.zeros(len(tr))]
            y_te = np.r_[np.ones(len(pos_test)), np.zeros(len(te))]
            for name, X in sets.items():
                search = GridSearchCV(
                    build_pipeline(model, seed, y_tr),
                    pipeline_grid(model),
                    scoring=scoring,
                    cv=StratifiedKFold(inner_splits, shuffle=True, random_state=seed),
                    n_jobs=-1,
                ).fit(X.loc[train].to_numpy(), y_tr)
                s = continuous_scores(search.best_estimator_, X.loc[test].to_numpy())
                rows.append(
                    {
                        "features": name,
                        "model": model,
                        "repeat": r,
                        "fold": f,
                        "roc_auc": roc_auc_score(y_te, s),
                        "average_precision": average_precision_score(y_te, s),
                        "n_train": len(train),
                        "n_test": len(te),
                    }
                )
    return pd.DataFrame(rows)
