"""Cross-validation with the thesis design and corrected metrics.

Design kept from the thesis: the folds are stratified over the ``cat_1`` genes; every
positive set is tested on the same ``cat_1`` test fold, and larger sets only add genes to
training. Hyperparameters are tuned by an inner ``GridSearchCV`` on the training part only.

Metric fixes: ROC-AUC and average precision are computed on continuous scores
(``predict_proba`` / ``decision_function``). The thesis computed them on 0/1 predictions;
those values are still reported as ``roc_auc_legacy`` and ``ap_legacy`` for comparison.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from asd_gene_predict.data.labels import PositiveSet, select_set
from asd_gene_predict.models.registry import build_pipeline, pipeline_grid

log = logging.getLogger(__name__)

ID = "ensembl_gene_id"
SCORE_METRICS = ["roc_auc", "average_precision", "precision_at_k"]
LABEL_METRICS = ["f1", "precision", "recall", "specificity", "mcc", "accuracy"]
LEGACY_METRICS = ["roc_auc_legacy", "ap_legacy"]
METRICS = SCORE_METRICS + LABEL_METRICS + LEGACY_METRICS


def continuous_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision among the ``k`` highest scores, with ties at the cut-off counted by their
    expected value (so the result does not depend on the order of the rows)."""
    if k <= 0:
        return np.nan
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    cut = np.sort(scores)[::-1][k - 1]
    above, tied = scores > cut, scores == cut
    hits = y_true[above].sum() + (k - above.sum()) * y_true[tied].mean()
    return float(hits / k)


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "precision_at_k": precision_at_k(y_true, scores, int(y_true.sum())),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc_legacy": roc_auc_score(y_true, y_pred),
        "ap_legacy": average_precision_score(y_true, y_pred),
    }


def make_folds(
    test_pool: pd.DataFrame, n_splits: int, n_repeats: int, seed: int
) -> list[tuple[int, int, set[str]]]:
    """Stratified folds over ``test_pool`` → ``[(repeat, fold, test_gene_ids)]``."""
    ids, y = test_pool[ID].to_numpy(), test_pool["label"].to_numpy()
    folds = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (_, test_idx) in enumerate(skf.split(ids, y)):
            folds.append((r, f, set(ids[test_idx])))
    return folds


def cross_validate(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    sets: Iterable[PositiveSet],
    models: Iterable[str],
    *,
    test_set: str = "cat_1",
    n_splits: int = 5,
    n_repeats: int = 1,
    inner_splits: int = 5,
    scoring: str = "average_precision",
    seed: int = 42,
    grids: dict | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Evaluate every (positive set × model) on shared ``test_set`` folds.

    ``X`` is the feature matrix indexed by Ensembl gene ID; genes without features are
    dropped. Returns one row per (set, model, repeat, fold) with all metrics.
    """
    sets = list(sets)
    by_name = {s.name: s for s in sets}
    if test_set not in by_name:
        raise ValueError(f"O conjunto de teste {test_set!r} tem de estar em {list(by_name)}")

    labels = labels[labels[ID].isin(X.index)]
    pool = select_set(labels, by_name[test_set])
    folds = make_folds(pool, n_splits, n_repeats, seed)
    log.info("%d genes no conjunto de teste %s; %d folds.", len(pool), test_set, len(folds))

    rows = []
    for pset in sets:
        data = select_set(labels, pset)
        for model_name in models:
            for repeat, fold, test_ids in folds:
                t0 = time.perf_counter()
                train = data[~data[ID].isin(test_ids)]
                test = pool[pool[ID].isin(test_ids)]
                X_tr, y_tr = X.loc[train[ID]].to_numpy(), train["label"].to_numpy()
                X_te, y_te = X.loc[test[ID]].to_numpy(), test["label"].to_numpy()

                search = GridSearchCV(
                    build_pipeline(model_name, seed, y_tr),
                    pipeline_grid(model_name, (grids or {}).get(model_name)),
                    scoring=scoring,
                    cv=StratifiedKFold(inner_splits, shuffle=True, random_state=seed),
                    n_jobs=n_jobs,
                    refit=True,
                )
                search.fit(X_tr, y_tr)
                best = search.best_estimator_
                metrics = compute_metrics(y_te, continuous_scores(best, X_te), best.predict(X_te))
                rows.append(
                    {
                        "set": pset.name,
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": len(y_tr),
                        "n_train_pos": int(y_tr.sum()),
                        "n_test": len(y_te),
                        "n_test_pos": int(y_te.sum()),
                        **metrics,
                        "best_params": json.dumps(
                            {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
                            default=str,
                        ),
                        "seconds": round(time.perf_counter() - t0, 2),
                    }
                )
                log.info(
                    "%s / %s / r%d f%d: AUPRC=%.3f AUC=%.3f",
                    pset.name,
                    model_name,
                    repeat,
                    fold,
                    metrics["average_precision"],
                    metrics["roc_auc"],
                )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Mean ± sd over folds, one row per (set, model)."""
    metrics = metrics or METRICS
    grouped = results.groupby(["set", "model"], sort=False)[metrics]
    mean, std = grouped.mean(), grouped.std(ddof=0)
    return mean.join(std, lsuffix="_mean", rsuffix="_sd")


def format_table(results: pd.DataFrame, metrics: list[str], digits: int = 3) -> pd.DataFrame:
    """``0.812 ± 0.021`` strings, for Markdown reports."""
    grouped = results.groupby(["set", "model"], sort=False)[metrics]
    mean, std = grouped.mean(), grouped.std(ddof=0)
    return mean.round(digits).astype(str) + " ± " + std.round(digits).astype(str)
