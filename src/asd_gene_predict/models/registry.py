"""Classifiers and their hyperparameter grids.

Every model is a ``Pipeline(StandardScaler → classifier)`` so scaling is fitted inside each
training fold. Compared with the thesis:

- XGBoost uses ``scale_pos_weight`` (``class_weight`` was silently ignored);
- the SVM grid tests ``degree`` only for the ``poly`` kernel and has no duplicated kernels;
- grids are smaller and centred on the values the thesis selected, so a full run is hours,
  not days;
- every stochastic model gets a fixed ``random_state``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[int, np.ndarray], BaseEstimator]  # (seed, y_train) -> classifier
    grid: dict | list[dict] = field(default_factory=dict)


def _pos_weight(y: np.ndarray) -> float:
    pos = int(np.sum(y == 1))
    return float(np.sum(y == 0)) / pos if pos else 1.0


MODELS: dict[str, ModelSpec] = {
    "lr": ModelSpec(
        "lr",
        lambda seed, y: LogisticRegression(
            class_weight="balanced", max_iter=5000, random_state=seed
        ),
        {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    ),
    "svm": ModelSpec(
        "svm",
        lambda seed, y: SVC(class_weight="balanced", random_state=seed),
        [
            {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.0001]},
            {"kernel": ["linear"], "C": [0.001, 0.01, 0.1, 1]},
            {"kernel": ["poly"], "C": [0.1, 1, 10], "degree": [2, 3]},
        ],
    ),
    "rf": ModelSpec(
        "rf",
        lambda seed, y: RandomForestClassifier(
            n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        {
            "max_features": ["sqrt", "log2"],
            "max_depth": [None, 10, 30],
            "min_samples_leaf": [1, 2, 4],
        },
    ),
    "knn": ModelSpec(
        "knn",
        lambda seed, y: KNeighborsClassifier(n_jobs=-1),
        {
            "n_neighbors": [5, 11, 21, 31],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "cosine"],
        },
    ),
    "lgbm": ModelSpec(
        "lgbm",
        lambda seed, y: LGBMClassifier(
            class_weight="balanced", n_jobs=-1, random_state=seed, verbose=-1
        ),
        {
            "n_estimators": [200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31],
            "reg_alpha": [0, 1],
        },
    ),
    "xgb": ModelSpec(
        "xgb",
        lambda seed, y: XGBClassifier(
            scale_pos_weight=_pos_weight(y),
            n_jobs=-1,
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
        ),
        {"n_estimators": [200, 500], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]},
    ),
    "nb": ModelSpec("nb", lambda seed, y: GaussianNB(), {}),
}


def build_pipeline(name: str, seed: int, y_train: np.ndarray) -> Pipeline:
    spec = MODELS[name]
    return Pipeline([("scale", StandardScaler()), ("clf", spec.build(seed, y_train))])


def pipeline_grid(name: str, override: dict | list[dict] | None = None) -> list[dict]:
    """Grid of ``MODELS[name]`` (or ``override``), with keys prefixed for the pipeline."""
    grid = MODELS[name].grid if override is None else override
    grids = grid if isinstance(grid, list) else [grid]
    return [{f"clf__{k}": v for k, v in g.items()} for g in grids]


def resolve(names: str | list[str]) -> list[str]:
    names = [names] if isinstance(names, str) else list(names)
    if names == ["all"]:
        return list(MODELS)
    unknown = sorted(set(names) - MODELS.keys())
    if unknown:
        raise KeyError(f"Modelos desconhecidos: {unknown}. Disponíveis: {sorted(MODELS)}")
    return names
