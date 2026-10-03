"""Figures shown in the README, with one size and one palette so they line up in a grid.

Needs the outputs of ``asd labels``, ``asd embed protein --legacy``, ``asd train`` and
``asd validate-temporal``.

    python scripts/readme_figures.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from asd_gene_predict.config import load_config  # noqa: E402
from asd_gene_predict.data.labels import PositiveSet, load_labels, select_set  # noqa: E402
from asd_gene_predict.embeddings.io import embedding_path, load_embeddings  # noqa: E402
from asd_gene_predict.paths import REPORTS  # noqa: E402

OUT = REPORTS / "figures"
RESULTS = REPORTS / "results"
SIZE = (6.4, 4.0)

# reference palette (dataviz skill): categorical slots 1-2, neutral grey for baselines
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY, OLD = "#2a78d6", "#eb6834", "#a3a29c", "#86b6ef"
MODEL_NAMES = {"lr": "Logistic Regression", "svm": "SVM", "knn": "KNN", "nb": "Naive Bayes"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    }
)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def roc_pr() -> None:
    """Out-of-fold ROC and PR curves of the best model, against the thesis single point."""
    seed = load_config()["seed"]
    X = load_embeddings(embedding_path("protein_prott5"))
    labels = load_labels()
    data = select_set(labels[labels.ensembl_gene_id.isin(X.index)], PositiveSet("cat_1", 1))
    Xa, y = X.loc[data.ensembl_gene_id].to_numpy(), data.label.to_numpy()
    runs = pd.read_parquet(RESULTS / "protein_prott5__lr.parquet")
    best = runs[runs.set == "cat_1"].best_params.map(lambda p: json.loads(p)["C"])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(best.mode()[0]), class_weight="balanced", max_iter=5000),
    )
    proba = cross_val_predict(
        model, Xa, y, method="predict_proba", cv=StratifiedKFold(5, shuffle=True, random_state=seed)
    )[:, 1]
    pred = (proba >= 0.5).astype(int)
    sens, spec = pred[y == 1].mean(), 1 - pred[y == 0].mean()
    prec = y[pred == 1].mean()

    fig, (a, b) = plt.subplots(1, 2, figsize=SIZE)
    fpr, tpr, _ = roc_curve(y, proba)
    a.fill_between(fpr, tpr, color=BLUE, alpha=0.1, lw=0)
    a.plot(fpr, tpr, color=BLUE, lw=2, label=f"Probabilities: {roc_auc_score(y, proba):.2f}")
    a.plot(
        [0, 1 - spec, 1],
        [0, sens, 1],
        color=OLD,
        lw=2,
        ls="--",
        label=f"0/1 point (thesis): {roc_auc_score(y, pred):.2f}",
    )
    a.scatter([1 - spec], [sens], s=40, color=OLD, edgecolor=SURFACE, lw=2, zorder=3)
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    a.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (AUC)")
    a.legend(loc="lower right")

    pre, rec, _ = precision_recall_curve(y, proba)
    b.plot(
        rec, pre, color=BLUE, lw=2, label=f"Probabilities: {average_precision_score(y, proba):.2f}"
    )
    b.scatter(
        [sens],
        [prec],
        s=40,
        color=OLD,
        edgecolor=SURFACE,
        lw=2,
        zorder=3,
        label="0/1 point (thesis)",
    )
    b.axhline(y.mean(), color=MUTED, lw=0.8, ls=":", label=f"Random: {y.mean():.2f}")
    b.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (AUPRC)", ylim=(0, 1.02))
    b.legend(loc="lower left")
    for ax in (a, b):
        ax.grid(True)
        ax.set_axisbelow(True)
    save(fig, "readme_roc_pr.png")


def models() -> None:
    """Thesis vs corrected AUC for each model."""
    summ = pd.read_csv(RESULTS / "protein_prott5_summary.csv")
    summ = summ[(summ.set == "cat_1") & summ.model.isin(MODEL_NAMES)].sort_values("roc_auc_mean")
    fig, ax = plt.subplots(figsize=SIZE)
    y = np.arange(len(summ))
    ax.hlines(y, summ.thesis_roc_auc_mean, summ.roc_auc_mean, color=GRID, lw=3)
    ax.errorbar(
        summ.roc_auc_mean, y, xerr=summ.roc_auc_sd, fmt="none", ecolor=BLUE, lw=1, capsize=3
    )
    ax.scatter(
        summ.thesis_roc_auc_mean,
        y,
        s=60,
        color=OLD,
        edgecolor=SURFACE,
        lw=2,
        zorder=3,
        label="Thesis (AUC on 0/1 predictions)",
    )
    ax.scatter(
        summ.roc_auc_mean,
        y,
        s=60,
        color=BLUE,
        edgecolor=SURFACE,
        lw=2,
        zorder=3,
        label="This repo (AUC on probabilities, ±1 sd)",
    )
    for yi, v, sd in zip(y, summ.roc_auc_mean, summ.roc_auc_sd, strict=True):
        ax.annotate(
            f"{v:.2f}",
            (v + sd, yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
    ax.set_yticks(y, [MODEL_NAMES[m] for m in summ.model], color=INK)
    ax.set(
        xlabel="ROC-AUC, ProtT5 embeddings, cat_1 test genes (5-fold CV)",
        title="Model comparison",
        xlim=(0.77, 0.98),
    )
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2)
    save(fig, "readme_models.png")


def topk(table: pd.DataFrame) -> None:
    """Share of genes added to SFARI after 2024 found in the top of each ranking."""
    series = [
        ("protein_prott5", "Protein (ProtT5)", BLUE),
        ("graph_deepwalk", "Graph (DeepWalk)", ORANGE),
        ("baseline_protein_length", "Protein length (baseline)", GREY),
    ]
    tops, rand = ["top1", "top5", "top10"], [1, 5, 10]
    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(tops))
    w = 0.24
    for i, (key, name, color) in enumerate(series):
        vals = [100 * table.loc[key, f"{t}_frac"] for t in tops]
        bars = ax.bar(
            x + (i - 1) * w, vals, w * 0.92, color=color, label=name, edgecolor=SURFACE, lw=2
        )
        if key == "protein_prott5":
            ax.bar_label(bars, fmt="%.0f%%", padding=2, color=INK, fontsize=8)
    for xi, r in zip(x, rand, strict=True):
        ax.hlines(
            r,
            xi - 0.42,
            xi + 0.42,
            colors=INK,
            linestyles="--",
            lw=1,
            label="Random" if r == 1 else None,
        )
    ax.set_xticks(x, ["Top 1%", "Top 5%", "Top 10%"], color=INK)
    ax.set(
        ylabel="% of new SFARI genes",
        ylim=(0, 62),
        title=f"Top-k enrichment: {int(table.loc['protein_prott5', 'n_new'])} genes added "
        "to SFARI after 2024",
    )
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    save(fig, "readme_topk.png")


def baseline(table: pd.DataFrame, diffs: dict) -> None:
    """Temporal-validation AUC of each ranking vs the protein-length baseline."""
    rows = [
        ("protein_prott5", "Protein (ProtT5)", BLUE),
        ("graph_deepwalk", "Graph (DeepWalk)", ORANGE),
        ("baseline_protein_length", "Protein length", GREY),
    ]
    fig, ax = plt.subplots(figsize=SIZE)
    y = np.arange(len(rows))[::-1]
    for yi, (key, _name, color) in zip(y, rows, strict=True):
        auc, adj = table.loc[key, "auc"], table.loc[key, "auc_adjusted"]
        ax.hlines(yi, adj, auc, color=GRID, lw=3)
        ax.scatter(auc, yi, s=60, color=color, edgecolor=SURFACE, lw=2, zorder=3)
        ax.scatter(adj, yi, s=60, facecolor=SURFACE, edgecolor=color, lw=2, zorder=3)
        ax.annotate(
            f"{auc:.2f}",
            (auc, yi),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
        ax.annotate(
            f"{adj:.2f}",
            (adj, yi),
            xytext=(-10, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            color=MUTED,
        )
        if key in diffs:
            d = diffs[key]
            ax.annotate(
                f"AUC gain vs length: {d['diff']:+.2f} "
                f"(95% CI {d['ci_low']:+.2f} to {d['ci_high']:+.2f})",
                (0.515, yi - 0.3),
                va="center",
                ha="left",
                fontsize=8,
                color=MUTED,
            )
    ax.axvline(0.5, color=INK, ls="--", lw=1)
    ax.scatter([], [], s=50, color=MUTED, label="AUC")
    ax.scatter(
        [],
        [],
        s=50,
        facecolor=SURFACE,
        edgecolor=MUTED,
        lw=2,
        label="AUC among genes of similar length",
    )
    ax.set_yticks(y, [r[1] for r in rows], color=INK)
    ax.set(
        xlim=(0.48, 0.91),
        xlabel="AUC for genes added to SFARI after 2024",
        title="Beyond gene length: temporal-validation AUC",
        ylim=(-0.6, len(rows) - 0.4),
    )
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2)
    save(fig, "readme_baseline.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(RESULTS / "temporal_validation.csv", index_col="score")
    diffs = json.loads((RESULTS / "temporal_validation.json").read_text())["auc_diff_vs_length"]
    roc_pr()
    models()
    topk(table)
    baseline(table, diffs)


if __name__ == "__main__":
    main()
