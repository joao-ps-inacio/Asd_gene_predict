"""Report for the protein vs graph comparison (``asd compare`` must run first).

Writes ``reports/embedding_comparison.md`` tables and ``reports/figures/embedding_comparison.png``.

    python scripts/report_comparison.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asd_gene_predict.models.compare import paired_comparison  # noqa: E402
from asd_gene_predict.paths import REPORTS  # noqa: E402

RESULTS = REPORTS / "results"
REPORT = REPORTS / "embedding_comparison.md"
FIGURE = REPORTS / "figures" / "embedding_comparison.png"

P, G, B = "protein_prott5", "graph_deepwalk", "protein_prott5+graph_deepwalk"
NAMES = {P: "Protein (ProtT5)", G: "Graph (DeepWalk)", B: "Protein + graph"}
MODELS = {"lr": "Logistic Regression", "svm": "SVM", "knn": "KNN"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {P: "#2a78d6", G: "#eb6834", B: "#a3a29c"}


def load() -> pd.DataFrame:
    files = sorted(RESULTS.glob(f"comparison__{P}-{G}__*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def cv_table(res: pd.DataFrame) -> str:
    g = res.groupby(["model", "features"])[["roc_auc", "average_precision"]]
    mean, sd = g.mean(), g.std()
    lines = ["| Model | Features | ROC-AUC | AUPRC |", "|---|---|---:|---:|"]
    for m in [m for m in MODELS if m in res.model.unique()]:
        for f in (P, G, B):
            mu, s = mean.loc[(m, f)], sd.loc[(m, f)]
            lines.append(
                f"| {MODELS[m]} | {NAMES[f]} | {mu.roc_auc:.3f} ± {s.roc_auc:.3f} "
                f"| {mu.average_precision:.3f} ± {s.average_precision:.3f} |"
            )
    return "\n".join(lines)


def test_table(res: pd.DataFrame) -> str:
    lines = [
        "| Model | Comparison | Metric | Difference | 95% CI | p |",
        "|---|---|---|---:|---|---:|",
    ]
    for a, b in ((G, P), (B, G)):
        t = paired_comparison(res, a, b)
        for _, r in t.iterrows():
            metric = "ROC-AUC" if r.metric == "roc_auc" else "AUPRC"
            p = "< 0.001" if r.p < 0.001 else f"{r.p:.3f}"
            lines.append(
                f"| {MODELS[r.model]} | {NAMES[a]} − {NAMES[b]} | {metric} | "
                f"{r['diff']:+.3f} | {r.ci_low:+.3f} to {r.ci_high:+.3f} | {p} |"
            )
    return "\n".join(lines)


def figure(res: pd.DataFrame, gen: pd.DataFrame, temporal: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11, 4.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.15]}
    )
    models = [m for m in MODELS if m in res.model.unique()]
    x = np.arange(len(models))
    for i, f in enumerate((P, G, B)):
        mu = [res[(res.model == m) & (res.features == f)].roc_auc.mean() for m in models]
        sd = [res[(res.model == m) & (res.features == f)].roc_auc.std() for m in models]
        a.errorbar(
            x + (i - 1) * 0.22,
            mu,
            yerr=sd,
            fmt="o",
            color=COLORS[f],
            ms=7,
            mec=SURFACE,
            mew=1.5,
            capsize=3,
            label=NAMES[f],
        )
    a.set_xticks(x, [MODELS[m] for m in models], color=INK)
    a.set(ylabel="ROC-AUC (5 × 5-fold CV, mean ± sd)", ylim=(0.84, 0.99), facecolor=SURFACE)
    a.set_title("Same genes, same folds: known SFARI genes", loc="left", color=INK, fontsize=10.5)
    a.grid(True, axis="y", color=GRID)
    a.set_axisbelow(True)
    a.legend(frameon=False, loc="lower left", fontsize=8)

    settings = [
        (
            "Cross-validation\n(category 1 genes)",
            res[res.model == "lr"].groupby("features").roc_auc.mean(),
        ),
        (
            "Train cat. 1 → test\ncat. 2/3 genes",
            gen.groupby("features").roc_auc.mean().rename({"protein": P, "graph": G}),
        ),
        (
            "Genes added to SFARI after\n2024 (thesis rankings)",
            temporal.auc.rename({"protein_prott5": P, "graph_deepwalk": G}),
        ),
    ]
    xs = np.arange(len(settings))
    for f in (P, G):
        b.plot(
            xs,
            [s[1][f] for s in settings],
            color=COLORS[f],
            lw=2,
            marker="o",
            ms=8,
            mec=SURFACE,
            mew=1.5,
            label=NAMES[f],
        )
        for xi, s in zip(xs, settings, strict=True):
            b.annotate(
                f"{s[1][f]:.2f}",
                (xi, s[1][f]),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                color=COLORS[f],
                fontsize=8.5,
            )
    b.set_xticks(xs, [s[0] for s in settings], color=INK)
    b.set(ylabel="ROC-AUC", ylim=(0.65, 1.0), facecolor=SURFACE, xlim=(-0.3, 2.45))
    b.set_title("Known genes vs newly discovered genes", loc="left", color=INK, fontsize=10.5)
    b.grid(True, axis="y", color=GRID)
    b.set_axisbelow(True)
    b.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150, facecolor=SURFACE)


def fill(text: str, key: str, content: str) -> str:
    start, end = f"<!-- {key} -->", f"<!-- /{key} -->"
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{content}\n{end}{tail}"


def main() -> None:
    res = load()
    gen = pd.read_parquet(RESULTS / "generalization_cat23_lr.parquet")
    temporal = pd.read_csv(RESULTS / "temporal_validation.csv", index_col="score")
    figure(res, gen, temporal)
    gen_test = paired_comparison(gen, G, P).set_index("metric")
    text = REPORT.read_text(encoding="utf-8")
    text = fill(text, "CV", cv_table(res))
    text = fill(text, "TESTS", test_table(res))
    REPORT.write_text(text, encoding="utf-8")
    print(
        cv_table(res),
        test_table(res),
        gen.groupby("features")[["roc_auc"]].mean(),
        gen_test,
        json.loads((RESULTS / "temporal_validation.json").read_text())["top50_hits"],
        sep="\n\n",
    )


if __name__ == "__main__":
    main()
