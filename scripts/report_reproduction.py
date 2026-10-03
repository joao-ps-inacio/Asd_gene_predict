"""Compare the thesis ProtT5 results with the corrected pipeline.

Reads ``reports/results/protein_prott5__*.parquet`` (from ``asd train``) and the published
thesis numbers in ``reports/reference/thesis_prott5_scaled.csv``; writes a summary CSV and a
dumbbell figure per metric (thesis → corrected) for the ``cat_1`` test set.

    python scripts/report_reproduction.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from asd_gene_predict.paths import REPORTS  # noqa: E402

RESULTS = REPORTS / "results"
FIGURES = REPORTS / "figures"
THESIS = REPORTS / "reference" / "thesis_prott5_scaled.csv"

MODEL_NAMES = {
    "lr": "Logistic Regression",
    "svm": "SVM",
    "knn": "KNN",
    "nb": "Naive Bayes",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
}

# reference palette (dataviz skill): one hue, two shades for before → after
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BEFORE, AFTER = "#86b6ef", "#1c5cab"


def load_runs() -> pd.DataFrame:
    files = sorted(RESULTS.glob("protein_prott5__*.parquet"))
    if not files:
        raise FileNotFoundError("Sem resultados. Correr `asd train --features protein_prott5`.")
    runs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return runs.drop_duplicates(["set", "model", "repeat", "fold"], keep="last")


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "average_precision",
        "precision_at_k",
        "mcc",
        "f1",
        "roc_auc_legacy",
        "ap_legacy",
    ]
    g = runs.groupby(["set", "model"])[metrics]
    out = g.mean().join(g.std(ddof=0), lsuffix="_mean", rsuffix="_sd")
    thesis = pd.read_csv(THESIS).set_index(["set", "model"])
    thesis = thesis[["roc_auc_mean", "average_precision_mean", "mcc_mean"]].add_prefix("thesis_")
    return out.join(thesis, how="left").reset_index()


def dumbbell(df: pd.DataFrame, before: str, after: str, title: str, ax) -> None:
    df = df.sort_values(after)
    y = range(len(df))
    ax.set_facecolor(SURFACE)
    ax.hlines(y, df[before], df[after], color=GRID, linewidth=2, zorder=1)
    ax.scatter(
        df[before],
        y,
        s=64,
        color=BEFORE,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=2,
        label="Tese (AUC/AUPRC sobre 0/1)",
    )
    ax.scatter(
        df[after],
        y,
        s=64,
        color=AFTER,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
        label="Corrigido (sobre probabilidades)",
    )
    for yi, (b, a) in enumerate(zip(df[before], df[after], strict=True)):
        ax.annotate(
            f"+{a - b:.2f}",
            (a, yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=MUTED,
        )
    ax.set_yticks(list(y), [MODEL_NAMES.get(m, m) for m in df["model"]], color=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    lo, hi = df[[before, after]].min().min(), df[[before, after]].max().max()
    ax.set_xlim(lo - 0.03, hi + 0.08)


def figure(summ: pd.DataFrame) -> None:
    cat1 = summ[summ["set"] == "cat_1"].dropna(subset=["thesis_roc_auc_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(cat1) + 1.8), facecolor=SURFACE)
    dumbbell(cat1, "thesis_roc_auc_mean", "roc_auc_mean", "AUC-ROC", axes[0])
    dumbbell(cat1, "thesis_average_precision_mean", "average_precision_mean", "AUPRC", axes[1])
    axes[1].set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK
    )
    fig.suptitle(
        "ProtT5, conjunto cat_1 (média de 5 folds)", x=0.01, ha="left", color=INK, fontsize=13
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "prott5_tese_vs_corrigido.png", dpi=160, facecolor=SURFACE)


SET_ORDER = ["cat_1", "cat_1_sd", "cat_1_2", "cat_1_2_sd", "cat_1_2_3", "complete"]
REPORT = REPORTS / "reproducao_prott5.md"


def table_cat1(summ: pd.DataFrame) -> str:
    rows = summ[(summ["set"] == "cat_1") & summ["thesis_roc_auc_mean"].notna()]
    lines = [
        "| Modelo | AUC tese | AUC (cálculo da tese, pipeline nova) | **AUC corrigido** "
        "| AUPRC tese | **AUPRC corrigido** |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in rows.sort_values("roc_auc_mean", ascending=False).iterrows():
        lines.append(
            f"| {MODEL_NAMES[r.model]} | {r.thesis_roc_auc_mean:.3f} | {r.roc_auc_legacy_mean:.3f} "
            f"| **{r.roc_auc_mean:.3f}** ± {r.roc_auc_sd:.3f} "
            f"| {r.thesis_average_precision_mean:.3f} "
            f"| **{r.average_precision_mean:.3f}** ± {r.average_precision_sd:.3f} |"
        )
    return "\n".join(lines)


def table_sets(summ: pd.DataFrame) -> str:
    wide = summ.pivot(index="set", columns="model", values="average_precision_mean")
    wide = wide.reindex([s for s in SET_ORDER if s in wide.index]).dropna(axis=1)
    models = [m for m in MODEL_NAMES if m in wide.columns]
    lines = [
        "| Positivos no treino | " + " | ".join(MODEL_NAMES[m] for m in models) + " |",
        "|---|" + "---:|" * len(models),
    ]
    for name, row in wide[models].iterrows():
        lines.append(f"| `{name}` | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    return "\n".join(lines)


def fill(text: str, key: str, content: str) -> str:
    start, end = f"<!-- {key} -->", f"<!-- /{key} -->"
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{content}\n{end}{tail}"


def main() -> None:
    summ = summary(load_runs())
    summ.to_csv(RESULTS / "protein_prott5_summary.csv", index=False, float_format="%.4f")
    figure(summ)
    text = REPORT.read_text(encoding="utf-8")
    text = fill(text, "TABELA_CAT1", table_cat1(summ))
    REPORT.write_text(fill(text, "TABELA_SETS", table_sets(summ)), encoding="utf-8")
    cols = [
        "set",
        "model",
        "thesis_roc_auc_mean",
        "roc_auc_legacy_mean",
        "roc_auc_mean",
        "thesis_average_precision_mean",
        "ap_legacy_mean",
        "average_precision_mean",
        "thesis_mcc_mean",
        "mcc_mean",
    ]
    print(summ[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
