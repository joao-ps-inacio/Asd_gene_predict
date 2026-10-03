"""Figure and tables for the temporal validation (``asd validate-temporal`` must run first).

python scripts/report_temporal.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from asd_gene_predict.paths import REPORTS  # noqa: E402

RESULTS = REPORTS / "results"
REPORT = REPORTS / "temporal_validation.md"
FIGURE = REPORTS / "figures" / "temporal_validation.png"

NAMES = {
    "protein_prott5": "Protein (ProtT5)",
    "graph_deepwalk": "Graph (DeepWalk)",
    "baseline_protein_length": "Protein length",
    "baseline_loeuf": "Gene constraint (LOEUF)",
}
# reference palette (dataviz skill): categorical slots 1-2; baselines in neutral grey
COLORS = {
    "protein_prott5": "#2a78d6",
    "graph_deepwalk": "#eb6834",
    "baseline_protein_length": "#a3a29c",
    "baseline_loeuf": "#6f6e69",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(table: pd.DataFrame) -> None:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]}
    )
    tops = ["top1", "top5", "top10"]
    x = np.arange(len(tops))
    width = 0.8 / len(table)
    for i, (name, row) in enumerate(table.iterrows()):
        vals = [100 * row[f"{t}_frac"] for t in tops]
        bars = left.bar(
            x + (i - (len(table) - 1) / 2) * width,
            vals,
            width * 0.92,
            color=COLORS[name],
            label=NAMES[name],
            edgecolor=SURFACE,
            linewidth=2,
        )
        if name == "protein_prott5":
            left.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK, padding=2)
    for xi, top in zip(x, (1, 5, 10), strict=True):
        left.hlines(top, xi - 0.42, xi + 0.42, colors=INK, linestyles=(0, (3, 2)), linewidth=1)
    left.plot([], [], color=INK, linestyle=(0, (3, 2)), linewidth=1, label="Random")
    left.set_xticks(x, ["Top 1%", "Top 5%", "Top 10%"], color=INK)
    left.set_ylabel("% of new SFARI genes", color=MUTED, fontsize=9)
    left.set_title(
        "Genes added to SFARI after 2024, by ranking position",
        loc="left",
        color=INK,
        fontsize=11,
    )
    left.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    style(left)

    order = list(table.index)
    y = np.arange(len(order))[::-1]
    right.hlines(y, table["auc_adjusted"], table["auc"], color=GRID, linewidth=2)
    right.scatter(
        table["auc"],
        y,
        s=60,
        color=[COLORS[n] for n in order],
        zorder=3,
        edgecolor=SURFACE,
        linewidth=2,
        label="AUC",
    )
    right.scatter(
        table["auc_adjusted"],
        y,
        s=60,
        facecolor=SURFACE,
        zorder=3,
        edgecolor=[COLORS[n] for n in order],
        linewidth=2,
        label="AUC among genes of similar length",
    )
    right.axvline(0.5, color=INK, linestyle=(0, (3, 2)), linewidth=1)
    right.set_yticks(y, [NAMES[n] for n in order], color=INK, fontsize=9)
    right.set_xlim(0.45, 0.9)
    right.set_title("Controlling for gene length", loc="left", color=INK, fontsize=11)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=7, label="AUC"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=SURFACE,
            color=MUTED,
            markeredgewidth=2,
            markersize=7,
            label="AUC among genes of similar length",
        ),
    ]
    right.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(-0.05, -0.08),
        ncol=1,
        labelcolor=INK,
    )
    style(right)
    right.grid(axis="x", color=GRID, linewidth=0.8)
    right.grid(axis="y", visible=False)

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160, facecolor=SURFACE)


def table_md(table: pd.DataFrame) -> str:
    lines = [
        "| Score | AUC | Length-adjusted AUC | Top 1% | Top 10% | Median percentile |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in table.iterrows():
        lines.append(
            f"| {NAMES[name]} | {r.auc:.2f} | {r.auc_adjusted:.2f} | {r.top1_frac:.0%} "
            f"({r.top1_enrichment:.0f}×) | {r.top10_frac:.0%} ({r.top10_enrichment:.1f}×) "
            f"| {r.median_percentile:.0%} |"
        )
    return "\n".join(lines)


def fill(text: str, key: str, content: str) -> str:
    start, end = f"<!-- {key} -->", f"<!-- /{key} -->"
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{content}\n{end}{tail}"


def main() -> None:
    table = pd.read_csv(RESULTS / "temporal_validation.csv", index_col="score")
    summary = json.loads((RESULTS / "temporal_validation.json").read_text(encoding="utf-8"))
    figure(table)
    if REPORT.exists():
        REPORT.write_text(
            fill(REPORT.read_text(encoding="utf-8"), "TABELA", table_md(table)), encoding="utf-8"
        )
    print(table_md(table))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
