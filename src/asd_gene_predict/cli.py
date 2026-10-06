"""Ponto de entrada da linha de comandos: `asd <comando>`.

Cada comando corresponde a uma etapa do pipeline. Por agora são só esqueletos.
"""

import logging
from typing import Annotated

import typer

app = typer.Typer(help="Pipeline de previsão de genes de risco de ASD.", no_args_is_help=True)


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Mostrar logs detalhados.")
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def fetch(
    names: Annotated[
        list[str] | None, typer.Argument(help="Fontes a obter (por defeito, todas).")
    ] = None,
    stage: Annotated[
        str | None, typer.Option(help="Só as fontes desta etapa (gene_map, labels, graph).")
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Voltar a descarregar mesmo que o ficheiro exista.")
    ] = False,
    update_lock: Annotated[
        bool,
        typer.Option("--update-lock", help="Aceitar checksums novos (fonte mudou de propósito)."),
    ] = False,
) -> None:
    """Descarregar e verificar os dados brutos declarados em configs/sources.yaml."""
    from asd_gene_predict.data.sources import Status, load_sources, select_sources
    from asd_gene_predict.data.sources import fetch as do_fetch

    try:
        selected = select_sources(load_sources(), names or (), stage)
    except KeyError as e:
        raise typer.BadParameter(str(e.args[0])) from e

    results = do_fetch(selected, force=force, update_lock=update_lock)
    failed = False
    for r in results:
        digest = f" {r.sha256[:12]}" if r.sha256 else ""
        typer.echo(f"{r.status.value:>10}  {r.source.name}{digest}")
        if r.message:
            typer.echo(f"            {r.message}")
        failed |= r.status in (Status.MISSING, Status.MISMATCH)
    if failed:
        raise typer.Exit(1)


@app.command("gene-map")
def gene_map() -> None:
    """Construir data/processed/gene_map.parquet (Ensembl ↔ HGNC ↔ Entrez ↔ MANE ↔ STRING)."""
    from asd_gene_predict.data import gene_map as gm
    from asd_gene_predict.data.sources import load_sources

    src = load_sources()
    paths = {n: src[n].path() for n in ("hgnc", "mane", "string_aliases")}
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        typer.echo(f"Faltam ficheiros: {missing}. Correr `asd fetch --stage gene_map`.", err=True)
        raise typer.Exit(1)

    gmap = gm.build_gene_map(paths["hgnc"], paths["mane"], paths["string_aliases"])
    out = gm.save_gene_map(gmap)
    for k, v in gm.summarize(gmap).items():
        typer.echo(f"{k:>15}: {v}")
    typer.echo(f"Guardado em {out}")


@app.command()
def labels() -> None:
    """Etapa 1: construir data/processed/labels.parquet (SFARI + Krishnan)."""
    from asd_gene_predict.config import load_config
    from asd_gene_predict.data import labels as lb
    from asd_gene_predict.data.sources import load_sources

    cfg = load_config()
    src = load_sources()
    paths = {n: src[n].path() for n in ("sfari", "krishnan_2016")}
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        typer.echo(f"Faltam ficheiros: {missing}. Correr `asd fetch --stage labels`.", err=True)
        raise typer.Exit(1)

    gene_map = None
    if cfg["labels"].get("recover_missing_ensembl"):
        from asd_gene_predict.data.gene_map import load_gene_map

        gene_map = load_gene_map()

    table = lb.build_labels(
        lb.read_sfari(paths["sfari"], gene_map),
        lb.read_krishnan(paths["krishnan_2016"]),
        lb.load_exclusions(),
    )
    out = lb.save_labels(table)
    typer.echo(lb.summarize(table, lb.positive_sets(cfg)).to_string(index=False))
    typer.echo(f"Guardado em {out}")


@app.command()
def embed(
    kind: Annotated[str, typer.Argument(help="protein | graph")],
    legacy: Annotated[
        bool, typer.Option("--legacy", help="Importar os embeddings calculados na tese.")
    ] = False,
) -> None:
    """Etapa 2: importar os embeddings da tese (ProtT5 ou DeepWalk)."""
    if not legacy:
        raise typer.BadParameter("only --legacy is supported: the thesis embeddings are imported.")
    from asd_gene_predict.data.sources import load_sources
    from asd_gene_predict.embeddings.io import embedding_path, save_embeddings
    from asd_gene_predict.embeddings.legacy import LEGACY_META, read_legacy, read_legacy_graph

    legacy_sets = {
        "protein": ("legacy_prott5", read_legacy, "protein_prott5", "prott5"),
        "graph": ("legacy_graph_deepwalk", read_legacy_graph, "graph_deepwalk", "deepwalk"),
    }
    if kind not in legacy_sets:
        raise typer.BadParameter("Com --legacy, só `protein` (ProtT5) e `graph` (DeepWalk).")
    source, reader, name, meta = legacy_sets[kind]
    src = load_sources()[source].path()
    if not src.exists():
        typer.echo(f"Falta o ficheiro. Correr `asd fetch {source}`.", err=True)
        raise typer.Exit(1)
    df = reader(src)
    out = save_embeddings(df, embedding_path(name), LEGACY_META[meta])
    typer.echo(f"{len(df)} genes × {df.shape[1] - 1} dimensões → {out}")


@app.command()
def train(
    features: Annotated[
        str, typer.Option(help="Embeddings a usar, ex.: protein_prott5 (data/processed/emb_*).")
    ],
    model: Annotated[
        list[str], typer.Option(help="lr | svm | rf | knn | lgbm | xgb | nb | all (repetível).")
    ] = ["all"],  # noqa: B006
    sets: Annotated[
        list[str] | None, typer.Option("--set", help="Conjuntos de positivos (por defeito, todos).")
    ] = None,
    repeats: Annotated[
        int | None, typer.Option(help="Repetições da CV (sobrepõe a config).")
    ] = None,
    scoring: Annotated[str | None, typer.Option(help="Métrica da afinação (ex.: f1).")] = None,
) -> None:
    """Etapa 3: treinar e avaliar modelos com validação cruzada (métricas corrigidas)."""
    from asd_gene_predict.config import load_config
    from asd_gene_predict.data.labels import load_labels, positive_sets
    from asd_gene_predict.embeddings.io import embedding_path, load_embeddings
    from asd_gene_predict.models import evaluate as ev
    from asd_gene_predict.models.registry import resolve
    from asd_gene_predict.paths import REPORTS

    cfg = load_config()
    cv = cfg["cv"]
    try:
        models = resolve(model)
    except KeyError as e:
        raise typer.BadParameter(str(e.args[0])) from e
    all_sets = {s.name: s for s in positive_sets(cfg)}
    chosen = [all_sets[n] for n in sets] if sets else list(all_sets.values())
    if cv["test_set"] not in {s.name for s in chosen}:
        chosen.insert(0, all_sets[cv["test_set"]])

    results = ev.cross_validate(
        load_embeddings(embedding_path(features)),
        load_labels(),
        chosen,
        models,
        test_set=cv["test_set"],
        n_splits=cv["n_splits"],
        n_repeats=repeats or cv.get("n_repeats", 1),
        inner_splits=cv.get("inner_splits", 5),
        scoring=scoring or cv["scoring"],
        seed=cfg["seed"],
    )
    out = REPORTS / "results" / f"{features}__{'-'.join(models)}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.assign(features=features).to_parquet(out, index=False)
    metrics = ["average_precision", "roc_auc", "precision_at_k", "mcc"]
    typer.echo(ev.format_table(results, metrics).to_string())
    typer.echo(f"Guardado em {out}")


@app.command()
def compare(
    features: Annotated[
        list[str], typer.Option(help="Embeddings a comparar (repetível), ex.: protein_prott5.")
    ],
    model: Annotated[
        list[str], typer.Option(help="lr | svm | rf | knn | lgbm | xgb | nb (repetível).")
    ] = ["lr"],  # noqa: B006
    fusion: Annotated[
        bool, typer.Option(help="Juntar também todas as fontes numa só (concatenação).")
    ] = True,
    repeats: Annotated[int, typer.Option(help="Repetições da CV de 5 folds.")] = 5,
    cross_category: Annotated[
        bool,
        typer.Option(help="Também treinar na categoria 1 e testar nas categorias 2/3 (só LR)."),
    ] = False,
) -> None:
    """Comparação justa entre fontes: mesmos genes, mesmos folds, mesmos modelos."""
    from asd_gene_predict.config import load_config
    from asd_gene_predict.data.labels import load_labels, positive_sets
    from asd_gene_predict.embeddings.io import embedding_path, load_embeddings
    from asd_gene_predict.models import compare as cmp
    from asd_gene_predict.models.registry import resolve
    from asd_gene_predict.paths import REPORTS

    cfg = load_config()
    cv = cfg["cv"]
    try:
        models = resolve(model)
    except KeyError as e:
        raise typer.BadParameter(str(e.args[0])) from e
    sets = cmp.feature_sets({f: load_embeddings(embedding_path(f)) for f in features}, fusion)
    test_set = {s.name: s for s in positive_sets(cfg)}[cv["test_set"]]
    results = cmp.compare_sources(
        sets,
        load_labels(),
        test_set,
        models,
        test_set=cv["test_set"],
        n_splits=cv["n_splits"],
        n_repeats=repeats,
        inner_splits=cv.get("inner_splits", 5),
        scoring=cv["scoring"],
        seed=cfg["seed"],
    )
    out = REPORTS / "results" / f"comparison__{'-'.join(features)}__{'-'.join(models)}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out, index=False)
    summary = results.groupby(["features", "model"])[["roc_auc", "average_precision"]].agg(
        ["mean", "std"]
    )
    typer.echo(summary.round(3).to_string())
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            typer.echo(cmp.paired_comparison(results, a, b).round(4).to_string(index=False))
    typer.echo(f"Guardado em {out}")
    if cross_category:
        gen = cmp.cross_category(
            sets,
            load_labels(),
            "lr",
            n_repeats=repeats,
            inner_splits=cv.get("inner_splits", 5),
            scoring=cv["scoring"],
            seed=cfg["seed"],
        )
        gen_out = REPORTS / "results" / "generalization_cat23_lr.parquet"
        gen.to_parquet(gen_out, index=False)
        typer.echo(gen.groupby("features")[["roc_auc", "average_precision"]].mean().round(3))
        typer.echo(f"Guardado em {gen_out}")


@app.command("validate-temporal")
def validate_temporal(
    sfari_new: Annotated[
        str | None,
        typer.Option(help="CSV do SFARI recente (por defeito, a fonte sfari_2026q2)."),
    ] = None,
    shared_pool: Annotated[
        bool, typer.Option(help="Avaliar todos os scores nos mesmos genes.")
    ] = True,
) -> None:
    """Validação temporal: os rankings da tese anteciparam os genes que o SFARI acrescentou?"""
    import json
    from pathlib import Path

    from asd_gene_predict.data import labels as lb
    from asd_gene_predict.data.sources import load_sources
    from asd_gene_predict.paths import REPORTS
    from asd_gene_predict.validation import temporal as tv

    src = load_sources()
    new_path = Path(sfari_new) if sfari_new else src["sfari_2026q2"].path()
    needed = {
        "sfari": src["sfari"].path(),
        "krishnan_2016": src["krishnan_2016"].path(),
        "sfari_new": new_path,
        "legacy_rank_protein": src["legacy_rank_protein"].path(),
        "legacy_rank_graph": src["legacy_rank_graph"].path(),
    }
    missing = [n for n, p in needed.items() if not p.exists()]
    if missing:
        typer.echo(f"Faltam ficheiros: {missing}. Correr `asd fetch --stage validation`.", err=True)
        raise typer.Exit(1)

    old = lb.read_sfari(needed["sfari"])
    new = lb.read_sfari(new_path)
    added = tv.new_genes(old, new)
    exclude = set(old["ensembl_gene_id"]) | set(lb.read_krishnan(needed["krishnan_2016"]))

    scores = {
        "protein_prott5": tv.read_legacy_ranking(needed["legacy_rank_protein"]),
        "graph_deepwalk": tv.read_legacy_ranking(needed["legacy_rank_graph"]),
    }
    length = None
    if src["legacy_peptides"].path().exists():
        length = tv.protein_length(src["legacy_peptides"].path())
        scores["baseline_protein_length"] = length
    if src["gnomad_constraint"].path().exists():
        scores["baseline_loeuf"] = tv.loeuf_score(src["gnomad_constraint"].path())

    positives = set(added["ensembl_gene_id"])
    table = tv.compare(scores, positives, exclude, shared_pool=shared_pool, confounder=length)
    out_dir = REPORTS / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "temporal_validation.csv", float_format="%.6g")

    symbol = dict(zip(new["ensembl_gene_id"], new["symbol"], strict=True))
    summary = {
        "new_genes": int(len(added)),
        "new_by_score": {
            str(k): int(v) for k, v in added["sfari_score"].value_counts(dropna=False).items()
        },
        "promoted": [symbol.get(g, g) for g in tv.promoted_genes(old, new)["ensembl_gene_id"]],
        "top50_hits": {
            name: [symbol.get(g, g) for g in tv.top_hits(s, positives, exclude)]
            for name, s in scores.items()
        },
    }
    if length is not None:
        pool = {k: v[~v.index.isin(exclude)] for k, v in scores.items()}
        summary["auc_diff_vs_length"] = {
            name: dict(
                zip(
                    ("diff", "ci_low", "ci_high"),
                    tv.bootstrap_auc_diff(pool[name], length, positives),
                    strict=True,
                )
            )
            for name in ("protein_prott5", "graph_deepwalk")
        }
    (out_dir / "temporal_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cols = ["n_pool", "n_new", "auc", "auprc", "median_percentile", "top1_frac", "top10_frac"]
    cols += ["auc_adjusted"] if "auc_adjusted" in table else []
    typer.echo(f"{len(added)} genes novos no SFARI.")
    typer.echo(table[cols].astype(float).round(3).to_string())
    typer.echo(f"Guardado em {out_dir / 'temporal_validation.csv'}")


if __name__ == "__main__":
    app()
