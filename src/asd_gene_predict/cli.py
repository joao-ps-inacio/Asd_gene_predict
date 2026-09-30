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
def embed(kind: str = typer.Argument(..., help="dna | protein | graph")) -> None:
    """Etapa 2: gerar embeddings (DNABERT-2, ProtT5 ou GRAPE)."""
    raise NotImplementedError("Por implementar — ver CLAUDE.md, Roadmap.")


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
def rank() -> None:
    """Etapa 4: gerar a lista ordenada de genes candidatos."""
    raise NotImplementedError("Por implementar — ver CLAUDE.md, Roadmap.")


if __name__ == "__main__":
    app()
