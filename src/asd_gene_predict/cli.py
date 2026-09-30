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
    """Etapa 1: construir a lista de genes positivos (SFARI) e negativos (Krishnan)."""
    raise NotImplementedError("Por implementar — ver CLAUDE.md, Roadmap.")


@app.command()
def embed(kind: str = typer.Argument(..., help="dna | protein | graph")) -> None:
    """Etapa 2: gerar embeddings (DNABERT-2, ProtT5 ou GRAPE)."""
    raise NotImplementedError("Por implementar — ver CLAUDE.md, Roadmap.")


@app.command()
def train(
    model: str = typer.Option("all", help="lr | rf | svm | knn | lgbm | xgb | nb | all"),
) -> None:
    """Etapa 3: treinar e avaliar modelos com validação cruzada."""
    raise NotImplementedError("Por implementar — ver CLAUDE.md, Roadmap.")


@app.command()
def rank() -> None:
    """Etapa 4: gerar a lista ordenada de genes candidatos."""
    raise NotImplementedError("Por implementar — ver CLAUDE.md, Roadmap.")


if __name__ == "__main__":
    app()
