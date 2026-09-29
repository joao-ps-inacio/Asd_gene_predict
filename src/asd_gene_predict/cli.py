"""Ponto de entrada da linha de comandos: `asd <comando>`.

Cada comando corresponde a uma etapa do pipeline. Por agora são só esqueletos.
"""

import typer

app = typer.Typer(help="Pipeline de previsão de genes de risco de ASD.", no_args_is_help=True)


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
