from asd_gene_predict import __version__
from asd_gene_predict.config import load_config


def test_version():
    assert __version__


def test_default_config_loads():
    cfg = load_config()
    assert cfg["cv"]["n_splits"] == 5
    assert "cat_1" in cfg["labels"]["positive_sets"]


def test_cli_commands_registered():
    from typer.testing import CliRunner

    from asd_gene_predict.cli import app

    runner = CliRunner()
    out = runner.invoke(app, ["--help"]).output
    for cmd in ("fetch", "gene-map", "labels", "embed", "train", "rank"):
        assert cmd in out
    assert runner.invoke(app, ["fetch", "nao-existe"]).exit_code == 2
