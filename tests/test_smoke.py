from asd_gene_predict import __version__
from asd_gene_predict.config import load_config


def test_version():
    assert __version__


def test_default_config_loads():
    cfg = load_config()
    assert cfg["cv"]["n_splits"] == 5
    assert "cat_1" in cfg["labels"]["positive_sets"]
