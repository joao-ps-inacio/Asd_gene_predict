from pathlib import Path

import pandas as pd
import pytest

from asd_gene_predict.config import load_config
from asd_gene_predict.data import labels as lb

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def table() -> pd.DataFrame:
    return lb.build_labels(
        lb.read_sfari(FIX / "sfari_small.csv"), lb.read_krishnan(FIX / "krishnan_small.csv")
    )


def test_labels_rules(table):
    by_id = table.set_index("ensembl_gene_id")
    # SFARI wins over Krishnan, blank and duplicated negatives are dropped
    assert by_id.loc["ENSG00000000001", "label"] == 1
    assert sorted(by_id.index[by_id["label"] == 0]) == [
        "ENSG00000000002",
        "ENSG00000000003",
        "ENSG00000000004",
    ]
    # SFARI gene without Ensembl ID is dropped
    assert "MET" not in set(table["symbol"].dropna())
    assert pd.isna(by_id.loc["ENSG00000101126", "sfari_score"])
    assert bool(by_id.loc["ENSG00000251322", "syndromic"])


def test_missing_ids_recovered_with_gene_map():
    gmap = pd.DataFrame({"symbol": ["MET"], "ensembl_gene_id": ["ENSG00000105976"]})
    sfari = lb.read_sfari(FIX / "sfari_small.csv", gene_map=gmap)
    assert sfari.set_index("symbol").loc["MET", "ensembl_gene_id"] == "ENSG00000105976"


def test_exclusions(table):
    out = lb.build_labels(
        lb.read_sfari(FIX / "sfari_small.csv"),
        lb.read_krishnan(FIX / "krishnan_small.csv"),
        {"ENSG00000100888": "test", "ENSG00000000003": "test"},
    )
    assert len(out) == len(table) - 2


def test_positive_sets(table):
    counts = lb.summarize(table, lb.positive_sets(load_config())).set_index("set")
    assert counts.loc["cat_1", "positives"] == 2  # SHANK3, CHD8
    assert counts.loc["cat_1_sd", "positives"] == 3  # + ADNP (no score)
    assert counts.loc["cat_1_2", "positives"] == 4  # + SCN2A, OVERLAP
    assert counts.loc["cat_1_2_3", "positives"] == 5  # + ABAT
    assert counts.loc["complete", "positives"] == 6
    assert (counts["negatives"] == 3).all()


def test_validation_rejects_conflicts(table):
    bad = pd.concat([table, table.iloc[[0]].assign(label=0)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicados"):
        lb.validate_labels(bad)


def test_parquet_roundtrip(tmp_path, table):
    back = lb.load_labels(lb.save_labels(table, tmp_path / "labels.parquet"))
    pd.testing.assert_frame_equal(back, table)


def test_exclusions_file_is_valid():
    assert len(lb.load_exclusions()) == 3
