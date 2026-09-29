import gzip
import shutil
from pathlib import Path

import pandas as pd
import pytest

from asd_gene_predict.data import gene_map as gm

FIX = Path(__file__).parent / "fixtures"
HGNC, MANE, STRING = (
    FIX / "hgnc_complete_set.txt",
    FIX / "mane_summary.txt",
    FIX / "string_aliases.txt",
)


@pytest.fixture(scope="module")
def gmap() -> pd.DataFrame:
    return gm.build_gene_map(HGNC, MANE, STRING).set_index("symbol")


def test_hgnc_filters(gmap):
    # withdrawn, without Ensembl ID, and duplicated Ensembl ID are dropped
    assert set(gmap.index) == {"SHANK3", "CHD8", "SCN2A", "ADNP", "NOENTREZ"}
    assert gmap.loc["CHD8", "entrez_id"] == 57680
    assert pd.isna(gmap.loc["NOENTREZ", "entrez_id"])
    assert str(gmap["entrez_id"].dtype) == "Int64"


def test_mane_prefers_select_and_keeps_versions(gmap):
    assert gmap.loc["SCN2A", "mane_status"] == "MANE Select"
    assert gmap.loc["SCN2A", "mane_ensembl_protein"] == "ENSP00000283256.6"
    assert gmap.loc["SHANK3", "mane_ensembl_transcript"] == "ENST00000710353.1"
    assert pd.isna(gmap.loc["ADNP", "mane_status"])


def test_string_mapping(gmap):
    assert gmap.loc["SHANK3", "string_id"] == "9606.ENSP00000262260"  # versioned alias works
    assert gmap.loc["CHD8", "string_id"] == "9606.ENSP00000399831"  # most supported wins
    # one STRING protein listed for two genes → ambiguous, both links dropped
    assert pd.isna(gmap.loc["SCN2A", "string_id"]) and pd.isna(gmap.loc["ADNP", "string_id"])


def test_gzip_inputs_give_same_result(tmp_path, gmap):
    gz = {}
    for name, src in {"hgnc": HGNC, "mane": MANE, "string": STRING}.items():
        gz[name] = tmp_path / f"{src.name}.gz"
        with open(src, "rb") as fi, gzip.open(gz[name], "wb") as fo:
            shutil.copyfileobj(fi, fo)
    again = gm.build_gene_map(gz["hgnc"], gz["mane"], gz["string"]).set_index("symbol")
    pd.testing.assert_frame_equal(again, gmap)


def test_parquet_roundtrip(tmp_path, gmap):
    path = gm.save_gene_map(gmap.reset_index()[gm.COLUMNS], tmp_path / "gene_map.parquet")
    back = gm.load_gene_map(path)
    assert list(back.columns) == gm.COLUMNS
    assert str(back["entrez_id"].dtype) == "Int64"


def test_validate_rejects_duplicates(gmap):
    bad = gmap.reset_index()
    bad.loc[1, "symbol"] = bad.loc[0, "symbol"]
    with pytest.raises(ValueError, match="symbol"):
        gm.validate_gene_map(bad[gm.COLUMNS])


def test_map_ids(gmap):
    table = gmap.reset_index()
    out = gm.map_ids(["57680", 85358, 424242], "entrez_id", "ensembl_gene_id", table)
    assert out.tolist()[:2] == ["ENSG00000100888", "ENSG00000251322"]
    assert pd.isna(out.iloc[2])
    assert gm.map_ids(["CHD8"], "symbol", "string_id", table).iloc[0] == "9606.ENSP00000399831"
    with pytest.raises(ValueError):
        gm.map_ids(["x"], "name", "symbol", table)
