import numpy as np
import pandas as pd

from asd_gene_predict.embeddings.legacy import parse_vector, read_legacy


def test_parse_vector():
    np.testing.assert_allclose(parse_vector("[0.5, -1.25, 3e-2]"), [0.5, -1.25, 0.03])


def test_read_legacy(tmp_path):
    path = tmp_path / "complete.csv.gz"
    pd.DataFrame(
        {
            "0": ["SHANK3", "CHD8", "SHANK3"],
            "1": ["ENSG00000251322", "ENSG00000100888", "ENSG00000251322"],
            "2": ["ENSP1", "ENSP2", "ENSP1"],
            "3": ["[1.0, 2.0]", "[3.0, 4.0]", "[1.0, 2.0]"],
            "4": [1, 1, 1],
        }
    ).to_csv(path, index=False, compression="gzip")
    df = read_legacy(path)
    assert list(df.columns) == ["ensembl_gene_id", "f0", "f1"]
    assert len(df) == 2 and df["f1"].dtype == np.float32
