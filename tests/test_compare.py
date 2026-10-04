import numpy as np
import pandas as pd
import pytest

from asd_gene_predict.data.labels import PositiveSet
from asd_gene_predict.embeddings.io import to_frame
from asd_gene_predict.models import compare as cmp


def emb(ids, dim, seed):
    rng = np.random.default_rng(seed)
    return to_frame(ids, rng.normal(size=(len(ids), dim))).set_index("ensembl_gene_id")


def test_feature_sets_use_shared_genes_and_fusion():
    a = emb([f"G{i}" for i in range(10)], 3, 0)
    b = emb([f"G{i}" for i in range(5, 15)], 2, 1)
    sets = cmp.feature_sets({"a": a, "b": b})
    assert list(sets) == ["a", "b", "a+b"]
    assert all(list(df.index) == [f"G{i}" for i in range(5, 10)] for df in sets.values())
    assert sets["a+b"].shape[1] == 5 and sets["a+b"].columns[0] == "a:f0"
    assert list(cmp.feature_sets({"a": a, "b": b}, fusion=False)) == ["a", "b"]


def test_corrected_ttest_is_wider_than_naive():
    diffs = np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.02, 0.01, 0.03, 0.04])
    naive = cmp.corrected_ttest(diffs, test_train_ratio=0.0)
    corrected = cmp.corrected_ttest(diffs, test_train_ratio=0.25)
    assert naive["diff"] == pytest.approx(diffs.mean())
    assert corrected["ci_high"] - corrected["ci_low"] > naive["ci_high"] - naive["ci_low"]
    assert corrected["p"] > naive["p"]


def test_paired_comparison_pairs_folds():
    rows = []
    for feat, shift in (("a", 0.05), ("b", 0.0)):
        for r in range(2):
            for f in range(5):
                base = 0.8 + 0.01 * f  # fold difficulty shared by both sources
                rows.append(
                    {
                        "features": feat,
                        "model": "lr",
                        "repeat": r,
                        "fold": f,
                        "roc_auc": base + shift,
                        "average_precision": base,
                        "n_test": 20,
                        "n_train": 80,
                    }
                )
    res = cmp.paired_comparison(pd.DataFrame(rows), "a", "b").set_index("metric")
    assert res.loc["roc_auc", "diff"] == pytest.approx(0.05)
    assert res.loc["average_precision", "diff"] == pytest.approx(0.0)
    assert res.loc["roc_auc", "n"] == 10


def test_compare_sources_same_folds():
    ids = [f"ENSG{i:011d}" for i in range(120)]
    label = np.r_[np.zeros(80), np.ones(40)].astype("int8")
    labels = pd.DataFrame(
        {
            "ensembl_gene_id": ids,
            "symbol": pd.array([pd.NA] * 120, dtype="string"),
            "label": label,
            "sfari_score": pd.array([pd.NA] * 80 + [1] * 40, dtype="Int64"),
            "syndromic": False,
            "source": "test",
        }
    )
    signal = emb(ids, 4, 0)
    signal.loc[signal.index[label == 1]] += 1.5
    sets = cmp.feature_sets({"signal": signal, "noise": emb(ids, 4, 1)})
    res = cmp.compare_sources(
        sets,
        labels,
        PositiveSet("cat_1", 1),
        ["nb"],
        n_splits=3,
        n_repeats=2,
        inner_splits=3,
        n_jobs=1,
    )
    sizes = res.pivot_table(index=["repeat", "fold"], columns="features", values="n_test")
    assert (sizes.nunique(axis=1) == 1).all()  # identical test folds for every source
    test = cmp.paired_comparison(res, "signal", "noise").set_index("metric")
    assert test.loc["roc_auc", "diff"] > 0.2
