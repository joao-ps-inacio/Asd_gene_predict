import numpy as np
import pandas as pd
import pytest

from asd_gene_predict.validation import temporal as tv


def sfari(rows):
    return pd.DataFrame(rows, columns=["ensembl_gene_id", "symbol", "sfari_score"]).astype(
        {"sfari_score": "Int64"}
    )


def test_new_and_promoted_genes():
    old = sfari([("G1", "A", 1), ("G2", "B", 2), ("G3", "C", 3)])
    new = sfari([("G1", "A", 1), ("G2", "B", 1), ("G3b", "C", 3), ("G4", "D", 3)])
    assert tv.new_genes(old, new)["symbol"].tolist() == ["D"]  # C changed ID, same symbol
    assert tv.promoted_genes(old, new)["ensembl_gene_id"].tolist() == ["G2"]


@pytest.fixture
def ranked():
    ids = [f"G{i}" for i in range(1000)]
    scores = pd.Series(np.linspace(1, 0, 1000), index=ids)  # G0 is the top gene
    return scores


def test_perfect_and_random_rankings(ranked):
    top = tv.evaluate_ranking(ranked, [f"G{i}" for i in range(10)])
    assert top["auc"] == pytest.approx(1.0)
    assert top["top1_frac"] == 1.0 and top["top1_enrichment"] == pytest.approx(100)
    assert top["top1_p"] < 1e-15

    rng = np.random.default_rng(0)
    spread = tv.evaluate_ranking(ranked, rng.choice(ranked.index, 100, replace=False))
    assert 0.4 < spread["auc"] < 0.6 and 0.3 < spread["median_percentile"] < 0.7


def test_compare_uses_shared_pool_and_excludes_known(ranked):
    other = ranked.iloc[:800] * 0 + np.arange(800)  # covers fewer genes
    res = tv.compare({"a": ranked, "b": other}, ["G5", "G900"], exclude=["G0"])
    assert (res["n_pool"] == 799).all()  # 800 shared genes minus the excluded one
    assert (res["n_new"] == 1).all()  # G900 is outside the shared pool


def test_stratified_auc_removes_confounder():
    rng = np.random.default_rng(1)
    n = 4000
    length = pd.Series(rng.lognormal(6, 1, n), index=[f"G{i}" for i in range(n)])
    # positives are just the longest genes; a score equal to length looks great overall
    positives = length.sort_values().index[-200:]
    assert tv.evaluate_ranking(length, positives)["auc"] > 0.9
    noise = pd.Series(rng.normal(size=n), index=length.index)
    assert tv.stratified_auc(noise, length, positives) == pytest.approx(0.5, abs=0.1)


def test_bootstrap_auc_diff(ranked):
    positives = [f"G{i}" for i in range(50)]
    worse = pd.Series(np.random.default_rng(2).random(1000), index=ranked.index)
    diff, lo, hi = tv.bootstrap_auc_diff(ranked, worse, positives, n_boot=200)
    assert diff > 0.3 and lo > 0 and lo <= diff <= hi


def test_readers(tmp_path):
    rank = tmp_path / "rank.csv"
    pd.DataFrame(
        {
            "Gene": ["A", "B", "B"],
            "Ensembl_ID": ["G1", "G2", "G2"],
            "Probability_Class_0": [0.1, 0.8, 0.8],
            "Probability_Class_1": [0.9, 0.2, 0.2],
        }
    ).to_csv(rank, index=False)
    assert tv.read_legacy_ranking(rank).to_dict() == {"G1": 0.9, "G2": 0.2}

    pep = tmp_path / "pep.csv"
    pd.DataFrame(
        {"Gene stable ID": ["G1", "G1", "G2"], "protein_seq": ["MAAA", "MA", "MAAAAA"]}
    ).to_csv(pep, index=False)
    assert tv.protein_length(pep).to_dict() == {"G1": 4.0, "G2": 6.0}

    gn = tmp_path / "gnomad.tsv"
    pd.DataFrame(
        {
            "gene_id": ["ENSG00000000002", "ENSG00000000002", "ENSG00000000001"],
            "mane_select": [True, False, True],
            "lof.oe_ci.upper": [0.2, 0.9, 1.5],
        }
    ).to_csv(gn, sep="\t", index=False)
    loeuf = tv.loeuf_score(gn)
    assert loeuf["ENSG00000000002"] == pytest.approx(-0.2)  # MANE transcript only
    assert loeuf.idxmax() == "ENSG00000000002"  # more constrained = higher score
