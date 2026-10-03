import numpy as np
import pandas as pd
import pytest

from asd_gene_predict.data.labels import PositiveSet
from asd_gene_predict.embeddings.io import load_embeddings, read_metadata, save_embeddings, to_frame
from asd_gene_predict.models import evaluate as ev
from asd_gene_predict.models.registry import MODELS, build_pipeline, pipeline_grid, resolve

SETS = [PositiveSet("cat_1", 1), PositiveSet("cat_1_2", 2)]


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """90 negatives, 30 cat-1 and 30 cat-2 positives; positives shifted on 3 features."""
    rng = np.random.default_rng(0)
    n_neg, n_c1, n_c2 = 90, 30, 30
    ids = [f"ENSG{i:011d}" for i in range(n_neg + n_c1 + n_c2)]
    label = np.r_[np.zeros(n_neg), np.ones(n_c1 + n_c2)].astype("int8")
    score = pd.array([pd.NA] * n_neg + [1] * n_c1 + [2] * n_c2, dtype="Int64")
    X = rng.normal(size=(len(ids), 8))
    X[label == 1, :3] += 1.5
    labels = pd.DataFrame(
        {
            "ensembl_gene_id": ids,
            "symbol": pd.array([pd.NA] * len(ids), dtype="string"),
            "label": label,
            "sfari_score": score,
            "syndromic": False,
            "source": "test",
        }
    )
    return to_frame(ids, X).set_index("ensembl_gene_id"), labels


def test_metrics_use_scores_not_labels():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    m = ev.compute_metrics(y, scores, (scores > 0.5).astype(int))
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["roc_auc_legacy"] == pytest.approx(0.75)  # 0/1 predictions: [0,0,0,1]
    assert m["precision_at_k"] == pytest.approx(0.5)
    perfect = ev.compute_metrics(y, np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 0, 0]))
    assert perfect["roc_auc"] == 1.0 and perfect["roc_auc_legacy"] == 0.5


def test_precision_at_k_ignores_row_order_on_ties():
    y = np.array([1, 1, 0, 0])
    tied = np.ones(4)
    assert ev.precision_at_k(y, tied, 2) == pytest.approx(0.5)
    assert ev.precision_at_k(y[::-1], tied, 2) == pytest.approx(0.5)
    assert ev.precision_at_k(y, np.array([0.9, 0.5, 0.5, 0.1]), 2) == pytest.approx(0.75)


def test_folds_are_shared_and_leak_free(data):
    X, labels = data
    res = ev.cross_validate(X, labels, SETS, ["lr"], n_splits=3, inner_splits=3, n_jobs=1)
    assert len(res) == 2 * 3
    # every set is tested on the same cat_1 folds
    tests = res.pivot(index="fold", columns="set", values="n_test")
    assert (tests["cat_1"] == tests["cat_1_2"]).all()
    assert res.groupby("set")["n_test"].sum().eq(120).all()
    # larger set only adds positives to training
    train_pos = res.pivot(index="fold", columns="set", values="n_train_pos")
    assert (train_pos["cat_1_2"] == train_pos["cat_1"] + 30).all()
    assert res["roc_auc"].mean() > 0.8


def test_folds_cover_test_pool_once():
    pool = pd.DataFrame(
        {"ensembl_gene_id": [f"g{i}" for i in range(20)], "label": [0] * 14 + [1] * 6}
    )
    folds = ev.make_folds(pool, n_splits=4, n_repeats=2, seed=1)
    for r in (0, 1):
        genes = [g for rep, _, ids in folds if rep == r for g in ids]
        assert sorted(genes) == sorted(pool["ensembl_gene_id"])
    assert folds[0][2] != folds[4][2]  # repeats reshuffle


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_fits_and_scores(name, data):
    X, labels = data
    y = labels["label"].to_numpy()
    grid = pipeline_grid(name)
    params = {k: v[0] for k, v in grid[0].items()}
    model = build_pipeline(name, 0, y).set_params(**params).fit(X.to_numpy(), y)
    scores = ev.continuous_scores(model, X.to_numpy())
    assert scores.shape == (len(y),) and np.isfinite(scores).all()


def test_xgb_uses_scale_pos_weight(data):
    _, labels = data
    y = labels["label"].to_numpy()
    clf = build_pipeline("xgb", 0, y).named_steps["clf"]
    assert clf.get_params()["scale_pos_weight"] == pytest.approx(90 / 60)


def test_svm_grid_has_no_useless_degree():
    for g in pipeline_grid("svm"):
        assert ("clf__degree" in g) == (g["clf__kernel"] == ["poly"])


def test_resolve():
    assert resolve("all") == list(MODELS)
    with pytest.raises(KeyError):
        resolve(["lr", "nope"])


def test_embeddings_roundtrip(tmp_path):
    df = to_frame(["ENSG1", "ENSG2"], np.eye(2))
    path = save_embeddings(df, tmp_path / "emb.parquet", {"model": "x"})
    back = load_embeddings(path)
    assert back.shape == (2, 2) and back.dtypes.eq(np.float32).all()
    assert read_metadata(path) == {"model": "x"}
