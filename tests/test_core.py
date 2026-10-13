import numpy as np
import pytest

from src import core, data


@pytest.fixture(scope="module")
def split():
    df = data.generate_transactions(n=40000, fraud_rate=0.01, seed=2)
    return df, core.time_split(df)


def test_fraud_is_rare(split):
    df, _ = split
    assert 0.005 < df["is_fraud"].mean() < 0.02


def test_split_respects_time_order(split):
    df, _ = split
    cut = int(len(df) * 0.75)
    assert df["timestamp"].iloc[cut - 1] <= df["timestamp"].iloc[cut]


def test_undersampling_keeps_all_fraud(split):
    _, (x_tr, _, y_tr, _) = split
    xs, ys = core.random_undersample(x_tr, y_tr, ratio=5)
    assert ys.sum() == y_tr.sum()
    assert (ys == 0).sum() == 5 * ys.sum()


def test_precision_and_recall_at_k():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert core.precision_at_k(y, s, 2) == 0.5
    assert core.recall_at_k(y, s, 3) == 1.0


def test_models_find_fraud_better_than_chance(split):
    _, (x_tr, x_te, y_tr, y_te) = split
    fitted = core.fit_all(x_tr, y_tr)
    pr = core.evaluate(y_te, fitted["boosting_weighted"].predict_proba(x_te)[:, 1])["pr_auc"]
    assert pr > 5 * y_te.mean()
