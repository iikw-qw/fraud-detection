import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def time_split(df, test_fraction=0.25):
    """Train on the past, test on the future: a random split would leak patterns across time."""
    cut = int(len(df) * (1 - test_fraction))
    train, test = df.iloc[:cut], df.iloc[cut:]
    cols = NUMERIC + CATEGORICAL
    return train[cols], test[cols], train["is_fraud"], test["is_fraud"]


def random_undersample(x, y, ratio=10, seed=0):
    """Keep all fraud and ratio times as many legitimate rows."""
    rng = np.random.default_rng(seed)
    pos = np.where(y.to_numpy() == 1)[0]
    neg = np.where(y.to_numpy() == 0)[0]
    keep_neg = rng.choice(neg, size=min(len(neg), ratio * len(pos)), replace=False)
    idx = np.sort(np.r_[pos, keep_neg])
    return x.iloc[idx], y.iloc[idx]


def make_pipeline(estimator, scale=True):
    prep = ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("prep", prep), ("clf", estimator)])


def strategies(seed=0):
    return {
        "logistic_plain": (make_pipeline(LogisticRegression(max_iter=1000)), None),
        "logistic_balanced": (make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")), None),
        "forest_balanced": (
            make_pipeline(
                RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
                scale=False,
            ),
            None,
        ),
        "boosting_undersampled": (
            make_pipeline(HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.08, random_state=seed), scale=False),
            10,
        ),
        "boosting_weighted": (
            make_pipeline(HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.08, class_weight="balanced", random_state=seed), scale=False),
            None,
        ),
    }


def fit_all(x_tr, y_tr, seed=0):
    fitted = {}
    for name, (pipe, ratio) in strategies(seed).items():
        if ratio:
            xs, ys = random_undersample(x_tr, y_tr, ratio, seed)
            fitted[name] = pipe.fit(xs, ys)
        else:
            fitted[name] = pipe.fit(x_tr, y_tr)
    return fitted


def precision_at_k(y, score, k):
    order = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(y)[order].mean())


def recall_at_k(y, score, k):
    order = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(y)[order].sum() / np.asarray(y).sum())


def evaluate(y, score, alert_budget=0.005):
    k = max(1, int(len(y) * alert_budget))
    return {
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "precision_at_budget": precision_at_k(y, score, k),
        "recall_at_budget": recall_at_k(y, score, k),
    }


def threshold_for_precision(y, score, target=0.8):
    prec, rec, thr = precision_recall_curve(y, score)
    ok = np.where(prec[:-1] >= target)[0]
    if len(ok) == 0:
        return None
    i = ok[0]
    return {"threshold": float(thr[i]), "precision": float(prec[i]), "recall": float(rec[i])}
