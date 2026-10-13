import numpy as np
import pandas as pd

NUMERIC = ["amount", "hour", "distance_from_home_km", "txn_last_hour", "card_age_days", "amount_vs_avg"]
CATEGORICAL = ["merchant_category", "channel"]


def generate_transactions(n=120000, fraud_rate=0.004, seed=71):
    """Card transactions ordered in time, with rare fraud that looks different in a few ways."""
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < fraud_rate
    n_f = fraud.sum()

    amount = rng.lognormal(3.4, 0.9, n)
    amount[fraud] = rng.lognormal(4.6, 1.0, n_f)
    hour = rng.choice(24, n, p=_hour_profile())
    hour[fraud] = rng.choice(24, n_f, p=_hour_profile(night=True))
    distance = rng.exponential(8, n)
    distance[fraud] = rng.exponential(60, n_f)
    velocity = rng.poisson(0.4, n)
    velocity[fraud] = rng.poisson(2.2, n_f)
    card_age = rng.integers(10, 3000, n)
    card_age[fraud] = rng.integers(5, 900, n_f)
    avg_amount = rng.lognormal(3.4, 0.4, n)
    amount_vs_avg = amount / avg_amount

    merchant = rng.choice(["grocery", "fuel", "restaurant", "online", "electronics", "travel"], n, p=[0.3, 0.15, 0.2, 0.2, 0.08, 0.07])
    merchant[fraud] = rng.choice(["grocery", "fuel", "restaurant", "online", "electronics", "travel"], n_f, p=[0.04, 0.05, 0.06, 0.45, 0.3, 0.1])
    channel = rng.choice(["pos", "online", "atm"], n, p=[0.6, 0.3, 0.1])
    channel[fraud] = rng.choice(["pos", "online", "atm"], n_f, p=[0.15, 0.75, 0.1])

    df = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 180 * 24 * 3600, n)), unit="s"),
            "amount": amount.round(2),
            "hour": hour,
            "distance_from_home_km": distance.round(1),
            "txn_last_hour": velocity,
            "card_age_days": card_age,
            "amount_vs_avg": amount_vs_avg.round(3),
            "merchant_category": merchant,
            "channel": channel,
            "is_fraud": fraud.astype(int),
        }
    )
    return df


def _hour_profile(night=False):
    p = np.ones(24)
    p[8:22] = 3.0
    if night:
        p = np.ones(24)
        p[0:6] = 3.0
        p[22:24] = 2.0
    return p / p.sum()
