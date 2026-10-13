import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    df = data.generate_transactions()
    x_tr, x_te, y_tr, y_te = core.time_split(df)
    fitted = core.fit_all(x_tr, y_tr)

    scores = {name: model.predict_proba(x_te)[:, 1] for name, model in fitted.items()}
    table = pd.DataFrame({name: core.evaluate(y_te, s) for name, s in scores.items()}).T.sort_values("pr_auc", ascending=False)
    best = table.index[0]
    at_precision = core.threshold_for_precision(y_te, scores[best], 0.8)

    plt.figure(figsize=(6.5, 4.5))
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y_te, s)
        plt.plot(rec, prec, label=name, linewidth=2 if name == best else 1, color="black" if name == best else None)
    plt.axhline(y_te.mean(), linestyle="--", color="grey")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall on the last 25% of the period")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "pr_curves.png", dpi=130)

    table.round(4).to_csv(OUT / "metrics.csv")
    summary = {
        "transactions": int(len(df)),
        "fraud_rate": round(float(df["is_fraud"].mean()), 4),
        "test_fraud_cases": int(y_te.sum()),
        "metrics": {k: {m: round(float(v), 4) for m, v in row.items()} for k, row in table.iterrows()},
        "best_by_pr_auc": best,
        "operating_point_at_80pct_precision": at_precision,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
