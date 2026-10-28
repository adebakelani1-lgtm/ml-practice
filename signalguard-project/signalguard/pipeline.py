from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def generate_transactions(n: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Create synthetic, privacy-safe transaction events with planted anomalies."""
    rng = np.random.default_rng(seed)
    customers = rng.integers(10001, 10201, n)
    amounts = np.round(rng.lognormal(mean=3.6, sigma=0.75, size=n), 2)
    hours = rng.integers(0, 24, n)
    countries = rng.choice(["CA", "US", "GB", "NG", "SG"], n, p=[.72, .14, .06, .04, .04])
    channels = rng.choice(["mobile", "web", "branch", "atm"], n, p=[.46, .32, .08, .14])
    labels = np.zeros(n, dtype=int)

    suspicious = rng.choice(n, size=70, replace=False)
    amounts[suspicious] *= rng.uniform(5, 14, len(suspicious))
    hours[suspicious] = rng.choice([1, 2, 3, 4], len(suspicious))
    countries[suspicious] = rng.choice(["NG", "SG"], len(suspicious))
    labels[suspicious] = 1

    return pd.DataFrame({
        "transaction_id": [f"TX-{i:06d}" for i in range(n)],
        "customer_id": customers,
        "amount": amounts,
        "hour": hours,
        "country": countries,
        "channel": channels,
        "is_known_anomaly": labels,
    })


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build interpretable numeric features and return reason-code inputs."""
    work = df.copy()
    work["is_overnight"] = work["hour"].between(0, 5).astype(int)
    work["is_cross_border"] = (work["country"] != "CA").astype(int)
    work["amount_log"] = np.log1p(work["amount"])
    work["amount_vs_typical"] = work["amount"] / work["amount"].median()
    features = ["amount_log", "amount_vs_typical", "is_overnight", "is_cross_border"]
    return work, features


def add_reason_codes(scored: pd.DataFrame) -> pd.DataFrame:
    def explain(row: pd.Series) -> str:
        reasons = []
        if row["amount_vs_typical"] >= 4:
            reasons.append("amount far above typical")
        if row["is_overnight"]:
            reasons.append("overnight activity")
        if row["is_cross_border"]:
            reasons.append("cross-border activity")
        return "; ".join(reasons) or "unusual combination of signals"

    scored["reason_codes"] = scored.apply(explain, axis=1)
    scored["risk_level"] = pd.cut(
        scored["risk_score"], bins=[-np.inf, .45, .72, np.inf], labels=["low", "review", "high"]
    ).astype(str)
    return scored


def run(seed: int = 42) -> pd.DataFrame:
    raw = generate_transactions(seed=seed)
    features_df, feature_names = engineer_features(raw)
    train, _ = train_test_split(
        features_df, test_size=.25, random_state=seed, stratify=features_df["is_known_anomaly"]
    )
    model = IsolationForest(n_estimators=160, contamination=.03, random_state=seed)
    model.fit(train[feature_names])

    raw_score = -model.decision_function(features_df[feature_names])
    risk_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
    scored = features_df.copy()
    scored["risk_score"] = np.round(risk_score, 4)
    scored["model_flag"] = (model.predict(scored[feature_names]) == -1).astype(int)
    scored = add_reason_codes(scored)

    auc = roc_auc_score(scored["is_known_anomaly"], scored["risk_score"])
    print(f"Rows processed: {len(scored):,}")
    print(f"Known anomalies: {scored['is_known_anomaly'].sum():,}")
    print(f"Risk ranking ROC-AUC: {auc:.3f}")
    print("\nModel classification report:")
    print(classification_report(scored["is_known_anomaly"], scored["model_flag"], zero_division=0))
    print("Top explainable review candidates:")
    print(scored.nlargest(5, "risk_score")[
        ["transaction_id", "amount", "country", "hour", "risk_score", "risk_level", "reason_codes"]
    ].to_string(index=False))

    ARTIFACTS.mkdir(exist_ok=True)
    scored.to_csv(ARTIFACTS / "scored_transactions.csv", index=False)
    return scored


if __name__ == "__main__":
    run()
