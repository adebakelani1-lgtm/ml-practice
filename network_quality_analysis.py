import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
n = 600
network = pd.DataFrame(
    {
        "technology": rng.choice(["4G", "5G"], n, p=[0.6, 0.4]),
        "signal_dbm": rng.normal(-88, 12, n).clip(-120, -55),
        "latency_ms": rng.gamma(3, 8, n).clip(5, 120),
        "packet_loss_pct": rng.exponential(0.8, n).clip(0, 8),
        "bandwidth_mhz": rng.choice([10, 15, 20, 40, 80], n),
        "cell_load_pct": rng.beta(4, 2, n) * 100,
        "handover_count": rng.poisson(2, n),
        "distance_km": rng.gamma(2, 1.2, n).clip(0.1, 8),
    }
)
network["technology_5g"] = (network["technology"] == "5G").astype(int)
network.loc[rng.choice(n, 18, replace=False), "latency_ms"] = np.nan

throughput = (
    network["bandwidth_mhz"] * (1.35 + 0.65 * network["technology_5g"])
    + (network["signal_dbm"] + 100) * 0.8
    - network["latency_ms"].fillna(network["latency_ms"].median()) * 0.35
    - network["packet_loss_pct"] * 4
    - network["cell_load_pct"] * 0.16
    - network["distance_km"] * 1.8
    + rng.normal(0, 5, n)
).clip(1, None)
network["throughput_mbps"] = throughput.round(2)
network["poor_quality"] = (
    (network["throughput_mbps"] < network["throughput_mbps"].median())
    | (network["packet_loss_pct"] > 2.5)
).astype(int)

features = [
    "signal_dbm",
    "latency_ms",
    "packet_loss_pct",
    "bandwidth_mhz",
    "cell_load_pct",
    "handover_count",
    "distance_km",
    "technology_5g",
]
X = network[features]
X_train, X_test, y_train, y_test = train_test_split(
    X, network[["throughput_mbps", "poor_quality"]],
    test_size=0.2,
    random_state=42,
    stratify=network["poor_quality"],
)

prep = SimpleImputer(strategy="median")
X_train_ready = prep.fit_transform(X_train)
X_test_ready = prep.transform(X_test)

throughput_model = LinearRegression().fit(
    X_train_ready, y_train["throughput_mbps"]
)
throughput_prediction = throughput_model.predict(X_test_ready)

quality_model = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000)
).fit(X_train_ready, y_train["poor_quality"])
quality_prediction = quality_model.predict(X_test_ready)
quality_probability = quality_model.predict_proba(X_test_ready)[:, 1]

print("Cellular network quality")
print("Rows:", len(network))
print("Missing latency values repaired:", int(network["latency_ms"].isna().sum()))
print("Poor quality rate:", round(network["poor_quality"].mean(), 3))
print("Throughput MAE:", round(mean_absolute_error(y_test["throughput_mbps"], throughput_prediction), 2), "Mbps")
print("Poor quality accuracy:", round(accuracy_score(y_test["poor_quality"], quality_prediction), 3))
print("Poor quality ROC AUC:", round(roc_auc_score(y_test["poor_quality"], quality_probability), 3))
print(classification_report(y_test["poor_quality"], quality_prediction, digits=3))

importance = pd.Series(throughput_model.coef_, index=features).sort_values(
    key=abs, ascending=False
)
print("Throughput coefficients:")
print(importance.round(2).to_string())
