# SignalGuard — Responsible Banking Risk Intelligence

SignalGuard is a portfolio-ready AI project for a Master’s student targeting RBC Amplify roles in Data Engineering, Development, and Business Analysis.

It turns synthetic transaction events into explainable risk signals. The project demonstrates:

- reproducible data generation and validation;
- feature engineering for transaction behaviour;
- an unsupervised anomaly-detection model;
- transparent reason codes instead of black-box decisions;
- an analyst-facing API contract and dashboard prototype.

> This project uses synthetic data only. It is an educational prototype, not a banking or credit decision system.

## Run it

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m signalguard.pipeline

The pipeline writes a scored dataset to artifacts/scored_transactions.csv and prints evaluation and sample risk explanations.

Open dashboard/index.html to see the analyst dashboard concept.

## How to talk about it

“I built an explainable risk-intelligence pipeline using synthetic transaction data. I designed the feature layer, used Isolation Forest for anomaly detection, added human-readable reason codes, and separated model scoring from analyst workflow. My focus was making the system useful and auditable—not just maximizing a model metric.”

## Next production steps

Add a streaming ingestion layer, model monitoring, drift checks, role-based access controls, and a human review queue before considering any real financial use.
