"""
Data Drift Detection + Automated Retraining
Compares recent data distribution against the training baseline using
Population Stability Index (PSI) -- a standard drift metric. If drift
exceeds a threshold for enough features, triggers automated retraining
and re-registers the model via MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
import subprocess
import sys

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("demand_forecasting_monitoring")

TARGET = "units_sold"
DROP_COLS = ["date", "product_id", "warehouse_id", TARGET]

PSI_DRIFT_THRESHOLD = 0.25       # PSI > 0.25 = significant drift (industry standard cutoff)
DRIFTED_FEATURE_RATIO_TRIGGER = 0.15  # if >15% of features drift, trigger retrain
BASELINE_WINDOW_DAYS = 180
RECENT_WINDOW_DAYS = 30


def load_data(path="data/processed/features.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def compute_psi(baseline, current, bins=10):
    """
    Population Stability Index between two distributions of the same
    numeric feature. PSI < 0.1: no significant shift. 0.1-0.25: moderate
    shift. > 0.25: significant shift (retraining candidate).
    """
    baseline = baseline.dropna()
    current = current.dropna()
    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(baseline, breakpoints)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    baseline_pct = np.where(baseline_counts == 0, 1e-4, baseline_counts / baseline_counts.sum())
    current_pct = np.where(current_counts == 0, 1e-4, current_counts / current_counts.sum())

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def detect_drift(df):
    df = df.sort_values("date")
    max_date = df["date"].max()

    baseline_cutoff = max_date - pd.Timedelta(days=RECENT_WINDOW_DAYS + BASELINE_WINDOW_DAYS)
    recent_cutoff = max_date - pd.Timedelta(days=RECENT_WINDOW_DAYS)

    baseline_df = df[(df["date"] > baseline_cutoff) & (df["date"] <= recent_cutoff)]
    recent_df = df[df["date"] > recent_cutoff]

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    drift_results = []
    for col in numeric_cols:
        psi = compute_psi(baseline_df[col], recent_df[col])
        drift_results.append({
            "feature": col,
            "psi": round(psi, 4),
            "drifted": psi > PSI_DRIFT_THRESHOLD,
        })

    drift_df = pd.DataFrame(drift_results).sort_values("psi", ascending=False)
    return drift_df, baseline_df, recent_df


def trigger_retraining():
    """Calls the existing training pipeline as a subprocess (real automated retraining)."""
    print("\n>>> Drift threshold exceeded. Triggering automated retraining...\n")
    result = subprocess.run([sys.executable, "src/forecasting/train.py"], capture_output=False)
    if result.returncode == 0:
        print("\n>>> Retraining completed successfully. New model version registered in MLflow.")
    else:
        print("\n>>> Retraining failed. Check logs above.")


def main():
    df = load_data()
    drift_df, baseline_df, recent_df = detect_drift(df)

    n_drifted = drift_df["drifted"].sum()
    n_total = len(drift_df)
    drifted_ratio = n_drifted / n_total if n_total > 0 else 0

    print(f"Baseline window: {len(baseline_df):,} rows | Recent window: {len(recent_df):,} rows")
    print(f"\nDrift report ({n_drifted}/{n_total} features drifted, PSI > {PSI_DRIFT_THRESHOLD}):")
    print(drift_df.to_string(index=False))

    import os
    os.makedirs("reports", exist_ok=True)
    drift_df.to_csv("reports/drift_report.csv", index=False)

    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metric("drifted_feature_ratio", drifted_ratio)
        mlflow.log_metric("n_drifted_features", int(n_drifted))
        mlflow.log_artifact("reports/drift_report.csv")

    print(f"\nDrifted feature ratio: {drifted_ratio:.2%} (trigger threshold: {DRIFTED_FEATURE_RATIO_TRIGGER:.0%})")

    if drifted_ratio > DRIFTED_FEATURE_RATIO_TRIGGER:
        trigger_retraining()
    else:
        print("\nNo significant drift detected. Retraining not triggered.")


if __name__ == "__main__":
    main()