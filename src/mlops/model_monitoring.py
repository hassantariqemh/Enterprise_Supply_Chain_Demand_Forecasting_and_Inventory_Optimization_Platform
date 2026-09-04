"""
Model Performance Monitoring
Tracks the deployed model's actual prediction accuracy over recent
time windows (predicted vs actual demand), separate from feature
drift detection. Flags performance degradation and logs to MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("demand_forecasting_monitoring")

MAE_DEGRADATION_THRESHOLD = 1.15  # 15% worse than baseline triggers a flag
MONITORING_WINDOW_DAYS = 14


def load_forecast_log(path="data/processed/forecasts/daily_forecast.csv"):
    return pd.read_csv(path, parse_dates=["date"])


def load_baseline_mae(path="reports/model_metrics.json"):
    import json
    with open(path) as f:
        return json.load(f)["forecast_accuracy"]["avg_mae"]


def compute_recent_performance(df, window_days=MONITORING_WINDOW_DAYS):
    """Recent-window prediction accuracy, and a week-over-week trend."""
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=window_days)
    recent = df[df["date"] > cutoff]

    mae = mean_absolute_error(recent["units_sold"], recent["predicted_demand"])
    mape = mean_absolute_percentage_error(
        np.where(recent["units_sold"] == 0, 1e-3, recent["units_sold"]),
        recent["predicted_demand"]
    )

    # per-day trend to catch a worsening trajectory, not just a snapshot
    daily = recent.groupby("date").apply(
        lambda g: mean_absolute_error(g["units_sold"], g["predicted_demand"]), include_groups=False
    ).reset_index(name="daily_mae")

    return {"window_days": window_days, "recent_mae": mae, "recent_mape": mape}, daily


def main():
    df = load_forecast_log()
    baseline_mae = load_baseline_mae()

    recent_metrics, daily_trend = compute_recent_performance(df)

    degradation_ratio = recent_metrics["recent_mae"] / baseline_mae
    is_degraded = degradation_ratio > MAE_DEGRADATION_THRESHOLD

    print(f"Baseline training MAE: {baseline_mae:.3f}")
    print(f"Recent {MONITORING_WINDOW_DAYS}-day MAE: {recent_metrics['recent_mae']:.3f}")
    print(f"Recent MAPE: {recent_metrics['recent_mape']:.3f}")
    print(f"Degradation ratio: {degradation_ratio:.2f}x (flag threshold: {MAE_DEGRADATION_THRESHOLD}x)")

    with mlflow.start_run(run_name="model_performance_monitoring"):
        mlflow.log_metric("baseline_mae", baseline_mae)
        mlflow.log_metric("recent_mae", recent_metrics["recent_mae"])
        mlflow.log_metric("recent_mape", recent_metrics["recent_mape"])
        mlflow.log_metric("degradation_ratio", degradation_ratio)
        mlflow.log_metric("performance_degraded", int(is_degraded))

    import os
    os.makedirs("reports", exist_ok=True)
    daily_trend.to_csv("reports/model_performance_trend.csv", index=False)

    if is_degraded:
        print("\n>>> Model performance has degraded beyond threshold. Retraining recommended.")
    else:
        print("\nModel performance within acceptable range. No action needed.")

    print("Saved: reports/model_performance_trend.csv")


if __name__ == "__main__":
    main()