"""
Exports the latest model performance metrics from MLflow into a simple
JSON file the dashboard/API can read, without needing an MLflow client
call on every request.
"""

import mlflow
import json
import os

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")


def get_latest_run_metrics(experiment_name, run_name_prefix):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name_prefix}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return runs[0].data.metrics


def main():
    xgb_metrics = get_latest_run_metrics("demand_forecasting", "xgboost_tscv")

    result = {
        "forecast_accuracy": {
            "model": "XGBoost",
            "avg_mae": xgb_metrics.get("avg_mae") if xgb_metrics else None,
            "avg_rmse": xgb_metrics.get("avg_rmse") if xgb_metrics else None,
            "avg_mape": xgb_metrics.get("avg_mape") if xgb_metrics else None,
            "accuracy_pct": round((1 - xgb_metrics.get("avg_mape", 0)) * 100, 2) if xgb_metrics else None,
        }
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/model_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\nSaved: reports/model_metrics.json")


if __name__ == "__main__":
    main()