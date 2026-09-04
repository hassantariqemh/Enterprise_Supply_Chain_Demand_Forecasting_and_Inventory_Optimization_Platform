"""
Loads all processed pipeline outputs (CSV) into PostgreSQL tables.
Run this after any pipeline stage regenerates its outputs.
"""

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import json

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))


def load_table(csv_path, table_name, columns=None):
    df = pd.read_csv(csv_path)
    if columns:
        df = df[columns]
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"Loaded {len(df):,} rows into '{table_name}' from {csv_path}")


def load_model_metrics():
    try:
        with open("reports/model_metrics.json") as f:
            metrics = json.load(f)["forecast_accuracy"]
        df = pd.DataFrame([{
            "model_name": metrics["model"],
            "avg_mae": metrics["avg_mae"],
            "avg_rmse": metrics["avg_rmse"],
            "avg_mape": metrics["avg_mape"],
            "accuracy_pct": metrics["accuracy_pct"],
        }])
        df.to_sql("model_metrics", engine, if_exists="append", index=False)
        print(f"Loaded model metrics into 'model_metrics'")
    except FileNotFoundError:
        print("model_metrics.json not found, skipping.")


def main():
    load_table("data/processed/inventory_plan.csv", "inventory_plan", columns=[
        "product_id", "warehouse_id", "avg_daily_demand", "demand_std",
        "avg_lead_time_days", "reliability_score", "unit_cost", "current_stock",
        "safety_stock", "reorder_point", "eoq", "stock_status",
        "recommended_order_qty", "stockout_probability"
    ])
    load_table("data/processed/alerts.csv", "alerts")
    load_table("data/processed/transfer_recommendations.csv", "transfer_recommendations")
    load_table("data/processed/warehouse_allocation.csv", "warehouse_allocation")
    load_table("data/processed/warehouse_utilization.csv", "warehouse_utilization", columns=[
        "warehouse_id", "total_stock", "capacity_units", "utilization_pct", "capacity_status"
    ])
    load_table("data/processed/forecasts/daily_forecast.csv", "daily_forecast")
    load_model_metrics()

    print("\nAll data loaded into PostgreSQL.")


if __name__ == "__main__":
    main()