"""
Multi-Granularity Demand Forecasting
Generates daily predictions using the trained model, then aggregates
them into weekly, monthly, regional, and category-level demand views.
Also detects seasonal peaks. This is standard practice: forecast at
the finest grain (daily/product/warehouse), then roll up -- more
accurate than forecasting each granularity independently.
"""

import pandas as pd
import numpy as np
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

TARGET = "units_sold"
DROP_COLS = ["date", "product_id", "warehouse_id", TARGET]


def load_model_and_data():
    model = mlflow.xgboost.load_model("models:/demand_forecast_xgboost/latest")
    df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    return model, df, feature_cols


def generate_daily_predictions(model, df, feature_cols):
    """Predict demand for every row using the trained model."""
    X = df[feature_cols]
    preds = model.predict(X)
    df = df.copy()
    df["predicted_demand"] = np.clip(preds, 0, None)
    return df


def aggregate_weekly(df):
    weekly = (
        df.groupby([pd.Grouper(key="date", freq="W"), "product_id", "warehouse_id"])
        .agg(actual_demand=("units_sold", "sum"), predicted_demand=("predicted_demand", "sum"))
        .reset_index()
        .rename(columns={"date": "week_ending"})
    )
    return weekly


def aggregate_monthly(df):
    monthly = (
        df.groupby([pd.Grouper(key="date", freq="ME"), "product_id", "warehouse_id"])
        .agg(actual_demand=("units_sold", "sum"), predicted_demand=("predicted_demand", "sum"))
        .reset_index()
        .rename(columns={"date": "month"})
    )
    return monthly


def aggregate_regional(df):
    """Region column is label-encoded in features -- kept as-is (numeric code)."""
    regional = (
        df.groupby([pd.Grouper(key="date", freq="ME"), "region"])
        .agg(actual_demand=("units_sold", "sum"), predicted_demand=("predicted_demand", "sum"))
        .reset_index()
        .rename(columns={"date": "month"})
    )
    return regional


def aggregate_category(df):
    """Category column is label-encoded in features -- kept as-is (numeric code)."""
    category = (
        df.groupby([pd.Grouper(key="date", freq="ME"), "category"])
        .agg(actual_demand=("units_sold", "sum"), predicted_demand=("predicted_demand", "sum"))
        .reset_index()
        .rename(columns={"date": "month"})
    )
    return category


def detect_seasonal_peaks(df, top_n=10):
    """Identify the highest-demand days system-wide -- seasonal peak detection."""
    daily_total = df.groupby("date")["units_sold"].sum().reset_index()
    daily_total = daily_total.sort_values("units_sold", ascending=False)
    return daily_total.head(top_n)


def main():
    model, df, feature_cols = load_model_and_data()
    print(f"Generating predictions for {len(df):,} rows...")

    df = generate_daily_predictions(model, df, feature_cols)

    weekly = aggregate_weekly(df)
    monthly = aggregate_monthly(df)
    regional = aggregate_regional(df)
    category = aggregate_category(df)
    peaks = detect_seasonal_peaks(df)

    import os
    os.makedirs("data/processed/forecasts", exist_ok=True)

    df[["date", "product_id", "warehouse_id", "units_sold", "predicted_demand"]].to_csv(
        "data/processed/forecasts/daily_forecast.csv", index=False
    )
    weekly.to_csv("data/processed/forecasts/weekly_forecast.csv", index=False)
    monthly.to_csv("data/processed/forecasts/monthly_forecast.csv", index=False)
    regional.to_csv("data/processed/forecasts/regional_forecast.csv", index=False)
    category.to_csv("data/processed/forecasts/category_forecast.csv", index=False)
    peaks.to_csv("data/processed/forecasts/seasonal_peaks.csv", index=False)

    print("\nSaved: daily_forecast.csv, weekly_forecast.csv, monthly_forecast.csv,")
    print("       regional_forecast.csv, category_forecast.csv, seasonal_peaks.csv")

    print(f"\nWeekly aggregation: {len(weekly):,} rows")
    print(f"Monthly aggregation: {len(monthly):,} rows")
    print(f"Regional aggregation: {len(regional):,} rows")
    print(f"Category aggregation: {len(category):,} rows")

    print("\nTop 5 seasonal peak days (system-wide demand):")
    print(peaks.head(5).to_string(index=False))


if __name__ == "__main__":
    main()