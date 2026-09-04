"""
Feature Engineering Pipeline
Transforms raw sales/product/supplier/holiday/promotion/weather/event
data into model-ready features for the demand forecasting engine.
"""

import pandas as pd
import numpy as np


def load_raw_data(raw_dir="data/raw"):
    sales = pd.read_csv(f"{raw_dir}/sales_history.csv", parse_dates=["date"])
    products = pd.read_csv(f"{raw_dir}/products.csv")
    warehouses = pd.read_csv(f"{raw_dir}/warehouses.csv")
    suppliers = pd.read_csv(f"{raw_dir}/suppliers.csv")
    product_supplier = pd.read_csv(f"{raw_dir}/product_supplier_map.csv")
    holidays = pd.read_csv(f"{raw_dir}/holidays.csv", parse_dates=["date"])
    promotions = pd.read_csv(
        f"{raw_dir}/promotions.csv", parse_dates=["promo_start", "promo_end"]
    )
    weather = pd.read_csv(f"{raw_dir}/weather.csv", parse_dates=["date"])
    events = pd.read_csv(f"{raw_dir}/regional_events.csv", parse_dates=["event_date"])
    return sales, products, warehouses, suppliers, product_supplier, holidays, promotions, weather, events


def add_calendar_features(df):
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def add_holiday_features(df, holidays_df):
    holidays_df = holidays_df.rename(columns={"date": "holiday_date"})
    df = df.merge(
        holidays_df[["holiday_date", "demand_multiplier"]],
        left_on="date", right_on="holiday_date", how="left"
    )
    df["is_holiday"] = df["holiday_date"].notna().astype(int)
    df["holiday_demand_multiplier"] = df["demand_multiplier"].fillna(1.0)
    df = df.drop(columns=["holiday_date", "demand_multiplier"])

    df = df.sort_values(["product_id", "warehouse_id", "date"])
    df["holiday_within_7d"] = (
        df.groupby(["product_id", "warehouse_id"])["is_holiday"]
        .transform(lambda s: s[::-1].rolling(7, min_periods=1).max()[::-1])
    )
    return df


def add_promotion_features(df, promotions_df):
    df["promo_active"] = 0
    df["discount_pct"] = 0.0

    for pid, promo_group in promotions_df.groupby("product_id"):
        mask_pid = df["product_id"] == pid
        product_dates = df.loc[mask_pid, "date"]

        active = np.zeros(len(product_dates), dtype=int)
        discount = np.zeros(len(product_dates), dtype=float)

        for _, promo in promo_group.iterrows():
            in_range = (product_dates >= promo["promo_start"]) & (product_dates <= promo["promo_end"])
            active |= in_range.to_numpy()
            discount = np.where(in_range.to_numpy(), promo["discount_pct"], discount)

        df.loc[mask_pid, "promo_active"] = active
        df.loc[mask_pid, "discount_pct"] = discount

    return df


def add_sales_velocity_and_volatility(df, windows=(7, 14, 30)):
    df = df.sort_values(["product_id", "warehouse_id", "date"])
    grp = df.groupby(["product_id", "warehouse_id"])["units_sold"]

    for w in windows:
        df[f"sales_velocity_{w}d"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        rolling_std = grp.transform(lambda s: s.shift(1).rolling(w, min_periods=2).std())
        rolling_mean = df[f"sales_velocity_{w}d"]
        df[f"demand_volatility_{w}d"] = (rolling_std / rolling_mean.replace(0, np.nan)).fillna(0)

    df["units_sold_lag_1"] = grp.transform(lambda s: s.shift(1))
    df["units_sold_lag_7"] = grp.transform(lambda s: s.shift(7))

    return df


def add_seasonality_index(df):
    overall_mean = df.groupby(["product_id", "warehouse_id"])["units_sold"].transform("mean")
    dow_mean = df.groupby(["product_id", "warehouse_id", "day_of_week"])["units_sold"].transform("mean")
    month_mean = df.groupby(["product_id", "warehouse_id", "month"])["units_sold"].transform("mean")

    df["dow_seasonality_index"] = (dow_mean / overall_mean.replace(0, np.nan)).fillna(1.0)
    df["month_seasonality_index"] = (month_mean / overall_mean.replace(0, np.nan)).fillna(1.0)
    return df


def add_product_supplier_features(df, products_df, product_supplier_df, suppliers_df):
    primary_supplier = (
        product_supplier_df.merge(suppliers_df, on="supplier_id", how="left")
        .sort_values("reliability_score", ascending=False)
        .drop_duplicates(subset="product_id", keep="first")
    )[["product_id", "avg_lead_time_days", "reliability_score"]]

    df = df.merge(primary_supplier, on="product_id", how="left")
    df = df.merge(
        products_df[["product_id", "category", "lifecycle_stage", "unit_cost", "unit_price"]],
        on="product_id", how="left"
    )

    df["lifecycle_stage"] = df["lifecycle_stage"].astype("category").cat.codes
    df["category"] = df["category"].astype("category").cat.codes
    return df


def add_regional_pattern_features(df, warehouses_df):
    df = df.merge(warehouses_df[["warehouse_id", "region"]], on="warehouse_id", how="left")

    region_share = (
        df.groupby(["region", "date"])["units_sold"].transform("sum") /
        df.groupby("date")["units_sold"].transform("sum").replace(0, np.nan)
    )
    df["regional_demand_share"] = region_share.fillna(0)
    return df


def add_weather_and_event_features(df, weather_df, events_df):
    """
    Assumes df already has a 'region' column (added by add_regional_pattern_features,
    called before this). Merges weather by (date, warehouse_id) and regional events
    by (region, date).
    """
    df = df.merge(weather_df, on=["date", "warehouse_id"], how="left")

    events_df = events_df.rename(columns={"event_date": "date"})
    df = df.merge(
        events_df[["region", "date", "demand_impact_multiplier"]],
        on=["region", "date"], how="left"
    )
    df["is_regional_event"] = df["demand_impact_multiplier"].notna().astype(int)
    df["regional_event_multiplier"] = df["demand_impact_multiplier"].fillna(1.0)
    df = df.drop(columns=["demand_impact_multiplier"])

    # now encode region to numeric (after weather/event merges are done with it)
    df["region"] = df["region"].astype("category").cat.codes

    return df


def build_features(raw_dir="data/raw", output_path="data/processed/features.csv"):
    sales, products, warehouses, suppliers, product_supplier, holidays, promotions, weather, events = load_raw_data(raw_dir)

    print(f"Loaded {len(sales):,} raw sales rows.")

    df = add_calendar_features(sales)
    print("Added calendar features.")

    df = add_holiday_features(df, holidays)
    print("Added holiday features.")

    df = add_promotion_features(df, promotions)
    print("Added promotion features.")

    df = add_sales_velocity_and_volatility(df)
    print("Added sales velocity & volatility features.")

    df = add_seasonality_index(df)
    print("Added seasonality index features.")

    df = add_product_supplier_features(df, products, product_supplier, suppliers)
    print("Added product/supplier features.")

    df = add_regional_pattern_features(df, warehouses)
    print("Added regional pattern features.")

    df = add_weather_and_event_features(df, weather, events)
    print("Added weather & regional event features.")

    df = df.dropna(subset=["units_sold_lag_1"])

    import os
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nFeature engineering complete. {len(df):,} rows x {df.shape[1]} columns.")
    print(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    build_features()