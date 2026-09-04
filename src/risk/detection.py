"""
Supply Chain Risk Engine + Alert Center
Detects supplier delay risk, stockout/overstock risk, slow-moving/dead
inventory, and demand anomalies. Consolidates everything into a single
prioritized alert feed.
"""

import pandas as pd
import numpy as np


# ---------- Thresholds (would normally be configurable per business) ----------
SLOW_MOVING_DAYS_THRESHOLD = 60     # no meaningful sales velocity in this window
DEAD_STOCK_DAYS_THRESHOLD = 90      # near-zero sales for this long -> dead stock
ANOMALY_Z_THRESHOLD = 3.0           # demand spike/drop beyond 3 std devs
SUPPLIER_RELIABILITY_RISK_THRESHOLD = 0.75


def load_data():
    features = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
    inventory_plan = pd.read_csv("data/processed/inventory_plan.csv")
    return features, inventory_plan


# ---------- 1. Supplier Delay Risk ----------
def detect_supplier_delay_risk(inventory_plan_df):
    """
    Flags product-warehouse pairs sourced from suppliers with low
    reliability AND long lead times -- these are most exposed to delays.
    """
    df = inventory_plan_df.copy()
    df["supplier_delay_risk"] = (
        (df["reliability_score"] < SUPPLIER_RELIABILITY_RISK_THRESHOLD) &
        (df["avg_lead_time_days"] > df["avg_lead_time_days"].median())
    )
    return df


# ---------- 2. Stockout / Overstock Risk ----------
def detect_stock_risk(inventory_plan_df):
    """Reuses stock_status already computed in the inventory optimization engine."""
    df = inventory_plan_df.copy()
    df["stockout_risk"] = df["stock_status"] == "CRITICAL_STOCKOUT_RISK"
    df["overstock_risk"] = df["stock_status"] == "OVERSTOCK"
    return df


# ---------- 3. Slow-Moving / Dead Inventory ----------
def detect_slow_dead_inventory(features_df, inventory_plan_df):
    """
    Uses sales_velocity_30d already present in inventory_plan_df
    (computed upstream in the inventory optimization engine).
    Near-zero velocity over an extended window flags slow-moving or
    dead stock, especially when current stock is still high.
    """
    df = inventory_plan_df.copy()

    df["slow_moving"] = (df["sales_velocity_30d"] < 0.5) & (df["current_stock"] > df["reorder_point"])
    df["dead_inventory"] = (df["sales_velocity_30d"] < 0.1) & (df["current_stock"] > 0)

    return df


# ---------- 4. Demand Anomaly Detection ----------
def detect_demand_anomalies(features_df, z_threshold=ANOMALY_Z_THRESHOLD, lookback_days=30):
    """
    Z-score based anomaly detection: flags days where actual units_sold
    deviates more than `z_threshold` std devs from the trailing rolling
    mean for that product-warehouse -- catches spikes and sudden drops.
    """
    df = features_df.sort_values(["product_id", "warehouse_id", "date"]).copy()
    grp = df.groupby(["product_id", "warehouse_id"])["units_sold"]

    rolling_mean = grp.transform(lambda s: s.shift(1).rolling(lookback_days, min_periods=5).mean())
    rolling_std = grp.transform(lambda s: s.shift(1).rolling(lookback_days, min_periods=5).std())

    df["demand_zscore"] = (df["units_sold"] - rolling_mean) / rolling_std.replace(0, np.nan)
    df["is_demand_anomaly"] = df["demand_zscore"].abs() > z_threshold
    df["anomaly_type"] = np.where(
        df["demand_zscore"] > z_threshold, "SPIKE",
        np.where(df["demand_zscore"] < -z_threshold, "DROP", "NONE")
    )

    anomalies = df[df["is_demand_anomaly"]][
        ["date", "product_id", "warehouse_id", "units_sold", "demand_zscore", "anomaly_type"]
    ].sort_values("date", ascending=False)

    return anomalies


# ---------- 5. Alert Center: consolidate everything ----------
def build_alert_center(risk_df, anomalies_df, top_n_anomalies=50):
    alerts = []

    for _, row in risk_df.iterrows():
        base = {"product_id": row["product_id"], "warehouse_id": row["warehouse_id"]}

        if row.get("stockout_risk"):
            alerts.append({**base, "alert_type": "LOW_INVENTORY", "severity": "HIGH",
                           "message": f"Stock at/below safety stock ({row['current_stock']} units)."})

        if row.get("overstock_risk"):
            alerts.append({**base, "alert_type": "OVERSTOCK", "severity": "MEDIUM",
                           "message": f"Stock ({row['current_stock']}) far exceeds reorder point."})

        if row.get("supplier_delay_risk"):
            alerts.append({**base, "alert_type": "SUPPLIER_DELAY_RISK", "severity": "MEDIUM",
                           "message": f"Low-reliability supplier ({row['reliability_score']:.2f}) "
                                      f"with {row['avg_lead_time_days']}-day lead time."})

        if row.get("dead_inventory"):
            alerts.append({**base, "alert_type": "DEAD_INVENTORY", "severity": "LOW",
                           "message": "Near-zero sales velocity with stock still on hand."})
        elif row.get("slow_moving"):
            alerts.append({**base, "alert_type": "SLOW_MOVING", "severity": "LOW",
                           "message": "Sales velocity below threshold relative to stock held."})

    for _, row in anomalies_df.head(top_n_anomalies).iterrows():
        alerts.append({
            "product_id": row["product_id"], "warehouse_id": row["warehouse_id"],
            "alert_type": f"DEMAND_{row['anomaly_type']}", "severity": "HIGH",
            "message": f"Demand {row['anomaly_type'].lower()} on {row['date'].date()}: "
                       f"{row['units_sold']} units (z={row['demand_zscore']:.2f})."
        })

    alerts_df = pd.DataFrame(alerts)
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    alerts_df["severity_rank"] = alerts_df["severity"].map(severity_order)
    alerts_df = alerts_df.sort_values("severity_rank").drop(columns="severity_rank")

    return alerts_df


def main():
    features_df, inventory_plan_df = load_data()

    risk_df = detect_supplier_delay_risk(inventory_plan_df)
    risk_df = detect_stock_risk(risk_df)
    risk_df = detect_slow_dead_inventory(features_df, risk_df)

    print("Risk flags summary:")
    print(f"  Stockout risk:        {risk_df['stockout_risk'].sum()}")
    print(f"  Overstock risk:       {risk_df['overstock_risk'].sum()}")
    print(f"  Supplier delay risk:  {risk_df['supplier_delay_risk'].sum()}")
    print(f"  Slow-moving:          {risk_df['slow_moving'].sum()}")
    print(f"  Dead inventory:       {risk_df['dead_inventory'].sum()}")

    print("\nDetecting demand anomalies (this scans the full time series)...")
    anomalies_df = detect_demand_anomalies(features_df)
    print(f"  Anomalies found: {len(anomalies_df)}")

    alerts_df = build_alert_center(risk_df, anomalies_df)

    import os
    os.makedirs("data/processed", exist_ok=True)
    risk_df.to_csv("data/processed/risk_flags.csv", index=False)
    anomalies_df.to_csv("data/processed/demand_anomalies.csv", index=False)
    alerts_df.to_csv("data/processed/alerts.csv", index=False)

    print(f"\nGenerated {len(alerts_df)} total alerts.")
    print(alerts_df["alert_type"].value_counts())
    print("\nSaved: risk_flags.csv, demand_anomalies.csv, alerts.csv")


if __name__ == "__main__":
    main()