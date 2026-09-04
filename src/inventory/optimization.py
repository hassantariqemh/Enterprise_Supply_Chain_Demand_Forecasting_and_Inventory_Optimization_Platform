"""
Inventory Optimization Engine
Computes reorder points, safety stock, EOQ, warehouse allocation health,
and inter-warehouse transfer recommendations using forecasted demand
(sales velocity + volatility) and supplier lead time data.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

# Business assumptions (would normally come from finance/procurement config)
SERVICE_LEVEL = 0.95          # 95% service level -> Z score below
ORDERING_COST_PER_ORDER = 50  # $ per purchase order placed
HOLDING_COST_RATE = 0.20      # 20% of unit cost per year (storage, capital, etc.)

Z_SCORE = norm.ppf(SERVICE_LEVEL)  # ~1.645 for 95%


def load_data(features_path="data/processed/features.csv",
              inventory_path="data/raw/inventory_snapshot.csv"):
    features = pd.read_csv(features_path, parse_dates=["date"])
    inventory = pd.read_csv(inventory_path)
    return features, inventory


def get_latest_demand_stats(features_df):
    """
    For each product-warehouse, take the most recent row's rolling
    sales velocity (7d/30d) and volatility as the current demand signal.
    """
    latest = (
        features_df.sort_values("date")
        .groupby(["product_id", "warehouse_id"])
        .tail(1)
        .reset_index(drop=True)
    )

    stats = latest[[
        "product_id", "warehouse_id",
        "sales_velocity_7d", "sales_velocity_30d",
        "demand_volatility_30d",
        "avg_lead_time_days", "reliability_score",
        "unit_cost",
    ]].copy()

    stats["avg_daily_demand"] = stats["sales_velocity_30d"].fillna(stats["sales_velocity_7d"])
    stats["demand_std"] = (stats["avg_daily_demand"] * stats["demand_volatility_30d"]).fillna(0)
    return stats


def calc_safety_stock(row):
    """
    Safety Stock = Z * sqrt(lead_time) * demand_std_dev
    Standard formula accounting for demand variability during lead time.
    Reliability score reduces required safety stock for trustworthy suppliers,
    and increases it for unreliable ones (lead time variability proxy).
    """
    lead_time = row["avg_lead_time_days"]
    demand_std = row["demand_std"]
    reliability_adj = 1 + (1 - row["reliability_score"])  # unreliable supplier -> more buffer

    return Z_SCORE * np.sqrt(lead_time) * demand_std * reliability_adj


def calc_reorder_point(row):
    """Reorder Point = (avg daily demand * lead time) + safety stock."""
    return (row["avg_daily_demand"] * row["avg_lead_time_days"]) + row["safety_stock"]


def calc_eoq(row):
    """
    Economic Order Quantity = sqrt( (2 * D * S) / H )
    D = annual demand, S = ordering cost per order, H = holding cost per unit per year
    """
    annual_demand = row["avg_daily_demand"] * 365
    holding_cost = row["unit_cost"] * HOLDING_COST_RATE
    if holding_cost <= 0 or annual_demand <= 0:
        return 0
    return np.sqrt((2 * annual_demand * ORDERING_COST_PER_ORDER) / holding_cost)


def classify_stock_status(row):
    """Flag current stock position relative to reorder point and safety stock."""
    if row["current_stock"] <= row["safety_stock"]:
        return "CRITICAL_STOCKOUT_RISK"
    elif row["current_stock"] <= row["reorder_point"]:
        return "REORDER_NEEDED"
    elif row["current_stock"] > row["reorder_point"] * 3:
        return "OVERSTOCK"
    else:
        return "HEALTHY"


def build_inventory_plan(features_df, inventory_df):
    stats = get_latest_demand_stats(features_df)
    plan = stats.merge(inventory_df, on=["product_id", "warehouse_id"], how="left")

    plan["safety_stock"] = plan.apply(calc_safety_stock, axis=1).round().clip(lower=0)
    plan["reorder_point"] = plan.apply(calc_reorder_point, axis=1).round().clip(lower=0)
    plan["eoq"] = plan.apply(calc_eoq, axis=1).round().clip(lower=0)
    plan["stock_status"] = plan.apply(classify_stock_status, axis=1)

    from scipy.stats import norm
    lead_time_demand_std = plan["demand_std"] * np.sqrt(plan["avg_lead_time_days"].clip(lower=1))
    lead_time_demand_mean = plan["avg_daily_demand"] * plan["avg_lead_time_days"]
    z = (plan["current_stock"] - lead_time_demand_mean) / lead_time_demand_std.replace(0, np.nan)
    plan["stockout_probability"] = (1 - norm.cdf(z.fillna(5))).round(4)

    plan["recommended_order_qty"] = np.where(
        plan["stock_status"].isin(["REORDER_NEEDED", "CRITICAL_STOCKOUT_RISK"]),
        plan["eoq"],
        0
    )

    return plan


def recommend_transfers(plan_df, min_surplus_ratio=2.0):
    """
    Inter-warehouse transfer suggestions: if one warehouse is OVERSTOCK
    on a product and another warehouse (same product) is CRITICAL/REORDER,
    recommend transferring stock instead of placing a new supplier order.
    """
    transfers = []

    for pid, group in plan_df.groupby("product_id"):
        surplus = group[group["stock_status"] == "OVERSTOCK"]
        deficit = group[group["stock_status"].isin(["CRITICAL_STOCKOUT_RISK", "REORDER_NEEDED"])]

        if surplus.empty or deficit.empty:
            continue

        for _, s_row in surplus.iterrows():
            transferable = s_row["current_stock"] - s_row["reorder_point"]
            if transferable <= 0:
                continue

            for _, d_row in deficit.iterrows():
                needed = d_row["reorder_point"] - d_row["current_stock"]
                if needed <= 0:
                    continue

                transfer_qty = min(transferable, needed)
                if transfer_qty > 0:
                    transfers.append({
                        "product_id": pid,
                        "from_warehouse": s_row["warehouse_id"],
                        "to_warehouse": d_row["warehouse_id"],
                        "transfer_qty": round(transfer_qty),
                    })
                    transferable -= transfer_qty
                    if transferable <= 0:
                        break

    return pd.DataFrame(transfers)


def main():
    features_df, inventory_df = load_data()
    print(f"Loaded {len(features_df):,} feature rows, {len(inventory_df):,} inventory snapshot rows.")

    plan = build_inventory_plan(features_df, inventory_df)

    import os
    os.makedirs("data/processed", exist_ok=True)
    plan.to_csv("data/processed/inventory_plan.csv", index=False)

    print("\nStock status breakdown:")
    print(plan["stock_status"].value_counts())

    transfers = recommend_transfers(plan)
    transfers.to_csv("data/processed/transfer_recommendations.csv", index=False)
    print(f"\nRecommended {len(transfers)} inter-warehouse transfers.")
    print(f"Saved: data/processed/inventory_plan.csv, data/processed/transfer_recommendations.csv")


if __name__ == "__main__":
    main()