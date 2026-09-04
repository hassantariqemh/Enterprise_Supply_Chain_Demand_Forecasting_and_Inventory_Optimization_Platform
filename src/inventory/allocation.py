"""
Warehouse Allocation Engine
Determines how new incoming inventory / procurement quantity should be
distributed across warehouses for each product, based on each
warehouse's share of regional demand and current stock health.
Distinct from transfer recommendations (which move EXISTING stock) --
this decides how to split NEW stock/orders.
"""

import pandas as pd
import numpy as np


def load_data():
    inventory_plan = pd.read_csv("data/processed/inventory_plan.csv")
    features = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
    return inventory_plan, features


def compute_warehouse_demand_share(features_df, lookback_days=90):
    """
    For each product, compute each warehouse's share of that product's
    total recent demand -- this becomes the allocation weight.
    """
    cutoff = features_df["date"].max() - pd.Timedelta(days=lookback_days)
    recent = features_df[features_df["date"] >= cutoff]

    demand_by_product_wh = (
        recent.groupby(["product_id", "warehouse_id"])["units_sold"].sum().reset_index()
    )
    demand_by_product_wh["product_total_demand"] = (
        demand_by_product_wh.groupby("product_id")["units_sold"].transform("sum")
    )
    demand_by_product_wh["allocation_share"] = (
        demand_by_product_wh["units_sold"] / demand_by_product_wh["product_total_demand"].replace(0, np.nan)
    ).fillna(1 / demand_by_product_wh.groupby("product_id")["warehouse_id"].transform("count"))

    return demand_by_product_wh[["product_id", "warehouse_id", "allocation_share"]]


def build_allocation_plan(inventory_plan_df, allocation_shares_df):
    """
    For each product with a pending procurement order (recommended_order_qty > 0
    at ANY warehouse), distribute the TOTAL recommended order quantity for that
    product across warehouses proportional to their demand share -- rather than
    each warehouse independently ordering its own EOQ. This avoids over-ordering.
    """
    df = inventory_plan_df.merge(allocation_shares_df, on=["product_id", "warehouse_id"], how="left")
    df["allocation_share"] = df["allocation_share"].fillna(
        1 / df.groupby("product_id")["warehouse_id"].transform("count")
    )

    # total procurement need per product = sum of EOQ across warehouses that need reorder
    product_total_order = (
        df[df["recommended_order_qty"] > 0]
        .groupby("product_id")["recommended_order_qty"].sum()
        .rename("product_total_order_qty")
    )

    df = df.merge(product_total_order, on="product_id", how="left")
    df["product_total_order_qty"] = df["product_total_order_qty"].fillna(0)

    df["allocated_order_qty"] = (df["product_total_order_qty"] * df["allocation_share"]).round()

    # allocation health: is current stock proportion aligned with demand share?
    total_stock_per_product = df.groupby("product_id")["current_stock"].transform("sum")
    df["stock_share"] = (df["current_stock"] / total_stock_per_product.replace(0, np.nan)).fillna(0)
    df["allocation_gap"] = df["allocation_share"] - df["stock_share"]

    df["allocation_status"] = np.select(
        [df["allocation_gap"] > 0.15, df["allocation_gap"] < -0.15],
        ["UNDER_ALLOCATED", "OVER_ALLOCATED"],
        default="BALANCED"
    )

    return df[[
        "product_id", "warehouse_id", "current_stock", "allocation_share",
        "stock_share", "allocation_gap", "allocation_status", "allocated_order_qty"
    ]]


def main():
    inventory_plan, features = load_data()

    print("Computing warehouse demand shares...")
    shares = compute_warehouse_demand_share(features)

    print("Building allocation plan...")
    allocation = build_allocation_plan(inventory_plan, shares)

    allocation.to_csv("data/processed/warehouse_allocation.csv", index=False)

    print("\nAllocation status breakdown:")
    print(allocation["allocation_status"].value_counts())
    print(f"\nTotal newly allocated order quantity: {allocation['allocated_order_qty'].sum():.0f} units")
    print("Saved: data/processed/warehouse_allocation.csv")


if __name__ == "__main__":
    main()