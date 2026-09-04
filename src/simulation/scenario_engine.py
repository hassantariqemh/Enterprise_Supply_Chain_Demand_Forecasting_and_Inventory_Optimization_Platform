"""
Scenario Simulation Engine
Simulates the operational impact of supply chain disruptions/events
BEFORE they happen, using the current inventory plan and demand model
as the baseline. Supports: supplier failure, price increase, holiday
sales surge, new product launch, transportation delay, demand surge.
"""

import pandas as pd
import numpy as np


def load_baseline():
    inventory_plan = pd.read_csv("data/processed/inventory_plan.csv")
    return inventory_plan


# ---------- Scenario 1: Supplier Failure ----------
def simulate_supplier_failure(inventory_df, supplier_reliability_drop=0.0, lead_time_multiplier=3.0,
                               affected_products=None):
    """
    Simulates a supplier suddenly becoming unreliable/failing entirely --
    lead time blows up, reliability drops. Recomputes reorder point and
    safety stock under the new conditions to show which products go
    into stockout risk.
    """
    df = inventory_df.copy()
    if affected_products:
        mask = df["product_id"].isin(affected_products)
    else:
        mask = pd.Series(True, index=df.index)

    df.loc[mask, "avg_lead_time_days"] *= lead_time_multiplier
    df.loc[mask, "reliability_score"] = (df.loc[mask, "reliability_score"] - supplier_reliability_drop).clip(0, 1)

    df["new_safety_stock"] = df["avg_daily_demand"] * np.sqrt(df["avg_lead_time_days"]) * 1.645 * \
        (1 + (1 - df["reliability_score"]))
    df["new_reorder_point"] = (df["avg_daily_demand"] * df["avg_lead_time_days"]) + df["new_safety_stock"]

    df["days_of_stock_remaining"] = (df["current_stock"] / df["avg_daily_demand"].replace(0, np.nan)).fillna(0)
    df["new_stockout_risk"] = df["current_stock"] < df["new_safety_stock"]

    impacted = df[mask & df["new_stockout_risk"]]
    return {
        "scenario": "supplier_failure",
        "affected_pairs": int(mask.sum()),
        "new_stockout_risk_count": int(impacted.shape[0]),
        "products_at_risk": impacted["product_id"].unique().tolist()[:20],
        "detail": df[mask][["product_id", "warehouse_id", "current_stock", "new_safety_stock",
                             "new_reorder_point", "new_stockout_risk"]],
    }


# ---------- Scenario 2: Demand Surge / Holiday Sales ----------
def simulate_demand_surge(inventory_df, surge_multiplier=2.0, duration_days=7, affected_products=None):
    """
    Simulates a sudden demand surge (holiday sale, viral product, etc.)
    and estimates how many days of stock would survive, and how many
    warehouses would stock out before the surge ends.
    """
    df = inventory_df.copy()
    if affected_products:
        mask = df["product_id"].isin(affected_products)
    else:
        mask = pd.Series(True, index=df.index)

    df["surged_daily_demand"] = df["avg_daily_demand"]
    df.loc[mask, "surged_daily_demand"] *= surge_multiplier

    df["projected_consumption"] = df["surged_daily_demand"] * duration_days
    df["stock_after_surge"] = df["current_stock"] - df["projected_consumption"]
    df["will_stock_out"] = df["stock_after_surge"] < 0

    df["days_until_stockout"] = np.where(
        df["surged_daily_demand"] > 0,
        df["current_stock"] / df["surged_daily_demand"],
        np.inf
    )

    impacted = df[mask & df["will_stock_out"]]
    return {
        "scenario": "demand_surge",
        "surge_multiplier": surge_multiplier,
        "duration_days": duration_days,
        "pairs_that_stock_out": int(impacted.shape[0]),
        "products_at_risk": impacted["product_id"].unique().tolist()[:20],
        "detail": df[mask][["product_id", "warehouse_id", "current_stock", "surged_daily_demand",
                             "stock_after_surge", "days_until_stockout", "will_stock_out"]],
    }

def simulate_holiday_sales(inventory_df, holiday_multiplier=2.2, duration_days=5):
    """
    Simulates a major holiday sales event (e.g. Black Friday, Christmas) --
    distinct from generic demand_surge by using the actual holiday demand
    multiplier scale observed in the data, and a shorter, sharper window.
    """
    df = inventory_df.copy()
    df["holiday_daily_demand"] = df["avg_daily_demand"] * holiday_multiplier
    df["projected_holiday_consumption"] = df["holiday_daily_demand"] * duration_days
    df["stock_after_holiday"] = df["current_stock"] - df["projected_holiday_consumption"]
    df["will_stock_out"] = df["stock_after_holiday"] < 0

    impacted = df[df["will_stock_out"]]
    return {
        "scenario": "holiday_sales",
        "holiday_multiplier": holiday_multiplier,
        "duration_days": duration_days,
        "pairs_that_stock_out": int(impacted.shape[0]),
        "products_at_risk": impacted["product_id"].unique().tolist()[:20],
        "detail": df[["product_id", "warehouse_id", "current_stock", "holiday_daily_demand",
                       "stock_after_holiday", "will_stock_out"]],
    }

# ---------- Scenario 3: Transportation Delay ----------
def simulate_transportation_delay(inventory_df, extra_delay_days=5, affected_warehouses=None):
    """
    Simulates a transportation/logistics delay adding extra days on top
    of normal lead time for specific warehouses -- shows exposure.
    """
    df = inventory_df.copy()
    if affected_warehouses:
        mask = df["warehouse_id"].isin(affected_warehouses)
    else:
        mask = pd.Series(True, index=df.index)

    df.loc[mask, "delayed_lead_time"] = df.loc[mask, "avg_lead_time_days"] + extra_delay_days
    df["delayed_lead_time"] = df["delayed_lead_time"].fillna(df["avg_lead_time_days"])

    df["stock_coverage_days"] = (df["current_stock"] / df["avg_daily_demand"].replace(0, np.nan)).fillna(0)
    df["delay_exceeds_coverage"] = df["delayed_lead_time"] > df["stock_coverage_days"]

    impacted = df[mask & df["delay_exceeds_coverage"]]
    return {
        "scenario": "transportation_delay",
        "extra_delay_days": extra_delay_days,
        "pairs_at_risk": int(impacted.shape[0]),
        "detail": df[mask][["product_id", "warehouse_id", "delayed_lead_time",
                             "stock_coverage_days", "delay_exceeds_coverage"]],
    }


# ---------- Scenario 4: Price Increase (demand elasticity proxy) ----------
def simulate_price_increase(inventory_df, price_increase_pct=0.15, elasticity=-1.2):
    """
    Simulates a price hike and its effect on demand using a simple
    constant-elasticity assumption: %change in demand = elasticity * %change in price.
    (Standard economics approximation -- real elasticity would come from
    historical price/demand regression, noted as a future enhancement.)
    """
    df = inventory_df.copy()
    demand_change_pct = elasticity * price_increase_pct
    df["projected_daily_demand"] = df["avg_daily_demand"] * (1 + demand_change_pct)
    df["projected_daily_demand"] = df["projected_daily_demand"].clip(lower=0)

    df["revenue_per_unit_change_pct"] = price_increase_pct
    df["projected_daily_revenue_change_pct"] = (
        (1 + price_increase_pct) * (1 + demand_change_pct) - 1
    )

    return {
        "scenario": "price_increase",
        "price_increase_pct": price_increase_pct,
        "assumed_elasticity": elasticity,
        "avg_demand_change_pct": round(demand_change_pct * 100, 2),
        "avg_projected_revenue_change_pct": round(df["projected_daily_revenue_change_pct"].mean() * 100, 2),
        "detail": df[["product_id", "warehouse_id", "avg_daily_demand", "projected_daily_demand"]],
    }


# ---------- Scenario 5: New Product Launch ----------
def simulate_new_product_launch(inventory_df, comparable_category_avg_demand, initial_stock_units,
                                 num_warehouses, launch_surge_multiplier=1.8):
    """
    Simulates a new product launch using analogous-category average
    demand as the baseline forecast (cold-start problem -- no history
    exists yet), with a launch-week surge multiplier.
    """
    per_warehouse_stock = initial_stock_units / num_warehouses
    launch_week_demand = comparable_category_avg_demand * launch_surge_multiplier

    days_of_coverage = per_warehouse_stock / launch_week_demand if launch_week_demand > 0 else float("inf")
    stockout_before_replenishment = days_of_coverage < 14  # assume 14-day standard lead time

    return {
        "scenario": "new_product_launch",
        "initial_stock_per_warehouse": round(per_warehouse_stock, 1),
        "projected_launch_week_daily_demand": round(launch_week_demand, 1),
        "estimated_days_of_coverage": round(days_of_coverage, 1),
        "stockout_risk_before_replenishment": bool(stockout_before_replenishment),
    }


def main():
    inventory_df = load_baseline()

    print("=" * 60)
    print("SCENARIO 1: Supplier Failure (top 10 products)")
    print("=" * 60)
    top_products = inventory_df["product_id"].unique()[:10].tolist()
    result = simulate_supplier_failure(inventory_df, lead_time_multiplier=3.0, affected_products=top_products)
    print(f"Affected pairs: {result['affected_pairs']}")
    print(f"New stockout-risk pairs: {result['new_stockout_risk_count']}")
    print(f"Products at risk: {result['products_at_risk']}")

    print("\n" + "=" * 60)
    print("SCENARIO 2: Demand Surge (2x for 7 days, all products)")
    print("=" * 60)
    result2 = simulate_demand_surge(inventory_df, surge_multiplier=2.0, duration_days=7)
    print(f"Pairs that would stock out: {result2['pairs_that_stock_out']}")

    print("\n" + "=" * 60)
    print("SCENARIO 2b: Holiday Sales (2.2x for 5 days)")
    print("=" * 60)
    result2b = simulate_holiday_sales(inventory_df)
    print(f"Pairs that would stock out: {result2b['pairs_that_stock_out']}")

    print("\n" + "=" * 60)
    print("SCENARIO 3: Transportation Delay (+5 days, all warehouses)")
    print("=" * 60)
    result3 = simulate_transportation_delay(inventory_df, extra_delay_days=5)
    print(f"Pairs at risk: {result3['pairs_at_risk']}")

    print("\n" + "=" * 60)
    print("SCENARIO 4: Price Increase (15%)")
    print("=" * 60)
    result4 = simulate_price_increase(inventory_df, price_increase_pct=0.15)
    print(f"Avg demand change: {result4['avg_demand_change_pct']}%")
    print(f"Avg projected revenue change: {result4['avg_projected_revenue_change_pct']}%")

    print("\n" + "=" * 60)
    print("SCENARIO 5: New Product Launch")
    print("=" * 60)
    category_avg = inventory_df["avg_daily_demand"].mean()
    result5 = simulate_new_product_launch(
        inventory_df, comparable_category_avg_demand=category_avg,
        initial_stock_units=500, num_warehouses=6
    )
    print(result5)

    # Save all detail tables
    import os
    os.makedirs("data/processed/scenarios", exist_ok=True)
    result["detail"].to_csv("data/processed/scenarios/supplier_failure.csv", index=False)
    result2["detail"].to_csv("data/processed/scenarios/demand_surge.csv", index=False)
    result2b["detail"].to_csv("data/processed/scenarios/holiday_sales.csv", index=False)
    result3["detail"].to_csv("data/processed/scenarios/transportation_delay.csv", index=False)
    result4["detail"].to_csv("data/processed/scenarios/price_increase.csv", index=False)

    print("\nSaved scenario detail tables to data/processed/scenarios/")


if __name__ == "__main__":
    main()