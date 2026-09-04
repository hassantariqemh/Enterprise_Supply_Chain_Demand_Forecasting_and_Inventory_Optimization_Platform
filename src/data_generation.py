"""
Synthetic Enterprise Supply Chain Data Generator
Generates realistic historical sales, inventory, supplier, promotion,
weather, and regional event data for the demand forecasting &
inventory optimization platform.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

# ---------- CONFIG ----------
NUM_PRODUCTS = 60
NUM_WAREHOUSES = 6
NUM_SUPPLIERS = 15
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 12, 31)
CATEGORIES = ["Electronics", "Groceries", "Apparel", "Home & Kitchen", "Toys", "Sports"]
REGIONS = ["North", "South", "East", "West", "Central"]


def generate_products(n=NUM_PRODUCTS):
    products = []
    for i in range(1, n + 1):
        category = random.choice(CATEGORIES)
        products.append({
            "product_id": f"P{i:04d}",
            "product_name": f"{category[:-1] if category.endswith('s') else category} Item {i}",
            "category": category,
            "unit_cost": round(np.random.uniform(5, 500), 2),
            "unit_price": None,
            "lifecycle_stage": random.choices(
                ["Introduction", "Growth", "Maturity", "Decline"],
                weights=[0.1, 0.3, 0.5, 0.1]
            )[0],
        })
    df = pd.DataFrame(products)
    df["unit_price"] = (df["unit_cost"] * np.random.uniform(1.2, 2.0, size=len(df))).round(2)
    return df


def generate_warehouses(n=NUM_WAREHOUSES):
    warehouses = []
    for i in range(1, n + 1):
        warehouses.append({
            "warehouse_id": f"W{i:03d}",
            "warehouse_name": f"Warehouse {i}",
            "region": REGIONS[i % len(REGIONS)],
            "capacity_units": np.random.randint(20000, 100000),
        })
    return pd.DataFrame(warehouses)


def generate_suppliers(n=NUM_SUPPLIERS):
    suppliers = []
    for i in range(1, n + 1):
        suppliers.append({
            "supplier_id": f"S{i:03d}",
            "supplier_name": fake.company(),
            "avg_lead_time_days": np.random.randint(3, 30),
            "reliability_score": round(np.random.uniform(0.6, 0.99), 2),
        })
    return pd.DataFrame(suppliers)


def generate_product_supplier_map(products_df, suppliers_df):
    mapping = []
    for pid in products_df["product_id"]:
        chosen = random.sample(list(suppliers_df["supplier_id"]), k=random.choice([1, 2]))
        for sid in chosen:
            mapping.append({"product_id": pid, "supplier_id": sid})
    return pd.DataFrame(mapping)


def generate_holidays(start, end):
    holidays = []
    for year in range(start.year, end.year + 1):
        holidays += [
            {"date": datetime(year, 1, 1), "holiday_name": "New Year", "demand_multiplier": 1.4},
            {"date": datetime(year, 3, 25), "holiday_name": "Pakistan Day Sale", "demand_multiplier": 1.2},
            {"date": datetime(year, 7, 4), "holiday_name": "Summer Sale", "demand_multiplier": 1.3},
            {"date": datetime(year, 11, 25), "holiday_name": "Black Friday", "demand_multiplier": 2.2},
            {"date": datetime(year, 12, 25), "holiday_name": "Christmas", "demand_multiplier": 1.8},
        ]
    return pd.DataFrame(holidays)


def generate_promotions(products_df, start, end, n_promos=150):
    promos = []
    date_range_days = (end - start).days
    for _ in range(n_promos):
        pid = random.choice(products_df["product_id"].tolist())
        promo_start = start + timedelta(days=random.randint(0, date_range_days - 14))
        duration = random.randint(3, 14)
        promos.append({
            "product_id": pid,
            "promo_start": promo_start,
            "promo_end": promo_start + timedelta(days=duration),
            "discount_pct": random.choice([10, 15, 20, 25, 30, 40]),
        })
    return pd.DataFrame(promos)


def generate_sales(products_df, warehouses_df, holidays_df, promotions_df, start, end):
    date_range = pd.date_range(start, end, freq="D")
    holiday_map = holidays_df.set_index(holidays_df["date"].dt.date)["demand_multiplier"].to_dict()

    lifecycle_base = {
        "Introduction": 8, "Growth": 20, "Maturity": 35, "Decline": 10
    }

    records = []
    for _, product in products_df.iterrows():
        base_demand = lifecycle_base[product["lifecycle_stage"]] * np.random.uniform(0.7, 1.4)
        active_promos = promotions_df[promotions_df["product_id"] == product["product_id"]]

        for _, wh in warehouses_df.iterrows():
            wh_factor = np.random.uniform(0.5, 1.5)

            for date in date_range:
                dow_factor = 1.3 if date.dayofweek >= 5 else 1.0
                yearly_seasonality = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                holiday_mult = holiday_map.get(date.date(), 1.0)

                promo_mult = 1.0
                promo_hit = active_promos[
                    (active_promos["promo_start"] <= date) & (active_promos["promo_end"] >= date)
                ]
                if not promo_hit.empty:
                    promo_mult = 1 + (promo_hit.iloc[0]["discount_pct"] / 100) * 1.5

                noise = np.random.normal(1.0, 0.15)

                demand = (
                    base_demand * wh_factor * dow_factor *
                    yearly_seasonality * holiday_mult * promo_mult * noise
                )
                demand = max(0, round(demand))

                records.append({
                    "date": date,
                    "product_id": product["product_id"],
                    "warehouse_id": wh["warehouse_id"],
                    "units_sold": demand,
                })

    return pd.DataFrame(records)


def generate_inventory_snapshot(products_df, warehouses_df):
    """
    Current stock level snapshot per product-warehouse, sized relative
    to each warehouse's actual capacity so total utilization stays
    realistic (roughly 30-80% of capacity per warehouse).
    """
    records = []

    for _, wh in warehouses_df.iterrows():
        target_utilization = np.random.uniform(0.30, 0.80)
        target_total_stock = wh["capacity_units"] * target_utilization

        weights = np.random.dirichlet(np.ones(len(products_df)) * 2)
        product_stocks = (weights * target_total_stock).round().astype(int)

        for (_, product), stock in zip(products_df.iterrows(), product_stocks):
            records.append({
                "product_id": product["product_id"],
                "warehouse_id": wh["warehouse_id"],
                "current_stock": int(stock),
            })

    return pd.DataFrame(records)


def generate_weather_data(warehouses_df, start, end):
    """Daily weather per warehouse region -- temperature, precipitation, severity."""
    date_range = pd.date_range(start, end, freq="D")
    records = []
    for _, wh in warehouses_df.iterrows():
        base_temp = np.random.uniform(15, 30)
        for date in date_range:
            seasonal_temp = base_temp + 10 * np.sin(2 * np.pi * date.dayofyear / 365)
            temp = seasonal_temp + np.random.normal(0, 3)
            precipitation_mm = max(0, np.random.exponential(2) * (1.5 if date.month in [7, 8] else 1))
            is_severe = int(precipitation_mm > 15 or temp > 42 or temp < 2)

            records.append({
                "date": date,
                "warehouse_id": wh["warehouse_id"],
                "avg_temperature_c": round(temp, 1),
                "precipitation_mm": round(precipitation_mm, 1),
                "is_severe_weather": is_severe,
            })
    return pd.DataFrame(records)


def generate_regional_events(warehouses_df, start, end, n_events_per_region=8):
    """Regional events (festivals, sports events, expos) with a demand-impact multiplier."""
    event_types = ["Regional Festival", "Sports Event", "Trade Expo", "Local Holiday", "Concert/Fair"]
    date_range_days = (end - start).days
    records = []

    for region in warehouses_df["region"].unique():
        for _ in range(n_events_per_region):
            event_date = start + timedelta(days=random.randint(0, date_range_days))
            records.append({
                "region": region,
                "event_date": event_date,
                "event_name": random.choice(event_types),
                "demand_impact_multiplier": round(np.random.uniform(1.1, 1.6), 2),
            })
    return pd.DataFrame(records)


def main():
    print("Generating products...")
    products_df = generate_products()

    print("Generating warehouses...")
    warehouses_df = generate_warehouses()

    print("Generating suppliers...")
    suppliers_df = generate_suppliers()

    print("Generating product-supplier mapping...")
    product_supplier_df = generate_product_supplier_map(products_df, suppliers_df)

    print("Generating holiday calendar...")
    holidays_df = generate_holidays(START_DATE, END_DATE)

    print("Generating promotions...")
    promotions_df = generate_promotions(products_df, START_DATE, END_DATE)

    print("Generating sales history (this takes a bit)...")
    sales_df = generate_sales(products_df, warehouses_df, holidays_df, promotions_df, START_DATE, END_DATE)

    print("Generating current inventory snapshot...")
    inventory_df = generate_inventory_snapshot(products_df, warehouses_df)

    print("Generating weather data...")
    weather_df = generate_weather_data(warehouses_df, START_DATE, END_DATE)

    print("Generating regional events...")
    events_df = generate_regional_events(warehouses_df, START_DATE, END_DATE)

    products_df.to_csv("data/raw/products.csv", index=False)
    warehouses_df.to_csv("data/raw/warehouses.csv", index=False)
    suppliers_df.to_csv("data/raw/suppliers.csv", index=False)
    product_supplier_df.to_csv("data/raw/product_supplier_map.csv", index=False)
    holidays_df.to_csv("data/raw/holidays.csv", index=False)
    promotions_df.to_csv("data/raw/promotions.csv", index=False)
    sales_df.to_csv("data/raw/sales_history.csv", index=False)
    inventory_df.to_csv("data/raw/inventory_snapshot.csv", index=False)
    weather_df.to_csv("data/raw/weather.csv", index=False)
    events_df.to_csv("data/raw/regional_events.csv", index=False)

    print(f"\nDone. Generated {len(sales_df):,} sales records "
          f"across {NUM_PRODUCTS} products x {NUM_WAREHOUSES} warehouses "
          f"x {(END_DATE - START_DATE).days} days.")


if __name__ == "__main__":
    main()