"""
PostgreSQL Schema Setup
Creates all tables needed to back the FastAPI service, replacing the
CSV-based reads used during development.
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

SCHEMA_SQL = """
DROP TABLE IF EXISTS inventory_plan CASCADE;
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS transfer_recommendations CASCADE;
DROP TABLE IF EXISTS warehouse_allocation CASCADE;
DROP TABLE IF EXISTS warehouse_utilization CASCADE;
DROP TABLE IF EXISTS daily_forecast CASCADE;
DROP TABLE IF EXISTS model_metrics CASCADE;

CREATE TABLE inventory_plan (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(20),
    warehouse_id VARCHAR(20),
    avg_daily_demand FLOAT,
    demand_std FLOAT,
    avg_lead_time_days FLOAT,
    reliability_score FLOAT,
    unit_cost FLOAT,
    current_stock INTEGER,
    safety_stock FLOAT,
    reorder_point FLOAT,
    eoq FLOAT,
    stock_status VARCHAR(30),
    recommended_order_qty FLOAT,
    stockout_probability FLOAT
);

CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(20),
    warehouse_id VARCHAR(20),
    alert_type VARCHAR(50),
    severity VARCHAR(10),
    message TEXT
);

CREATE TABLE transfer_recommendations (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(20),
    from_warehouse VARCHAR(20),
    to_warehouse VARCHAR(20),
    transfer_qty INTEGER
);

CREATE TABLE warehouse_allocation (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(20),
    warehouse_id VARCHAR(20),
    current_stock INTEGER,
    allocation_share FLOAT,
    stock_share FLOAT,
    allocation_gap FLOAT,
    allocation_status VARCHAR(30),
    allocated_order_qty FLOAT
);

CREATE TABLE warehouse_utilization (
    id SERIAL PRIMARY KEY,
    warehouse_id VARCHAR(20),
    total_stock INTEGER,
    capacity_units INTEGER,
    utilization_pct FLOAT,
    capacity_status VARCHAR(20)
);

CREATE TABLE daily_forecast (
    id SERIAL PRIMARY KEY,
    date DATE,
    product_id VARCHAR(20),
    warehouse_id VARCHAR(20),
    units_sold FLOAT,
    predicted_demand FLOAT
);

CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    avg_mae FLOAT,
    avg_rmse FLOAT,
    avg_mape FLOAT,
    accuracy_pct FLOAT,
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_inventory_status ON inventory_plan(stock_status);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_daily_forecast_pw ON daily_forecast(product_id, warehouse_id);
"""


def create_schema():
    with engine.connect() as conn:
        conn.execute(text(SCHEMA_SQL))
        conn.commit()
    print("Schema created successfully.")


if __name__ == "__main__":
    create_schema()