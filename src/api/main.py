"""
FastAPI Service Layer
Exposes demand forecasting, inventory recommendations, risk alerts,
and transfer recommendations via REST endpoints. Backed by PostgreSQL,
cached via Redis, secured with an API key.
"""

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import mlflow
import mlflow.pyfunc
import numpy as np
import os
import json
import redis
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

DB_ENGINE = create_engine(os.getenv("POSTGRES_URL"))
REDIS_CLIENT = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
CACHE_TTL_SECONDS = 30

API_KEY = os.getenv("FORECAST_API_KEY", "dev-secret-key-change-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key


app = FastAPI(
    title="Supply Chain Demand Forecasting & Inventory Optimization API",
    version="1.0.0",
)

MODEL = None
FEATURE_COLS = None


@app.on_event("startup")
def load_artifacts():
    global MODEL, FEATURE_COLS
    MODEL = mlflow.pyfunc.load_model("models:/demand_forecast_xgboost/latest")
    features_df = pd.read_csv("data/processed/features.csv", nrows=1)
    FEATURE_COLS = [c for c in features_df.columns if c not in ["date", "product_id", "warehouse_id", "units_sold"]]
    print(f"Model loaded. Expecting {len(FEATURE_COLS)} features.")


def query_df(sql, params=None):
    with DB_ENGINE.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def cached_query(cache_key, sql, params=None):
    cached = REDIS_CLIENT.get(cache_key)
    if cached:
        return json.loads(cached)

    df = query_df(sql, params)
    records = df.to_dict(orient="records")
    REDIS_CLIENT.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(records, default=str))
    return records


class ForecastRequest(BaseModel):
    features: dict = Field(..., description="Dict of feature_name -> value, matching training feature columns.")


class ForecastResponse(BaseModel):
    predicted_demand: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/forecast/predict", response_model=ForecastResponse)
def predict_demand(request: ForecastRequest, api_key: str = Depends(verify_api_key)):
    row = {col: request.features.get(col, 0) for col in FEATURE_COLS}
    X = pd.DataFrame([row])[FEATURE_COLS]
    pred = MODEL.predict(X)
    pred_value = max(0.0, float(np.array(pred).flatten()[0]))
    return ForecastResponse(predicted_demand=round(pred_value, 2))


@app.get("/inventory/recommendations")
def get_inventory_recommendations(
    warehouse_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    api_key: str = Depends(verify_api_key),
):
    sql = "SELECT * FROM inventory_plan WHERE 1=1"
    params = {}
    if warehouse_id:
        sql += " AND warehouse_id = :warehouse_id"
        params["warehouse_id"] = warehouse_id
    if status:
        sql += " AND stock_status = :status"
        params["status"] = status.upper()
    sql += " LIMIT :limit"
    params["limit"] = limit

    return query_df(sql, params).to_dict(orient="records")


@app.get("/inventory/transfers")
def get_transfer_recommendations(product_id: Optional[str] = None, api_key: str = Depends(verify_api_key)):
    sql = "SELECT * FROM transfer_recommendations WHERE 1=1"
    params = {}
    if product_id:
        sql += " AND product_id = :product_id"
        params["product_id"] = product_id
    return query_df(sql, params).to_dict(orient="records")


@app.get("/alerts")
def get_alerts(
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = 100,
    api_key: str = Depends(verify_api_key),
):
    cache_key = f"alerts:{severity}:{alert_type}:{limit}"
    sql = "SELECT * FROM alerts WHERE 1=1"
    params = {}
    if severity:
        sql += " AND severity = :severity"
        params["severity"] = severity.upper()
    if alert_type:
        sql += " AND alert_type = :alert_type"
        params["alert_type"] = alert_type.upper()
    sql += " LIMIT :limit"
    params["limit"] = limit

    return cached_query(cache_key, sql, params)


@app.get("/dashboard/summary")
def dashboard_summary(api_key: str = Depends(verify_api_key)):
    cache_key = "dashboard:summary"
    cached = REDIS_CLIENT.get(cache_key)
    if cached:
        return json.loads(cached)

    inventory = query_df("SELECT stock_status, COUNT(*) as cnt FROM inventory_plan GROUP BY stock_status")
    status_counts = dict(zip(inventory["stock_status"], inventory["cnt"]))

    alerts = query_df("SELECT severity, COUNT(*) as cnt FROM alerts GROUP BY severity")
    alert_counts = dict(zip(alerts["severity"], alerts["cnt"]))

    total_pairs = query_df("SELECT COUNT(*) as cnt FROM inventory_plan")["cnt"].iloc[0]
    critical = status_counts.get("CRITICAL_STOCKOUT_RISK", 0)
    fill_rate = round(1 - (critical / total_pairs), 4) if total_pairs else 0

    transfers_count = query_df("SELECT COUNT(*) as cnt FROM transfer_recommendations")["cnt"].iloc[0]

    metrics_row = query_df(
        "SELECT * FROM model_metrics ORDER BY recorded_at DESC LIMIT 1"
    )
    forecast_accuracy = metrics_row.to_dict(orient="records")[0] if len(metrics_row) else None

    warehouse_utilization = query_df(
        "SELECT warehouse_id, utilization_pct, capacity_status FROM warehouse_utilization"
    ).to_dict(orient="records")

    avg_stockout_prob = query_df(
        "SELECT AVG(stockout_probability) as avg_prob, COUNT(*) FILTER (WHERE stockout_probability > 0.5) as high_risk_count FROM inventory_plan"
    )
    stockout_probability_summary = {
        "avg_stockout_probability": round(float(avg_stockout_prob["avg_prob"].iloc[0] or 0), 4),
        "high_risk_pairs_count": int(avg_stockout_prob["high_risk_count"].iloc[0]),
    }

    result = {
        "total_product_warehouse_pairs": int(total_pairs),
        "stock_status_breakdown": status_counts,
        "alert_severity_breakdown": alert_counts,
        "estimated_fill_rate": fill_rate,
        "total_transfer_recommendations": int(transfers_count),
        "forecast_accuracy": forecast_accuracy,
        "warehouse_utilization": warehouse_utilization,
        "stockout_probability_summary": stockout_probability_summary,
    }

    REDIS_CLIENT.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(result, default=str))
    return result

@app.get("/inventory/allocation")
def get_allocation_recommendations(product_id: Optional[str] = None, api_key: str = Depends(verify_api_key)):
    sql = "SELECT * FROM warehouse_allocation WHERE 1=1"
    params = {}
    if product_id:
        sql += " AND product_id = :product_id"
        params["product_id"] = product_id
    return query_df(sql, params).to_dict(orient="records")