# Enterprise Supply Chain Demand Forecasting & Inventory Optimization Platform

## Overview

An end-to-end AI-powered platform that forecasts product demand, optimizes
inventory levels, detects supply chain risks, simulates operational
scenarios, and exposes everything via a secured REST API and executive
dashboard — built for a simulated multinational retail operation
(60 products × 6 warehouses × 3 years of daily sales history, ~394,200 records).

## Project Structure

```
ml021-demand-forecasting/
├── data/
│   ├── raw/                   # generated source data (sales, weather, events, etc.)
│   └── processed/             # engineered features, inventory plans, forecasts, scenarios
├── src/
│   ├── data_generation.py     # synthetic enterprise dataset generator
│   ├── feature_engineering.py # 38-column feature pipeline
│   ├── forecasting/
│   │   ├── train.py                  # XGBoost/LightGBM/Prophet training + time-series CV
│   │   ├── hyperparameter_tuning.py  # Optuna Bayesian optimization
│   │   ├── feature_selection.py      # correlation pruning + importance ranking
│   │   ├── multi_granularity.py      # daily/weekly/monthly/regional/category forecasts
│   │   └── explainability.py         # SHAP global + per-prediction explanations
│   ├── inventory/
│   │   ├── optimization.py    # reorder point, safety stock, EOQ, stockout probability
│   │   └── allocation.py      # warehouse allocation for new procurement
│   ├── risk/
│   │   ├── detection.py               # supplier delay, stockout, overstock, dead stock, anomalies
│   │   └── capacity_and_drift.py      # warehouse capacity risk + forecast drift alerts
│   ├── simulation/
│   │   └── scenario_engine.py # supplier failure, demand surge, holiday sales, price
│   │                           # increase, transportation delay, new product launch
│   ├── mlops/
│   │   ├── drift_monitoring.py    # PSI-based feature drift + automated retraining trigger
│   │   ├── model_monitoring.py    # deployed model performance tracking vs baseline
│   │   └── export_metrics.py      # exports MLflow metrics for the dashboard/API
│   ├── db/
│   │   ├── schema.py           # PostgreSQL schema
│   │   └── load_to_postgres.py # loads processed outputs into PostgreSQL
│   └── api/
│       └── main.py             # FastAPI service (PostgreSQL + Redis + API-key auth)
├── dashboard/
│   └── app.py                  # Streamlit executive dashboard
├── reports/                     # SHAP plots, drift reports, feature selection, model metrics
├──docs/
│   └──Deployment_Guide.pdf
│   └──Model Evaluation Report.pdf
│   └──Architecture_Diagram_Enterprise_Supply_Chain_Demand_Forecasting_and_Inventory_Optimization_Platform.jpg
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Requires PostgreSQL (database `ml021_supply_chain`) and Redis/Memurai
running locally. Configure `.env`:

```
POSTGRES_URL=postgresql://postgres:<password>@localhost:5432/ml021_supply_chain
REDIS_URL=redis://localhost:6379/0
FORECAST_API_KEY=<your-key>
```

Full step-by-step instructions: `Deployment_Guide.pdf`.

## Running the Pipeline

Run in this order (each stage feeds the next):

```bash
python src/data_generation.py
python src/feature_engineering.py
python src/inventory/optimization.py
python src/risk/detection.py
python src/inventory/allocation.py
python src/simulation/scenario_engine.py
python src/risk/capacity_and_drift.py
python src/forecasting/train.py
python src/forecasting/feature_selection.py
python src/forecasting/hyperparameter_tuning.py
python src/forecasting/multi_granularity.py
python src/forecasting/explainability.py
python src/mlops/drift_monitoring.py
python src/mlops/model_monitoring.py
python src/mlops/export_metrics.py
python src/db/schema.py
python src/db/load_to_postgres.py
```

## Running the Services

```bash
# Terminal 1
uvicorn src.api.main:app --reload --port 8000

# Terminal 2
streamlit run dashboard/app.py
```

API docs: `http://127.0.0.1:8000/docs` (use the "Authorize" button with
your `FORECAST_API_KEY`).

## Key Results

| Metric | Value |
|---|---|
| XGBoost forecast accuracy | 86.5% (MAPE 13.4%) |
| Bayesian-tuned XGBoost MAE | 3.82
| Baseline XGBoost MAE | 3.77
| Total alerts generated | 156 |
| Inter-warehouse transfer recommendations | 63 |
| Avg stockout probability across all product-warehouse pairs | 52.7% |
| High-risk (>50% stockout probability) pairs | 191 / 360 |
| Drift-triggered automated retraining | Verified (24.1% feature drift → real retraining run) |

## ML Stack

Python, FastAPI, PostgreSQL, Redis, Pandas, NumPy, Scikit-learn,
XGBoost, LightGBM, Prophet, MLflow, SHAP, Optuna, Streamlit.

## Scope Notes

- **LSTM and CatBoost** (marked Optional in the spec) were not implemented;
  XGBoost, LightGBM, and Prophet cover the required forecasting comparison.
- **Bonus challenges** (RL inventory policies, GNN, digital twin, etc.)
  were out of scope for this implementation by design.
- **100M forecast requests / High Availability / Horizontal Scaling** are
  documented as architectural design intent (see `Architecture_Diagram`)
  rather than load-tested — infeasible to demonstrate on a single-developer
  local setup, but the stateless FastAPI + PostgreSQL + Redis design
  supports horizontal scaling in principle. 
- **Feature selection** was implemented and evaluated but not adopted for
  production: the reduced 15-feature set scored worse (MAE 4.02) than the
  full 34-feature set (MAE 3.92), so the full feature set is used.
- Synthetic data has known limitations (e.g., minimum simulated demand
  level meant slow-moving/dead-inventory detection logic never triggered
  on this dataset, though it functions correctly) — noted in the Model
  Evaluation Report.
