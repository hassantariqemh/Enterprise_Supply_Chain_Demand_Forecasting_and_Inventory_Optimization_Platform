"""
Demand Forecasting Engine
Trains XGBoost & LightGBM models (with time-series CV) and a Prophet
baseline for comparison. Tracks everything via MLflow and registers
the best model.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("demand_forecasting")

TARGET = "units_sold"
DROP_COLS = ["date", "product_id", "warehouse_id", TARGET]


def load_features(path="data/processed/features.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")  # critical for time-series split
    return df


def get_feature_cols(df):
    return [c for c in df.columns if c not in DROP_COLS]


def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(np.where(y_true == 0, 1e-3, y_true), y_pred),
    }


def train_xgboost_cv(df, feature_cols, n_splits=5):
    """Walk-forward time-series cross-validation for XGBoost."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X, y = df[feature_cols], df[TARGET]

    fold_metrics = []
    best_model, best_mae = None, float("inf")

    with mlflow.start_run(run_name="xgboost_tscv"):
        params = {
            "n_estimators": 400,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        mlflow.log_params(params)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = np.clip(preds, 0, None)  # demand can't be negative

            metrics = evaluate(y_val, preds)
            fold_metrics.append(metrics)
            print(f"[XGBoost] Fold {fold+1}: MAE={metrics['mae']:.2f} "
                  f"RMSE={metrics['rmse']:.2f} MAPE={metrics['mape']:.3f}")

            mlflow.log_metrics({f"fold{fold+1}_{k}": v for k, v in metrics.items()})

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_model = model

        avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        mlflow.log_metrics({f"avg_{k}": v for k, v in avg_metrics.items()})
        mlflow.xgboost.log_model(best_model, "model", registered_model_name="demand_forecast_xgboost")

        print(f"\n[XGBoost] Avg across folds: {avg_metrics}")

    return best_model, avg_metrics


def train_lightgbm_cv(df, feature_cols, n_splits=5):
    """Walk-forward time-series cross-validation for LightGBM."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X, y = df[feature_cols], df[TARGET]

    fold_metrics = []
    best_model, best_mae = None, float("inf")

    with mlflow.start_run(run_name="lightgbm_tscv"):
        params = {
            "n_estimators": 400,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        mlflow.log_params(params)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = np.clip(preds, 0, None)

            metrics = evaluate(y_val, preds)
            fold_metrics.append(metrics)
            print(f"[LightGBM] Fold {fold+1}: MAE={metrics['mae']:.2f} "
                  f"RMSE={metrics['rmse']:.2f} MAPE={metrics['mape']:.3f}")

            mlflow.log_metrics({f"fold{fold+1}_{k}": v for k, v in metrics.items()})

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_model = model

        avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        mlflow.log_metrics({f"avg_{k}": v for k, v in avg_metrics.items()})
        mlflow.lightgbm.log_model(best_model, "model", registered_model_name="demand_forecast_lightgbm")

        print(f"\n[LightGBM] Avg across folds: {avg_metrics}")

    return best_model, avg_metrics


def train_prophet_baseline(df, product_id, warehouse_id):
    """
    Prophet baseline for a single product-warehouse series (Prophet works
    per-series, not on the full tabular feature matrix). Used as a
    comparison benchmark against XGBoost/LightGBM.
    """
    from prophet import Prophet

    series = df[(df["product_id"] == product_id) & (df["warehouse_id"] == warehouse_id)]
    series = series[["date", TARGET]].rename(columns={"date": "ds", TARGET: "y"})

    split_point = int(len(series) * 0.85)
    train, test = series.iloc[:split_point], series.iloc[split_point:]

    with mlflow.start_run(run_name=f"prophet_baseline_{product_id}_{warehouse_id}"):
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(train)

        future = test[["ds"]]
        forecast = model.predict(future)
        preds = np.clip(forecast["yhat"].values, 0, None)

        metrics = evaluate(test["y"].values, preds)
        mlflow.log_metrics(metrics)
        print(f"[Prophet baseline {product_id}/{warehouse_id}]: {metrics}")

    return model, metrics


def main():
    df = load_features()
    feature_cols = get_feature_cols(df)
    print(f"Training on {len(df):,} rows with {len(feature_cols)} features.\n")

    xgb_model, xgb_metrics = train_xgboost_cv(df, feature_cols)
    lgb_model, lgb_metrics = train_lightgbm_cv(df, feature_cols)

    # Prophet baseline on one representative product-warehouse pair
    sample_pid = df["product_id"].iloc[0]
    sample_wid = df["warehouse_id"].iloc[0]
    train_prophet_baseline(df, sample_pid, sample_wid)

    print("\n=== Summary ===")
    print(f"XGBoost  avg MAE: {xgb_metrics['mae']:.2f}, MAPE: {xgb_metrics['mape']:.3f}")
    print(f"LightGBM avg MAE: {lgb_metrics['mae']:.2f}, MAPE: {lgb_metrics['mape']:.3f}")


if __name__ == "__main__":
    main()