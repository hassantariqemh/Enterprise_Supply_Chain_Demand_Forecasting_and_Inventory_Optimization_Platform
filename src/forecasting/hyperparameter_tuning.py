"""
Bayesian Hyperparameter Optimization
Uses Optuna (Tree-structured Parzen Estimator -- a Bayesian optimization
method) to tune XGBoost hyperparameters against time-series CV performance,
then logs the best trial and retrains/registers the tuned model via MLflow.
"""

import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("demand_forecasting_tuning")

TARGET = "units_sold"
DROP_COLS = ["date", "product_id", "warehouse_id", TARGET]
N_TRIALS = 25
N_SPLITS = 3  # fewer folds during search to keep it fast; final model uses full CV


def load_data(path="data/processed/features.csv"):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    return df, feature_cols


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "objective": "reg:squarederror",
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_maes = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_val), 0, None)
        fold_maes.append(mean_absolute_error(y_val, preds))

    return float(np.mean(fold_maes))


def run_optimization():
    df, feature_cols = load_data()
    X, y = df[feature_cols], df[TARGET]

    study = optuna.create_study(direction="minimize", study_name="xgboost_demand_forecast")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest MAE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    with mlflow.start_run(run_name="xgboost_bayesian_tuned"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mae", study.best_value)

        best_model = xgb.XGBRegressor(**study.best_params, objective="reg:squarederror", random_state=42)
        best_model.fit(X, y)

        mlflow.xgboost.log_model(
            best_model, "model", registered_model_name="demand_forecast_xgboost_tuned"
        )

    return study, best_model


if __name__ == "__main__":
    run_optimization()