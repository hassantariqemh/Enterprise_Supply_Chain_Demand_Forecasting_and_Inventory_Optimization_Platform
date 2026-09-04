"""
Explicit Feature Selection
Uses SHAP-based importance ranking + correlation pruning to select the
top predictive features, then compares model performance with the full
feature set vs the reduced set (documents the trade-off explicitly).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import json

TARGET = "units_sold"
DROP_COLS = ["date", "product_id", "warehouse_id", TARGET]
TOP_K_FEATURES = 15


def load_data():
    df = pd.read_csv("data/processed/features.csv", parse_dates=["date"]).sort_values("date")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    return df, feature_cols


def remove_correlated_features(df, feature_cols, threshold=0.95):
    """Drop features that are highly correlated with another (redundant signal)."""
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    remaining = [c for c in feature_cols if c not in to_drop]
    return remaining, to_drop


def rank_by_importance(df, feature_cols):
    """Quick XGBoost fit to rank features by gain-based importance."""
    X, y = df[feature_cols], df[TARGET]
    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return importance


def evaluate_feature_set(df, feature_cols, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X, y = df[feature_cols], df[TARGET]
    maes = []
    for train_idx, val_idx in tscv.split(X):
        model = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = np.clip(model.predict(X.iloc[val_idx]), 0, None)
        maes.append(mean_absolute_error(y.iloc[val_idx], preds))
    return float(np.mean(maes))


def main():
    df, all_features = load_data()
    print(f"Starting with {len(all_features)} features.")

    reduced_features, dropped_corr = remove_correlated_features(df, all_features)
    print(f"Dropped {len(dropped_corr)} highly correlated features: {dropped_corr}")

    importance = rank_by_importance(df, reduced_features)
    top_features = importance.head(TOP_K_FEATURES).index.tolist()
    print(f"\nTop {TOP_K_FEATURES} features by importance:\n{importance.head(TOP_K_FEATURES)}")

    print("\nEvaluating full feature set vs selected feature set...")
    mae_full = evaluate_feature_set(df, all_features)
    mae_selected = evaluate_feature_set(df, top_features)

    print(f"\nFull feature set ({len(all_features)} features)   -> MAE: {mae_full:.4f}")
    print(f"Selected feature set ({len(top_features)} features) -> MAE: {mae_selected:.4f}")

    result = {
        "full_feature_count": len(all_features),
        "selected_feature_count": len(top_features),
        "dropped_correlated_features": dropped_corr,
        "selected_features": top_features,
        "mae_full_features": mae_full,
        "mae_selected_features": mae_selected,
    }

    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/feature_selection.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved: reports/feature_selection.json")


if __name__ == "__main__":
    main()