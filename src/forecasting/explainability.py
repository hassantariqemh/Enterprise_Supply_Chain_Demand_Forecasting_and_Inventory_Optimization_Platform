"""
Explainable AI Module
Uses SHAP to explain the trained XGBoost demand forecasting model —
global feature importance and per-prediction breakdowns. Required for
the "Explainable Forecasts" evaluation criterion.
"""

import pandas as pd
import numpy as np
import shap
import mlflow
import matplotlib.pyplot as plt
import os

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

TARGET = "units_sold"
DROP_COLS = ["date", "product_id", "warehouse_id", TARGET]


def load_model_and_data():
    model = mlflow.xgboost.load_model("models:/demand_forecast_xgboost/latest")
    df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    return model, df, feature_cols


def compute_global_importance(model, X, feature_cols, output_dir="reports"):
    """Global feature importance via SHAP summary plot + ranked bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # summary (beeswarm) plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_plot.png", dpi=150)
    plt.close()

    # mean absolute SHAP value ranking
    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(importance["feature"].head(15)[::-1], importance["mean_abs_shap"].head(15)[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 15 Features Driving Demand Forecasts")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_bar.png", dpi=150)
    plt.close()

    print(f"Saved global explainability artifacts to {output_dir}/")
    print("\nTop 10 features by importance:")
    print(importance.head(10).to_string(index=False))

    return explainer, shap_values, importance


def explain_single_prediction(explainer, X, feature_cols, row_idx=0, output_dir="reports"):
    """Per-prediction explanation (waterfall plot) — why did the model predict this value?"""
    shap_values_single = explainer(X.iloc[[row_idx]])

    plt.figure()
    shap.plots.waterfall(shap_values_single[0], show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_waterfall_sample.png", dpi=150)
    plt.close()

    print(f"\nSaved single-prediction explanation to {output_dir}/shap_waterfall_sample.png")


def main():
    model, df, feature_cols = load_model_and_data()
    X = df[feature_cols].sample(n=min(5000, len(df)), random_state=42)  # sample for speed

    explainer, shap_values, importance = compute_global_importance(model, X, feature_cols)
    explain_single_prediction(explainer, X, feature_cols, row_idx=0)


if __name__ == "__main__":
    main()