"""
Executive Dashboard
Streamlit UI that consumes the FastAPI backend to display forecast
accuracy, inventory health, warehouse utilization, alerts, and
procurement recommendations.
"""

import streamlit as st
import pandas as pd
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Supply Chain Command Center", layout="wide")
st.title("📦 Enterprise Supply Chain — Executive Dashboard")


API_KEY = "dev-secret-key-change-in-production"

@st.cache_data(ttl=30)
def fetch(endpoint, params=None):
    try:
        r = requests.get(
            f"{API_BASE}{endpoint}",
            params=params,
            headers={"X-API-Key": API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to reach API at {endpoint}: {e}")
        return None


# ---------- Top KPIs ----------
summary = fetch("/dashboard/summary")

if summary:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Product-Warehouse Pairs", summary["total_product_warehouse_pairs"])
    col2.metric("Estimated Fill Rate", f"{summary['estimated_fill_rate']*100:.1f}%")
    col3.metric("Transfer Recommendations", summary["total_transfer_recommendations"])
    critical = summary["stock_status_breakdown"].get("CRITICAL_STOCKOUT_RISK", 0)
    col4.metric("Critical Stockout Alerts", critical, delta_color="inverse")
    sp = summary.get("stockout_probability_summary")
    if sp:
        col5, col6 = st.columns(2)
        col5.metric("Avg Stockout Probability", f"{sp['avg_stockout_probability']*100:.1f}%")
        col6.metric("High-Risk Product-Warehouse Pairs", sp["high_risk_pairs_count"])

        # ---------- Forecast Accuracy + Warehouse Utilization ----------
    st.divider()
    col_x, col_y = st.columns(2)

    with col_x:
        st.subheader("Forecast Accuracy")
        fa = summary.get("forecast_accuracy")
        if fa and fa.get("accuracy_pct") is not None:
            st.metric(f"{fa['model_name']} Accuracy", f"{fa['accuracy_pct']}%")
            st.caption(f"MAE: {fa['avg_mae']:.2f} | RMSE: {fa['avg_rmse']:.2f} | MAPE: {fa['avg_mape']:.3f}")
        else:
            st.info("Run export_metrics.py to populate forecast accuracy.")

    with col_y:
        st.subheader("Warehouse Utilization")
        wu = summary.get("warehouse_utilization")
        if wu:
            wu_df = pd.DataFrame(wu)
            wu_df["utilization_pct"] = (wu_df["utilization_pct"] * 100).round(1)
            st.dataframe(wu_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run capacity_and_drift.py to populate warehouse utilization.")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Inventory Health Breakdown")
        status_df = pd.DataFrame(
            list(summary["stock_status_breakdown"].items()),
            columns=["Status", "Count"]
        )
        st.bar_chart(status_df.set_index("Status"))

    with col_b:
        st.subheader("Alert Severity Breakdown")
        alert_df = pd.DataFrame(
            list(summary["alert_severity_breakdown"].items()),
            columns=["Severity", "Count"]
        )
        st.bar_chart(alert_df.set_index("Severity"))

st.divider()

# ---------- Alert Center ----------
st.subheader("🚨 Alert Center")
severity_filter = st.selectbox("Filter by severity", ["All", "HIGH", "MEDIUM", "LOW"])
params = {} if severity_filter == "All" else {"severity": severity_filter}
alerts = fetch("/alerts", params=params)

if alerts:
    st.dataframe(pd.DataFrame(alerts), use_container_width=True)
else:
    st.info("No alerts to display.")

st.divider()

# ---------- Inventory Recommendations ----------
st.subheader("📋 Inventory Recommendations")
warehouse_filter = st.text_input("Filter by warehouse_id (optional, e.g. W001)")
status_filter = st.selectbox(
    "Filter by stock status",
    ["All", "CRITICAL_STOCKOUT_RISK", "REORDER_NEEDED", "OVERSTOCK", "HEALTHY"]
)

params = {}
if warehouse_filter:
    params["warehouse_id"] = warehouse_filter
if status_filter != "All":
    params["status"] = status_filter

inventory = fetch("/inventory/recommendations", params=params)
if inventory:
    st.dataframe(pd.DataFrame(inventory), use_container_width=True)

st.divider()

# ---------- Transfer Recommendations ----------
st.subheader("🔄 Inter-Warehouse Transfer Recommendations")
transfers = fetch("/inventory/transfers")
if transfers:
    st.dataframe(pd.DataFrame(transfers), use_container_width=True)
else:
    st.info("No transfer recommendations available.")

# ---------- Procurement Allocation Recommendations ----------
st.divider()
st.subheader("📦 Procurement Allocation Recommendations")
allocation = fetch("/inventory/allocation")
if allocation:
    alloc_df = pd.DataFrame(allocation)
    st.dataframe(alloc_df[["product_id", "warehouse_id", "current_stock", "allocation_status", "allocated_order_qty"]],
                 use_container_width=True)
else:
    st.info("No allocation data available.")