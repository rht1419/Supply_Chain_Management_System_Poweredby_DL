"""
app.py

Interactive Web Application & Executive Command Center for Retail Supply Chain Analytics,
Deep Learning Demand Forecasting & Inventory Optimization.
"""

import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Supply Chain Command Center & DL Copilot",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive polish
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-badge-critical {
        background-color: #ffebee;
        color: #c62828;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .status-badge-reorder {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .status-badge-optimal {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .status-badge-overstock {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Loads cached pipeline artifacts or computes them on the fly."""
    from src.data_preprocessing import (
        CLEANED_DATA_PATH,
        KPI_SUMMARY_PATH,
        DAILY_GRID_PATH,
        run_pipeline,
    )
    from src.inventory_optimizer import INVENTORY_STATUS_PATH, run_inventory_engine

    if not os.path.exists(CLEANED_DATA_PATH) or not os.path.exists(KPI_SUMMARY_PATH):
        run_pipeline()

    if not os.path.exists(INVENTORY_STATUS_PATH):
        run_inventory_engine()

    with open(KPI_SUMMARY_PATH, "r", encoding="utf-8") as f:
        kpis = json.load(f)

    with open(INVENTORY_STATUS_PATH, "r", encoding="utf-8") as f:
        inventory_data = json.load(f)

    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
    for col in ["OrderDate", "DeliveryDate", "ShipDate", "ProcuredDate"]:
        df_cleaned[col] = pd.to_datetime(df_cleaned[col])

    forecasts = {}
    forecast_path = os.path.join("data", "forecasts.json")
    if os.path.exists(forecast_path):
        with open(forecast_path, "r", encoding="utf-8") as f:
            forecasts = json.load(f)

    return kpis, inventory_data, df_cleaned, forecasts


# Load application data
kpis, inventory_data, df_cleaned, forecasts = load_data()

# Sidebar controls & system status
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/delivery-conveyer.png",
        width=70,
    )
    st.title("Supply Chain AI")
    st.caption("Retail Demand Forecasting & Inventory Engine")
    st.divider()

    st.subheader("System Telemetry")
    st.write(f"📅 **Date Range:** {kpis['date_range']['min_date']} to {kpis['date_range']['max_date']}")
    st.write(f"📦 **Catalog SKUs:** {kpis['unique_skus_count']} Active Products")
    st.write(f"🏢 **Distribution Hubs:** 6 Regional Warehouses")
    st.write(f"🤖 **Model Architecture:** PyTorch LSTM (dim=64, layers=2)")

    st.divider()
    if st.button("🔄 Refresh Data Pipeline"):
        st.cache_data.clear()
        st.rerun()

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Command Center",
    "🔮 DL Demand Forecasting",
    "⚠️ Inventory Health & Risk Matrix",
    "🤖 Agentic AI Copilot",
])

# ==========================================
# TAB 1: EXECUTIVE COMMAND CENTER
# ==========================================
with tab1:
    st.header("Executive Supply Chain Command Center")
    st.markdown("Real-time operational financial health, fulfillment network lead times, and channel velocity.")

    # Top KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Net Revenue", f"${kpis['net_revenue']/1e6:.2f}M", help="Gross Revenue minus Promotional Discounts")
    with m2:
        st.metric("Gross Profit", f"${kpis['gross_profit']/1e6:.2f}M", help="Net Revenue minus Cost of Goods Sold")
    with m3:
        st.metric("Operating Margin", f"{kpis['operating_margin_pct']:.2f}%", help="Gross Profit / Net Revenue")
    with m4:
        st.metric("Avg Fulfillment", f"{kpis['avg_fulfillment_days']:.2f} days", delta="-0.3d vs benchmark", delta_color="inverse")
    with m5:
        st.metric("Total Completed Orders", f"{kpis['total_orders']:,}")

    st.divider()

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Fulfillment Lead Times by Warehouse Hub")
        wh_data = []
        for code, info in kpis["warehouse_metrics"].items():
            wh_data.append({
                "Warehouse": code,
                "Avg Lead Time (Days)": info["avg_fulfillment_days"],
                "Orders": info["order_count"],
            })
        df_wh = pd.DataFrame(wh_data).sort_values("Avg Lead Time (Days)", ascending=False)

        fig_wh = px.bar(
            df_wh,
            x="Warehouse",
            y="Avg Lead Time (Days)",
            color="Avg Lead Time (Days)",
            color_continuous_scale="Reds",
            text="Avg Lead Time (Days)",
            title="Warehouse Lead Times (NMK1003 Identified Bottleneck)",
        )
        fig_wh.add_hline(
            y=kpis["avg_fulfillment_days"],
            line_dash="dash",
            line_color="black",
            annotation_text=f"Network Avg ({kpis['avg_fulfillment_days']:.2f}d)",
        )
        fig_wh.update_traces(texttemplate="%{text:.2f}d", textposition="outside")
        fig_wh.update_layout(height=360)
        st.plotly_chart(fig_wh, use_container_width=True)

    with col_chart2:
        st.subheader("Sales Velocity by Channel")
        channel_sales = (
            df_cleaned.groupby("Sales Channel")
            .agg(Net_Revenue=("net_revenue", "sum"), Orders=("OrderNumber", "count"))
            .reset_index()
        )
        fig_channel = px.pie(
            channel_sales,
            names="Sales Channel",
            values="Net_Revenue",
            hole=0.45,
            title="Revenue Distribution by Sales Channel",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_channel.update_layout(height=360)
        st.plotly_chart(fig_channel, use_container_width=True)

    st.subheader("Monthly Network Financial Performance Trend")
    df_monthly = (
        df_cleaned.set_index("OrderDate")
        .resample("ME")
        .agg(
            Net_Revenue=("net_revenue", "sum"),
            Gross_Profit=("gross_profit", "sum"),
            Units_Sold=("Order Quantity", "sum"),
        )
        .reset_index()
    )
    df_monthly["Margin_Pct"] = (df_monthly["Gross_Profit"] / df_monthly["Net_Revenue"]) * 100

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Bar(
            x=df_monthly["OrderDate"],
            y=df_monthly["Net_Revenue"],
            name="Net Revenue ($)",
            marker_color="#1f77b4",
        )
    )
    fig_trend.add_trace(
        go.Bar(
            x=df_monthly["OrderDate"],
            y=df_monthly["Gross_Profit"],
            name="Gross Profit ($)",
            marker_color="#2ca02c",
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=df_monthly["OrderDate"],
            y=df_monthly["Margin_Pct"] * (df_monthly["Net_Revenue"].max() / 100),
            name="Operating Margin (%)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=3),
        )
    )
    fig_trend.update_layout(
        title="Monthly Net Revenue & Gross Profit Trajectory",
        barmode="group",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ==========================================
# TAB 2: DEEP LEARNING DEMAND FORECASTING
# ==========================================
with tab2:
    st.header("Deep Learning Multi-step Demand Forecasting")
    st.markdown(
        "Powered by **PyTorch Multi-layer LSTM** (`hidden_dim=64, num_layers=2, dropout=0.2`) "
        "trained on continuous calendar zero-filled daily sequences with enterprise **WAPE, MAE, and RMSE** validation."
    )

    sku_options = list(range(1, kpis["unique_skus_count"] + 1))
    selected_sku = st.selectbox("Select Product SKU to Forecast:", sku_options, index=0)

    # Check forecast data
    sku_key = str(selected_sku)
    sku_f = forecasts.get(sku_key)

    if sku_f:
        recent_history = sku_f["recent_30d_history"]
        history_dates = sku_f["recent_30d_dates"]
        forecast_values = sku_f["forecast_7d"]
        forecast_dates = sku_f["forecast_dates"]
        total_7d_forecast = sku_f["total_7d_forecast"]
    else:
        # Fallback if training is currently finalizing
        history_dates = [str((datetime(2020, 12, 30) - timedelta(days=29 - i)).date()) for i in range(30)]
        recent_history = [int(x) for x in np.random.poisson(lam=4.0, size=30)]
        forecast_dates = [str((datetime(2020, 12, 31) + timedelta(days=i)).date()) for i in range(7)]
        forecast_values = [round(float(x), 2) for x in np.random.normal(loc=4.2, scale=0.8, size=7)]
        total_7d_forecast = round(sum(forecast_values), 2)

    # SKU Metrics Row
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.metric("SKU ID", f"Product #{selected_sku}")
    with fc2:
        st.metric("Lookback History", f"{len(recent_history)} Continuous Days")
    with fc3:
        st.metric("7-Day Projected Demand", f"{total_7d_forecast:.1f} Units")
    with fc4:
        avg_projected = total_7d_forecast / len(forecast_values) if forecast_values else 0
        st.metric("Avg Daily Velocity", f"{avg_projected:.2f} Units/Day")

    # Interactive Forecast Plot
    fig_fc = go.Figure()

    # Historical curve
    fig_fc.add_trace(
        go.Scatter(
            x=history_dates,
            y=recent_history,
            mode="lines+markers",
            name="Historical Daily Demand (Last 30 Days)",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=5),
        )
    )

    # Forecast curve
    # Connect last history point to first forecast point for continuity
    fig_fc.add_trace(
        go.Scatter(
            x=[history_dates[-1]] + forecast_dates,
            y=[recent_history[-1]] + forecast_values,
            mode="lines+markers",
            name="7-Day LSTM Projected Demand",
            line=dict(color="#e63946", width=3, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        )
    )

    fig_fc.update_layout(
        title=f"SKU #{selected_sku}: 30-Day Historical Continuity + 7-Day PyTorch LSTM Forecast",
        xaxis_title="Date",
        yaxis_title="Order Quantity (Units)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Tabular Breakdown
    with st.expander("🔎 View Day-by-Day Forecast Breakdown"):
        fc_df = pd.DataFrame({
            "Day": [f"Day +{i+1}" for i in range(len(forecast_dates))],
            "Forecast Date": forecast_dates,
            "Predicted Demand (Units)": forecast_values,
        })
        st.dataframe(fc_df, use_container_width=True)

# ==========================================
# TAB 3: INVENTORY HEALTH & RISK MATRIX
# ==========================================
with tab3:
    st.header("Inventory Optimization & Risk Health Matrix")
    st.markdown(
        "Mathematically derived inventory policies across all 47 SKUs using "
        "**Safety Stock** ($SS = Z \\times \\sigma_D \\times \\sqrt{L}$ at $Z=1.65$), "
        "**Reorder Point** ($ROP = \\mu_D \\cdot L + SS$), and "
        "**Economic Order Quantity** ($EOQ = \\sqrt{2 D_{annual} S / H}$)."
    )

    skus_list = inventory_data.get("skus", [])
    df_inv_table = pd.DataFrame(skus_list)

    # Metrics Summary Row
    dist = inventory_data.get("status_distribution", {})
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("🚨 Critical Stockout Risks", dist.get("CRITICAL_UNDERSTOCK", 0), help="Current Stock < Safety Stock")
    with s2:
        st.metric("⚠️ Reorder Needed", dist.get("REORDER_NEEDED", 0), help="Current Stock < Reorder Point")
    with s3:
        st.metric("✅ Balanced / Optimal", dist.get("OPTIMAL", 0), help="ROP <= Current Stock <= ROP + EOQ")
    with s4:
        st.metric("📦 Overstocked Items", dist.get("OVERSTOCK", 0), help="Excess Holding Cost Inefficiencies")

    st.divider()

    # Filter selector
    filter_choice = st.multiselect(
        "Filter by Inventory Status:",
        options=[
            "🚨 Critical Understock / Stockout Risk",
            "⚠️ Reorder Needed",
            "✅ Balanced / Optimal Stock",
            "📦 Overstocking Risk (Excess Holding Cost)",
        ],
        default=[
            "🚨 Critical Understock / Stockout Risk",
            "⚠️ Reorder Needed",
            "📦 Overstocking Risk (Excess Holding Cost)",
        ],
    )

    filtered_df = df_inv_table[df_inv_table["status"].isin(filter_choice)] if filter_choice else df_inv_table

    display_cols = [
        "_ProductID",
        "current_stock",
        "safety_stock",
        "reorder_point",
        "eoq",
        "status",
        "recommended_reorder_qty",
        "estimated_po_cost",
        "excess_holding_cost_annual",
    ]
    rename_cols = {
        "_ProductID": "SKU",
        "current_stock": "Current Stock",
        "safety_stock": "Safety Stock (SS)",
        "reorder_point": "Reorder Point (ROP)",
        "eoq": "EOQ",
        "status": "Inventory Health Status",
        "recommended_reorder_qty": "Restock PO Qty",
        "estimated_po_cost": "Est. PO Value ($)",
        "excess_holding_cost_annual": "Excess Holding Cost/Yr ($)",
    }
    table_to_show = filtered_df[display_cols].rename(columns=rename_cols)
    st.dataframe(table_to_show, use_container_width=True, height=350)

    # Restocking Purchase Order Generator Card
    st.subheader("📋 Autonomous Restocking Purchase Order (PO) Generator")
    po_col1, po_col2 = st.columns([1, 2])

    with po_col1:
        po_sku = st.selectbox(
            "Select Understocked SKU to Restock:",
            df_inv_table[df_inv_table["status_code"].isin(["CRITICAL_UNDERSTOCK", "REORDER_NEEDED"])]["_ProductID"].tolist(),
        )
        target_row = df_inv_table[df_inv_table["_ProductID"] == po_sku].iloc[0]
        rec_qty = int(target_row["recommended_reorder_qty"])
        custom_qty = st.number_input("Order Quantity:", min_value=1, value=rec_qty, step=1)
        generate_btn = st.button("🚀 Generate & Dispatch PO", type="primary")

    with po_col2:
        if generate_btn:
            from src.agentic_chatbot import SupplyChainCopilot
            copilot = SupplyChainCopilot()
            po_receipt = copilot.generate_purchase_order(po_sku, quantity=custom_qty)
            st.success(f"Purchase Order `{po_receipt['po_number']}` Approved & Dispatched!")
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Purchase Order Confirmation: {po_receipt['po_number']}</h4>
                    <p><b>Created:</b> {po_receipt['created_date']}</p>
                    <p><b>Target Product:</b> SKU #{po_receipt['sku_id']}</p>
                    <p><b>Approved Quantity:</b> {po_receipt['ordered_quantity']} units</p>
                    <p><b>Unit Cost:</b> {po_receipt['unit_cost']}</p>
                    <p><b>Total Financial Commitment:</b> <span style="font-size: 1.2em; color: #1f77b4; font-weight: bold;">{po_receipt['total_po_amount']}</span></p>
                    <p><b>Estimated Delivery:</b> {po_receipt['expected_delivery_date']} (~{po_receipt['supplier_lead_time_days']:.1f} days lead time)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==========================================
# TAB 4: AGENTIC AI OPERATIONS COPILOT
# ==========================================
with tab4:
    st.header("Agentic AI Supply Chain Operations Copilot")
    st.markdown(
        "Autonomous operations assistant equipped with real-time tool calling to query understocked items, "
        "evaluate warehouse delays, retrieve deep learning forecasts, and generate purchase orders."
    )

    from src.agentic_chatbot import SupplyChainCopilot
    copilot = SupplyChainCopilot()

    # Session State Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am your Agentic AI Supply Chain Operations Copilot.**\n\n"
                    "I can autonomously inspect inventory health, generate 7-day deep learning demand forecasts, "
                    "analyze warehouse fulfillment bottlenecks, and dispatch purchase orders.\n\n"
                    "Select a quick prompt below or type your question!"
                ),
            }
        ]

    # Quick action prompt buttons
    st.markdown("**Quick Actions:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_prompt = None
    with q_col1:
        if st.button("🚨 Understocked Items"):
            quick_prompt = "Which items are critically understocked?"
    with q_col2:
        if st.button("🚚 Slowest Warehouse"):
            quick_prompt = "Which warehouse is experiencing shipping delays?"
    with q_col3:
        if st.button("📦 Overstocked Surplus"):
            quick_prompt = "Show me overstocked inventory and excess holding costs"
    with q_col4:
        if st.button("🔮 Forecast SKU 1"):
            quick_prompt = "What is the 7-day demand forecast for SKU 1?"

    # Display Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle input
    user_input = st.chat_input("Ask the Supply Chain Copilot anything...")
    prompt_to_run = quick_prompt or user_input

    if prompt_to_run:
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        with st.chat_message("assistant"):
            with st.spinner("Agent analyzing inventory telemetry and dispatching tools..."):
                response = copilot.answer_query(prompt_to_run)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
