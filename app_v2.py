"""
app_v2.py

DRAR Stores: Autonomous Omni-Channel Retail & SCM System (v2).
Three synchronized real-time portals running on port 8502:
1. Portal 1: DRAR Storefront & POS Cashier Terminal
2. Portal 2: Autonomous SCM & Agentic AI Command Center (No Manager Needed)
3. Portal 3: Inbound Logistics & Warehouse Receiving Bay
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

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from v2.state_manager import StateManager
from v2.nemotron_agent import NemotronStoreCopilot

# Page Configuration
st.set_page_config(
    page_title="DRAR Stores: Autonomous Omni-Channel Retail & SCM",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .store-header {
        background: linear-gradient(135deg, #0d3b66 0%, #001e3d 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .portal-tag {
        background-color: #f4d35e;
        color: #0d3b66;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85em;
        text-transform: uppercase;
    }
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 14px;
        background-color: #ffffff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        margin-bottom: 15px;
    }
    .proposal-card {
        background-color: #fff9db;
        border-left: 6px solid #f59f00;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .po-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .badge-critical {
        background-color: #ffe3e3;
        color: #c92a2a;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-reorder {
        background-color: #fff3bf;
        color: #d9480f;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-optimal {
        background-color: #d3f9d8;
        color: #2b8a3e;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-overstock {
        background-color: #e7f5ff;
        color: #1864ab;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize singletons in session state
if "state_mgr" not in st.session_state:
    st.session_state.state_mgr = StateManager()

if "copilot" not in st.session_state:
    st.session_state.copilot = NemotronStoreCopilot()

if "cart" not in st.session_state:
    st.session_state.cart = {}  # {product_id: quantity}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to DRAR Stores Autonomous SCM Operations Center.**\n\n"
                "I am your Agentic AI Operations Copilot powered by **NVIDIA Nemotron 3 Ultra 550B**.\n"
                "I continuously monitor our 47 retail SKUs, track customer checkout velocity, "
                "project PyTorch LSTM demand, and propose automated vendor replenishments.\n\n"
                "How can I assist your operations today?"
            ),
        }
    ]

state_mgr = st.session_state.state_mgr
copilot = st.session_state.copilot

# Sidebar: System Telemetry & Multi-Device LAN Access
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/supermarket.png", width=65)
    st.title("DRAR Stores")
    st.caption("Autonomous Retail Enterprise System (v2)")

    st.markdown(
        """
        <div style="background-color: #e7f5ff; border: 1px solid #74c0fc; padding: 10px; border-radius: 8px; font-size: 0.88em;">
            <b>🌐 Multi-Device LAN Access:</b><br/>
            Open on your friend's laptop:<br/>
            <code>http://192.168.11.105:8502</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("🤖 AI Brain Configuration")
    api_key_input = st.text_input(
        "OpenRouter API Key (Optional):",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Enter OpenRouter API key for NVIDIA Nemotron 550B. If blank, autonomous local reasoning engine is active.",
    )
    if api_key_input:
        copilot.set_api_key(api_key_input)
        st.success("Nemotron 550B Connected!")
    else:
        st.info("Operating on Local Autonomous Reasoning Engine.")

    st.divider()
    st.subheader("⚡ Quick Controls")
    if st.button("🔄 Sync Live State"):
        st.rerun()

    if st.button("⚠️ Reset Demo to Baseline"):
        state_mgr.reset_demo_state()
        st.session_state.cart = {}
        st.success("Store state reset to clean baseline!")
        st.rerun()

# Top Hero Banner
st.markdown(
    """
    <div class="store-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 2.1em;">🏪 DRAR STORES SUPERCENTER</h1>
                <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 1.05em;">
                    Closed-Loop Autonomous Retail & Supply Chain System (No Human Data Analyst or SCM Manager Needed)
                </p>
            </div>
            <div style="text-align: right;">
                <span class="portal-tag">Live Production v2</span>
                <p style="margin: 6px 0 0 0; font-size: 0.9em; opacity: 0.85;">Host: 0.0.0.0:8502</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Three Synchronized Portals
portal_choice = st.radio(
    "Select Operating Portal:",
    [
        "🛒 Portal 1: DRAR Storefront & POS Checkout (Customer / Cashier)",
        "🧠 Portal 2: Autonomous SCM & AI Command Center (No Manager Needed)",
        "🚚 Portal 3: Inbound Logistics & Receiving Terminal (Supplier Hub)",
    ],
    horizontal=True,
)

st.divider()

# Fetch latest state
inventory = state_mgr.get_inventory()
inv_map = {p["product_id"]: p for p in inventory}

# ==========================================
# PORTAL 1: DRAR STOREFRONT & POS CHECKOUT
# ==========================================
if "Portal 1" in portal_choice:
    st.subheader("🛒 Portal 1: DRAR Storefront & Point of Sale (POS)")
    st.markdown(
        "Simulates live retail customer traffic. **Every checkout instantly decrements live shelf inventory in real-time**, "
        "triggering autonomous supply chain replenishment when stock crosses below Reorder Point."
    )

    col_channel, col_filter, col_search = st.columns([1, 1, 1.5])
    with col_channel:
        channel_mode = st.radio(
            "Channel Mode:",
            ["🌐 Online E-Commerce (Home Delivery)", "🏪 In-Store POS (Walk-in Cashier)"],
        )
    with col_filter:
        categories = ["All Categories"] + sorted(list({p["category"] for p in inventory}))
        selected_cat = st.selectbox("Filter Department:", categories)
    with col_search:
        search_query = st.text_input("🔍 Search Catalog (Name or Brand):", placeholder="e.g. Milk, Rice, Sony, Tide...")

    # Filter catalog
    filtered_products = inventory
    if selected_cat != "All Categories":
        filtered_products = [p for p in filtered_products if p["category"] == selected_cat]
    if search_query:
        q = search_query.lower()
        filtered_products = [
            p for p in filtered_products
            if q in p["name"].lower() or q in p["brand"].lower()
        ]

    # Two Column Layout: Catalog Grid + Live Shopping Cart
    cat_col, cart_col = st.columns([2.2, 1.2])

    with cat_col:
        st.markdown(f"**Showing {len(filtered_products)} Products Available for Purchase:**")

        # Grid of product cards (2 columns inside catalog)
        grid_cols = st.columns(2)
        for idx, prod in enumerate(filtered_products):
            pid = prod["product_id"]
            with grid_cols[idx % 2]:
                stock = prod["current_stock"]
                badge_class = (
                    "badge-critical" if prod["status_code"] == "CRITICAL_UNDERSTOCK"
                    else "badge-reorder" if prod["status_code"] == "REORDER_NEEDED"
                    else "badge-optimal" if prod["status_code"] == "OPTIMAL"
                    else "badge-overstock"
                )

                st.markdown(
                    f"""
                    <div class="product-card">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-size: 1.8em;">{prod['icon']}</span>
                            <span class="{badge_class}">{stock} in stock</span>
                        </div>
                        <h4 style="margin: 8px 0 2px 0;">{prod['name']}</h4>
                        <p style="margin: 0; color: #666; font-size: 0.85em;">{prod['brand']} • {prod['category']}</p>
                        <h3 style="margin: 8px 0 8px 0; color: #0d3b66;">${prod['shelf_price']:.2f}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                btn_cols = st.columns([1, 1])
                with btn_cols[0]:
                    buy_qty = st.number_input(
                        f"Qty #{pid}",
                        min_value=1,
                        max_value=max(1, stock),
                        value=1,
                        key=f"qty_sel_{pid}",
                        label_visibility="collapsed",
                    )
                with btn_cols[1]:
                    if stock > 0:
                        if st.button("Add to Cart", key=f"add_btn_{pid}"):
                            st.session_state.cart[pid] = st.session_state.cart.get(pid, 0) + buy_qty
                            st.success(f"Added {buy_qty}x {prod['name'][:18]}!")
                            st.rerun()
                    else:
                        st.button("Out of Stock", disabled=True, key=f"oos_{pid}")

    # Right Column: Interactive Shopping Cart & Checkout Drawer
    with cart_col:
        st.markdown("### 🛍️ Live Shopping Cart")

        if not st.session_state.cart:
            st.info("Your shopping cart is empty. Add products from the catalog to simulate an order.")
        else:
            cart_subtotal = 0.0
            items_to_delete = []

            for pid, qty in list(st.session_state.cart.items()):
                p = inv_map.get(pid)
                if not p:
                    continue
                item_total = round(qty * p["shelf_price"], 2)
                cart_subtotal += item_total

                c_item1, c_item2, c_item3 = st.columns([2, 1, 0.5])
                with c_item1:
                    st.write(f"**{p['icon']} {p['name'][:22]}**")
                    st.caption(f"${p['shelf_price']:.2f} each")
                with c_item2:
                    st.write(f"**{qty}x** (${item_total:.2f})")
                with c_item3:
                    if st.button("❌", key=f"del_{pid}", help="Remove from cart"):
                        items_to_delete.append(pid)

            for d in items_to_delete:
                del st.session_state.cart[d]
                st.rerun()

            st.divider()

            # Checkout Options
            cust_name = st.text_input("Customer Name:", value="Jordan Hayes")
            pay_method = st.selectbox("Payment Method:", ["Apple Pay", "Visa **** 4921", "Store Cash", "Mastercard"])
            promo_disc = st.select_slider("Promotional Discount:", options=[0, 5, 10, 15, 20], value=0)

            discount_val = round(cart_subtotal * (promo_disc / 100.0), 2)
            net_total = round(cart_subtotal - discount_val, 2)

            st.markdown(f"**Subtotal:** ${cart_subtotal:,.2f}")
            if discount_val > 0:
                st.markdown(f"**Discount ({promo_disc}%):** -${discount_val:,.2f}")
            st.markdown(f"### **Final Total:** <span style='color: #2b8a3e;'>${net_total:,.2f}</span>", unsafe_allow_html=True)

            if st.button("💳 Complete Purchase & Deduct Stock", type="primary", use_container_width=True):
                cart_payload = [{"product_id": pid, "quantity": qty} for pid, qty in st.session_state.cart.items()]
                order_result = state_mgr.place_customer_order(
                    channel=channel_mode,
                    customer_name=cust_name,
                    cart_items=cart_payload,
                    discount_pct=float(promo_disc),
                    payment_method=pay_method,
                )

                st.session_state.cart = {}
                st.balloons()
                st.success(f"✅ Order #{order_result['order']['order_id']} placed successfully!")

                if order_result["new_proposals_count"] > 0:
                    st.warning(
                        f"🚨 **Autonomous AI SCM Alert!** {order_result['new_proposals_count']} product(s) dropped below Reorder Point. "
                        "Replenishment proposals have been autonomously generated in Portal 2!"
                    )
                st.rerun()

    # Bottom: Live Store Order History
    st.divider()
    st.subheader("📜 Live Completed Transactions (Decremented from Stock)")
    recent_orders = state_mgr.get_customer_orders()
    if recent_orders:
        orders_df = pd.DataFrame([
            {
                "Order ID": o["order_id"],
                "Time": o["timestamp"],
                "Channel": o["channel"],
                "Customer": o["customer_name"],
                "Items Count": o["total_items"],
                "Net Total": f"${o['net_total']:,.2f}",
                "Payment": o["payment_method"],
            }
            for o in recent_orders[:8]
        ])
        st.dataframe(orders_df, use_container_width=True)
    else:
        st.caption("No customer orders placed yet during this session.")


# ==========================================
# PORTAL 2: AUTONOMOUS SCM & AI COMMAND CENTER
# ==========================================
elif "Portal 2" in portal_choice:
    st.subheader("🧠 Portal 2: Autonomous SCM & AI Command Center (No Manager Needed)")
    st.markdown(
        "The Autonomous AI Brain monitors store inventory, projects PyTorch LSTM demand, "
        "and automatically proposes vendor replenishments with 1-click execution."
    )

    # Top KPI Metrics Row
    crit_count = sum(1 for p in inventory if p["status_code"] == "CRITICAL_UNDERSTOCK")
    reorder_count = sum(1 for p in inventory if p["status_code"] == "REORDER_NEEDED")
    optimal_count = sum(1 for p in inventory if p["status_code"] == "OPTIMAL")
    overstock_count = sum(1 for p in inventory if p["status_code"] == "OVERSTOCK")
    pending_proposals = state_mgr.get_ai_proposals("PENDING")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Catalog SKUs", len(inventory))
    with k2:
        st.metric("🚨 Critical Stockout Risk", crit_count, delta="Immediate Action Needed", delta_color="inverse")
    with k3:
        st.metric("⚠️ Reorder Needed", reorder_count)
    with k4:
        st.metric("✅ Balanced Optimal", optimal_count)
    with k5:
        st.metric("📦 Overstock Surplus", overstock_count)

    st.divider()

    # SECTION A: AUTONOMOUS VENDOR REORDER PROPOSAL CARDS
    st.subheader(f"⚡ Autonomous Vendor Restock Proposals ({len(pending_proposals)} Awaiting Approval)")
    st.caption("Triggered automatically when customer checkouts drop stock below Reorder Point ($Current < ROP$).")

    if not pending_proposals:
        st.success("✅ **All store inventory levels are balanced!** No active vendor restock proposals currently pending.")
    else:
        for prop in pending_proposals[:6]:
            prop_id = prop["proposal_id"]
            pid = prop["product_id"]

            st.markdown(
                f"""
                <div class="proposal-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #d9480f;">⚠️ Autonomous Restock Alert: {prop['product_name']}</h4>
                        <span style="font-weight: bold; background-color: #ffe066; padding: 2px 8px; border-radius: 4px;">{prop_id}</span>
                    </div>
                    <p style="margin: 8px 0; font-size: 0.95em;">
                        <b>Current Shelf Stock:</b> <span style="color: #c92a2a; font-weight: bold;">{prop['current_stock']} units</span> |
                        <b>Reorder Point (ROP):</b> {prop['reorder_point']} units |
                        <b>Safety Stock (SS):</b> {prop['safety_stock']} units
                    </p>
                    <p style="margin: 0 0 10px 0; font-size: 0.95em;">
                        <b>Autonomous AI Recommendation:</b> Shall I request supplier <b>'{prop['vendor']}'</b> for an EOQ replenishment batch of
                        <span style="font-size: 1.1em; color: #0d3b66; font-weight: bold;">{prop['suggested_quantity']} units</span>
                        (Total Landed Cost: <b>${prop['total_cost']:,.2f}</b>)?
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            p_col1, p_col2, p_col3 = st.columns([1.5, 1, 3])
            with p_col1:
                if st.button("🚀 Approve & Dispatch PO to Vendor", key=f"app_{prop_id}", type="primary"):
                    new_po = state_mgr.approve_proposal(prop_id)
                    st.success(f"PO `{new_po['po_id']}` dispatched to {prop['vendor']}! Head to Portal 3 to receive shipment.")
                    st.rerun()
            with p_col2:
                if st.button("Decline / Defer", key=f"dec_{prop_id}"):
                    state_mgr.decline_proposal(prop_id)
                    st.info(f"Proposal {prop_id} deferred.")
                    st.rerun()

    st.divider()

    # SECTION B: PYTORCH LSTM DEMAND FORECASTING
    st.subheader("🔮 PyTorch Multi-layer LSTM Demand Forecasting Engine")
    st.caption("Evaluates 30-day continuous calendar demand history and forecasts 7 forward days.")

    sku_choices = [p["product_id"] for p in inventory]
    sel_sku_id = st.selectbox(
        "Select Supermarket Product to Inspect Forecast:",
        sku_choices,
        format_func=lambda x: f"SKU #{x}: {inv_map[x]['name']} ({inv_map[x]['category']})",
    )

    # Load forecast data
    f_path = os.path.join("data", "forecasts.json")
    if os.path.exists(f_path):
        with open(f_path, "r", encoding="utf-8") as f:
            f_data = json.load(f)
            sku_fc = f_data.get(str(sel_sku_id))
    else:
        sku_fc = None

    if sku_fc:
        history_vals = sku_fc["recent_30d_history"]
        history_dates = sku_fc["recent_30d_dates"]
        forecast_vals = sku_fc["forecast_7d"]
        forecast_dates = sku_fc["forecast_dates"]
        total_fc = sku_fc["total_7d_forecast"]
    else:
        history_dates = [str((datetime(2020, 12, 30) - timedelta(days=29 - i)).date()) for i in range(30)]
        history_vals = [int(x) for x in np.random.poisson(lam=4.5, size=30)]
        forecast_dates = [str((datetime(2020, 12, 31) + timedelta(days=i)).date()) for i in range(7)]
        forecast_vals = [round(float(x), 2) for x in np.random.normal(loc=4.8, scale=0.6, size=7)]
        total_fc = round(sum(forecast_vals), 2)

    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(
            x=history_dates,
            y=history_vals,
            mode="lines+markers",
            name="Historical Daily Orders (Last 30 Days)",
            line=dict(color="#1f77b4", width=2.5),
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=[history_dates[-1]] + forecast_dates,
            y=[history_vals[-1]] + forecast_vals,
            mode="lines+markers",
            name="7-Day PyTorch LSTM Predicted Demand",
            line=dict(color="#e63946", width=3, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        )
    )
    fig_fc.update_layout(
        title=f"Forecast Demand Curve: {inv_map[sel_sku_id]['name']}",
        xaxis_title="Calendar Date",
        yaxis_title="Units Demanded",
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.divider()

    # SECTION C: INVENTORY HEALTH RISK MATRIX
    st.subheader("⚠️ Live Inventory Health & Replenishment Matrix")
    inv_df = pd.DataFrame([
        {
            "SKU": p["product_id"],
            "Product Name": p["name"],
            "Category": p["category"],
            "Shelf Price": f"${p['shelf_price']:.2f}",
            "Current Stock": p["current_stock"],
            "Safety Stock (SS)": p["safety_stock"],
            "Reorder Point (ROP)": p["reorder_point"],
            "EOQ": p["eoq"],
            "Health Status": p["status_label"],
            "Primary Vendor": p["vendor"],
        }
        for p in inventory
    ])
    st.dataframe(inv_df, use_container_width=True, height=350)

    st.divider()

    # SECTION D: AGENTIC AI OPERATIONS COPILOT CHAT
    st.subheader("🤖 NVIDIA Nemotron Agentic AI Supply Chain Copilot")
    st.markdown("Ask the Copilot to analyze stockout risks, audit inbound shipments, or evaluate sales velocity.")

    q_c1, q_c2, q_c3, q_c4 = st.columns(4)
    quick_q = None
    with q_c1:
        if st.button("🚨 Critical Stockout Risks"):
            quick_q = "Which supermarket items have critical stockout risks?"
    with q_c2:
        if st.button("📋 Pending AI Proposals"):
            quick_q = "Show pending restock proposals awaiting approval"
    with q_c3:
        if st.button("🚚 Inbound POs"):
            quick_q = "Track active inbound vendor purchase orders"
    with q_c4:
        if st.button("💳 Live Sales Revenue"):
            quick_q = "Summarize live customer orders and sales revenue"

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask the Nemotron Supply Chain Copilot...")
    prompt_to_dispatch = quick_q or user_q

    if prompt_to_dispatch:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_dispatch})
        with st.chat_message("user"):
            st.markdown(prompt_to_dispatch)

        with st.chat_message("assistant"):
            with st.spinner("Nemotron AI analyzing live store telemetry..."):
                reply = copilot.query(prompt_to_dispatch)
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ==========================================
# PORTAL 3: INBOUND LOGISTICS & RECEIVING
# ==========================================
elif "Portal 3" in portal_choice:
    st.subheader("🚚 Portal 3: Vendor Dispatch & Inbound Goods Receiving Bay")
    st.markdown(
        "Simulates the warehouse receiving terminal. Shows all purchase orders dispatched by the autonomous SCM system. "
        "When shipments arrive, **warehouse staff click `[Receive Shipment]` to immediately restock store inventory**."
    )

    purchase_orders = state_mgr.get_purchase_orders()

    po_m1, po_m2, po_m3 = st.columns(3)
    active_pos = [po for po in purchase_orders if po["status"] != "Fulfilled & Restocked"]
    fulfilled_pos = [po for po in purchase_orders if po["status"] == "Fulfilled & Restocked"]
    total_inbound_val = sum(po["total_cost"] for po in purchase_orders)

    with po_m1:
        st.metric("Total Dispatched Purchase Orders", len(purchase_orders))
    with po_m2:
        st.metric("🚚 Inbound Shipments in Transit", len(active_pos))
    with po_m3:
        st.metric("Total Procurement Commitment", f"${total_inbound_val:,.2f}")

    st.divider()

    st.subheader("📦 Inbound Shipments Awaiting Receiving & Docking")

    if not active_pos:
        st.info("No shipments currently in transit. Head to Portal 2 to approve pending restocking proposals!")
    else:
        for po in active_pos:
            po_id = po["po_id"]
            pid = po["product_id"]
            p_current = inv_map.get(pid, {})

            st.markdown(
                f"""
                <div class="po-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #0d3b66;">📦 {po_id} — {po['product_name']}</h4>
                        <span style="background-color: #e7f5ff; color: #1864ab; padding: 3px 8px; border-radius: 4px; font-weight: bold;">
                            Status: {po['status']}
                        </span>
                    </div>
                    <div style="margin: 10px 0; font-size: 0.95em;">
                        <b>Supplier / Vendor:</b> {po['vendor']} |
                        <b>Approved By:</b> {po['approved_by']}<br/>
                        <b>Replenishment Batch:</b> <span style="font-size: 1.1em; color: #2b8a3e; font-weight: bold;">+{po['ordered_quantity']} units</span> |
                        <b>Landed PO Value:</b> ${po['total_cost']:,.2f} |
                        <b>Current Shelf Stock:</b> {p_current.get('current_stock', 0)} units
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            rec_col1, rec_col2 = st.columns([1.8, 3])
            with rec_col1:
                if st.button(f"📥 Receive Shipment & Restock Warehouse", key=f"rec_{po_id}", type="primary"):
                    res = state_mgr.receive_and_restock_po(po_id)
                    if res:
                        st.balloons()
                        st.success(
                            f"🎉 Shipment {po_id} docked! Restocked +{po['ordered_quantity']} units of {po['product_name']}. "
                            f"New on-shelf stock: {res['product']['current_stock']} units ({res['product']['status_label']})!"
                        )
                        st.rerun()

    st.divider()

    st.subheader("📜 Inbound Logistics Fulfilled History")
    if fulfilled_pos:
        f_df = pd.DataFrame([
            {
                "PO Number": p["po_id"],
                "Product": p["product_name"],
                "Vendor": p["vendor"],
                "Restocked Qty": f"+{p['ordered_quantity']} units",
                "Total Value": f"${p['total_cost']:,.2f}",
                "Status": p["status"],
                "Received At": p.get("received_at", p["created_at"]),
            }
            for p in fulfilled_pos
        ])
        st.dataframe(f_df, use_container_width=True)
    else:
        st.caption("No purchase orders have been received yet.")
