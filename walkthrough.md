# 🏆 Capstone Project Walkthrough & System Verification Report

## Project Overview
This project delivers a production-grade, end-to-end retail supply chain analytics platform, deep learning demand forecasting engine, and an autonomous Agentic AI operations copilot.

---

## 1. Verified Core Supply Chain KPIs
All key business metrics derived from `US_Regional_Sales_Data.csv` match enterprise benchmarks:

| KPI Dimension | Calculated Result | Business Significance |
| :--- | :--- | :--- |
| **Gross Merchandise Volume (GMV)** | **$82.69 Million** | Total order volume across 7,991 customer orders |
| **Promotional Discounts** | **$9.66 Million** | Channel discount subsidies |
| **Net Recognized Revenue** | **$73.04 Million** | Top-line retail revenue |
| **Cost of Goods Sold (COGS)** | **$51.82 Million** | Landed product costs |
| **Gross Operating Profit** | **$21.22 Million** | Bottom-line inventory contribution |
| **Operating Margin** | **29.05%** | Strong margin performance |
| **Network Fulfillment Lead Time** | **20.67 Days** | Benchmark order-to-delivery SLA |
| **Bottleneck Warehouse** | **WARE-NMK1003 (20.96 Days)** | Identified delay bottleneck for regional re-routing |

---

## 2. Technical Component Verification

### ✅ Module 1: `src/data_preprocessing.py`
* **Currency Sanitization**: Stripped dirty currency characters (`$`, `,`) and cast to `float64`.
* **Date Normalization**: Parsed mixed `DD/MM/YY` and `DD/MM/YYYY` formats with `dayfirst=True` across `ProcuredDate`, `OrderDate`, `ShipDate`, and `DeliveryDate`.
* **Continuous Calendar Zero-Filling Grid**:
  * Solved non-consecutive date gaps in retail sales.
  * Reindexed daily order demand across all 47 SKUs and 945 calendar days ($47 \times 945 = 44,415$ sequence points).
  * Inactive order days zero-filled with `0.0`.
  * Generated `data/cleaned_sales_data.csv`, `data/daily_demand_grid.csv`, and `data/kpi_summary.json`.

### ✅ Module 2: `src/forecast_engine.py`
* **Architecture**: PyTorch Multi-layer LSTM (`input_dim=1`, `hidden_dim=64`, `num_layers=2`, `dropout=0.2`, `output_dim=7`).
* **Sliding Window Sequence Generator**: Lookback of 30 continuous calendar days to predict 7 days ahead.
* **Metric Bug Fix**: Replaced unstable MAPE (which diverges to $10^{16}\%$ on zero-demand days) with retail industry standards:
  * **WAPE (Weighted Absolute Percentage Error)**: Walmart/Amazon standard.
  * **MAE**: Mean Absolute Error in unit demand.
  * **RMSE**: Root Mean Squared Error.
* **Model Checkpoints**: Checkpoints saved to `models/demand_lstm.pt` and `models/scaler.pkl`. Precomputed 7-day future projections saved to `data/forecasts.json`.

### ✅ Module 3: `src/inventory_optimizer.py`
* **Formulated Mathematical Policies**:
  * Safety Stock ($SS$): $SS = Z \times \sigma_D \times \sqrt{L}$ ($Z=1.65$ for 95% service level).
  * Reorder Point ($ROP$): $ROP = (\mu_D \times L) + SS$.
  * Economic Order Quantity ($EOQ$): $EOQ = \sqrt{\frac{2 \times D_{\text{annual}} \times S}{H}}$.
* **Dynamic 4-Tier Health Matrix**:
  * 🚨 **Critical Understock / Stockout Risk**: 10 SKUs ($Current < SS$)
  * ⚠️ **Reorder Needed**: 10 SKUs ($Current < ROP$)
  * ✅ **Balanced / Optimal Stock**: 19 SKUs ($ROP \le Current \le ROP + EOQ$)
  * 📦 **Overstocking Risk**: 8 SKUs ($Current > ROP + EOQ$, identifying excess holding costs)
* Generated `data/inventory_status.json`.

### ✅ Module 4: `src/agentic_chatbot.py`
* Autonomous operations copilot with deterministic tool calling:
  * `get_stock_health()`: Responds to understocking and overstocking queries.
  * `get_sku_forecast(sku_id)`: Fetches 7-day deep learning demand curves for any SKU.
  * `get_warehouse_delays()`: Analyzes hub performance and identifies `WARE-NMK1003` as the slowest hub.
  * `generate_purchase_order(sku_id, quantity)`: Generates approved restocking purchase orders with estimated delivery dates.

### ✅ Module 5: `Supply_Chain_DL_Colab_Training.ipynb`
* Self-contained Google Colab notebook pre-configured for free T4 GPU.
* Side-by-side benchmarking of **LSTM vs. BiLSTM vs. GRU**.
* Loss convergence curve visualizations and WAPE/MAE/RMSE comparison table.
* Automatic weight download triggers for `demand_lstm.pt` and `scaler.pkl`.

### ✅ Module 6: `app.py` (Streamlit Dashboard)
* Runs live locally on `http://localhost:8501`.
* **Tab 1: Executive Command Center**: Metrics, warehouse lead times, channel distribution, revenue trends.
* **Tab 2: DL Demand Forecasting**: Interactive SKU selector, 30-day historical continuity + 7-day LSTM predicted curve.
* **Tab 3: Inventory Health & Risk Matrix**: Filterable table with color badges, SS, ROP, EOQ, and 1-click PO generator.
* **Tab 4: Agentic AI Copilot**: Interactive chat window with quick-action prompts.

### ✅ Module 7: Local Notebook & Documentation
* `capstone.ipynb`: Clean local notebook covering data preprocessing, EDA, LSTM modeling, inventory optimization, and copilot tests.
* `README.md`: Architecture diagrams, formulas, business KPIs, and execution guide.
