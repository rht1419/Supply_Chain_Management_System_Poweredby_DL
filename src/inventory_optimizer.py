"""
src/inventory_optimizer.py

Module 3: Inventory Optimization & Dynamic Stock Classification Engine.
Calculates Safety Stock (SS), Reorder Point (ROP), and Economic Order Quantity (EOQ)
across all 47 SKUs, classifying inventory health into 4 operational tiers.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DAILY_GRID_PATH = os.path.join("data", "daily_demand_grid.csv")
CLEANED_DATA_PATH = os.path.join("data", "cleaned_sales_data.csv")
INVENTORY_STATUS_PATH = os.path.join("data", "inventory_status.json")

# Industry standard parameters
SERVICE_LEVEL_Z = 1.65  # 95% cycle service level
DEFAULT_ORDERING_COST_S = 50.0  # Fixed procurement cost per PO ($50)
HOLDING_COST_RATE_H = 0.20  # Annual holding cost rate (20% of unit cost)
DEFAULT_LEAD_TIME_DAYS = 20.67  # Network average fulfillment lead time


def compute_sku_inventory_parameters(
    grid_df: pd.DataFrame,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
    z_score: float = SERVICE_LEVEL_Z,
    ordering_cost_s: float = DEFAULT_ORDERING_COST_S,
    holding_cost_rate: float = HOLDING_COST_RATE_H,
) -> pd.DataFrame:
    """
    Computes mathematical inventory policies for all 47 SKUs:
    - Safety Stock (SS) = Z * sigma_D * sqrt(L)
    - Reorder Point (ROP) = (mu_D * L) + SS
    - Economic Order Quantity (EOQ) = sqrt( (2 * D_annual * S) / H )
    """
    logger.info("Computing SKU demand statistics from continuous calendar grid...")

    sku_stats = (
        grid_df.groupby("_ProductID")
        .agg(
            mean_daily_demand=("Order Quantity", "mean"),
            std_daily_demand=("Order Quantity", "std"),
            unit_cost=("avg_unit_cost", "first"),
            unit_price=("avg_unit_price", "first"),
            total_historical_demand=("Order Quantity", "sum"),
        )
        .reset_index()
    )

    # Fill potential NaNs in std
    sku_stats["std_daily_demand"] = sku_stats["std_daily_demand"].fillna(0.0)

    # 1. Safety Stock (SS) = Z * sigma_D * sqrt(L)
    sku_stats["safety_stock"] = (
        z_score * sku_stats["std_daily_demand"] * np.sqrt(lead_time_days)
    ).round().astype(int)

    # Ensure at least 1 unit safety stock for active SKUs
    sku_stats["safety_stock"] = sku_stats["safety_stock"].clip(lower=1)

    # 2. Reorder Point (ROP) = (mu_D * L) + SS
    lead_time_demand = sku_stats["mean_daily_demand"] * lead_time_days
    sku_stats["lead_time_demand"] = lead_time_demand.round(2)
    sku_stats["reorder_point"] = (lead_time_demand + sku_stats["safety_stock"]).round().astype(int)

    # 3. Economic Order Quantity (EOQ) = sqrt( (2 * D_annual * S) / H )
    annual_demand = sku_stats["mean_daily_demand"] * 365.0
    sku_stats["annual_demand"] = annual_demand.round(1)

    # Holding cost per unit per year = holding_cost_rate * unit_cost
    unit_holding_cost = (sku_stats["unit_cost"] * holding_cost_rate).clip(lower=1.0)
    sku_stats["annual_unit_holding_cost"] = unit_holding_cost.round(2)

    eoq_calc = np.sqrt(
        (2.0 * annual_demand * ordering_cost_s) / unit_holding_cost
    )
    # Ensure minimum EOQ of 5 units
    sku_stats["eoq"] = eoq_calc.round().clip(lower=5).astype(int)

    # Max optimal inventory ceiling = ROP + EOQ
    sku_stats["max_optimal_stock"] = sku_stats["reorder_point"] + sku_stats["eoq"]

    return sku_stats


def simulate_and_classify_inventory(
    sku_policies: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulates operational current warehouse on-hand stock and classifies each SKU:
    - 🚨 Critical Understocking / Stockout Risk: Current Stock < SS
    - ⚠️ Reorder Needed: Current Stock < ROP (and >= SS)
    - ✅ Balanced / Optimal Stock: ROP <= Current Stock <= ROP + EOQ
    - 📦 Overstocking Risk: Current Stock > ROP + EOQ
    """
    logger.info("Classifying inventory health across SKUs...")
    rng = np.random.default_rng(seed)

    current_stocks = []
    statuses = []
    status_codes = []
    recommended_pos = []
    po_costs = []

    for _, row in sku_policies.iterrows():
        ss = row["safety_stock"]
        rop = row["reorder_point"]
        max_opt = row["max_optimal_stock"]
        eoq = row["eoq"]
        cost = row["unit_cost"]

        # To create a realistic enterprise distribution across all 4 categories:
        # 15% critical understock, 25% reorder needed, 45% balanced, 15% overstock
        sku_id = int(row["_ProductID"])
        category_slot = sku_id % 10

        if category_slot in (1, 6):
            # Critical understock (< SS)
            stock = int(rng.integers(max(0, ss - int(0.5 * ss) - 2), max(1, ss)))
        elif category_slot in (2, 7):
            # Reorder needed (SS <= stock < ROP)
            stock = int(rng.integers(ss, max(ss + 1, rop)))
        elif category_slot in (0, 9):
            # Overstocked (> max_opt)
            stock = int(rng.integers(max_opt + 5, int(max_opt * 1.5) + 15))
        else:
            # Balanced / Optimal (ROP <= stock <= max_opt)
            stock = int(rng.integers(rop, max_opt + 1))

        current_stocks.append(stock)

        # Classification logic
        if stock < ss:
            status = "🚨 Critical Understock / Stockout Risk"
            code = "CRITICAL_UNDERSTOCK"
            # Recommended PO brings stock back up to ROP + EOQ
            reorder_qty = int(max(eoq, (rop + eoq) - stock))
        elif stock < rop:
            status = "⚠️ Reorder Needed"
            code = "REORDER_NEEDED"
            reorder_qty = int(max(eoq, (rop + eoq) - stock))
        elif stock <= max_opt:
            status = "✅ Balanced / Optimal Stock"
            code = "OPTIMAL"
            reorder_qty = 0
        else:
            status = "📦 Overstocking Risk (Excess Holding Cost)"
            code = "OVERSTOCK"
            reorder_qty = 0

        po_cost = round(reorder_qty * cost, 2)
        statuses.append(status)
        status_codes.append(code)
        recommended_pos.append(reorder_qty)
        po_costs.append(po_cost)

    df_out = sku_policies.copy()
    df_out["current_stock"] = current_stocks
    df_out["status"] = statuses
    df_out["status_code"] = status_codes
    df_out["recommended_reorder_qty"] = recommended_pos
    df_out["estimated_po_cost"] = po_costs
    df_out["excess_holding_units"] = np.maximum(
        0, df_out["current_stock"] - df_out["max_optimal_stock"]
    )
    df_out["excess_holding_cost_annual"] = (
        df_out["excess_holding_units"] * df_out["annual_unit_holding_cost"]
    ).round(2)

    return df_out


def run_inventory_engine(
    grid_path: str = DAILY_GRID_PATH,
    output_path: str = INVENTORY_STATUS_PATH,
) -> pd.DataFrame:
    """
    Executes the inventory optimization engine and persists results.
    """
    if not os.path.exists(grid_path):
        from src.data_preprocessing import run_pipeline
        run_pipeline()

    grid_df = pd.read_csv(grid_path)
    sku_policies = compute_sku_inventory_parameters(grid_df)
    classified_df = simulate_and_classify_inventory(sku_policies)

    # Save to JSON for web app and chatbot
    classified_records = classified_df.to_dict(orient="records")

    summary_stats = {
        "total_skus": len(classified_df),
        "status_distribution": classified_df["status_code"].value_counts().to_dict(),
        "total_recommended_po_value": round(float(classified_df["estimated_po_cost"].sum()), 2),
        "total_excess_holding_cost_annual": round(
            float(classified_df["excess_holding_cost_annual"].sum()), 2
        ),
        "skus": classified_records,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary_stats, f, indent=4)

    logger.info(f"Inventory status saved successfully to: {output_path}")
    return classified_df


if __name__ == "__main__":
    df_inv = run_inventory_engine()
    print("\n" + "=" * 55)
    print("INVENTORY OPTIMIZATION HEALTH MATRIX")
    print("=" * 55)
    print("Status Breakdown:")
    print(df_inv["status"].value_counts().to_string())
    print("\nSample SKU Policies (First 5):")
    sample_cols = [
        "_ProductID",
        "current_stock",
        "safety_stock",
        "reorder_point",
        "eoq",
        "status",
        "recommended_reorder_qty",
    ]
    print(df_inv[sample_cols].head(5).to_string(index=False))
    print("=" * 55)
