"""
src/data_preprocessing.py

Module 1: Data Cleaning, Sanitization, Feature Engineering & Continuous Calendar Grid.
Calculates core retail supply chain KPIs and generates continuous zero-filled demand sequences.
"""

import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "US_Regional_Sales_Data.csv"
CLEANED_DATA_PATH = os.path.join("data", "cleaned_sales_data.csv")
DAILY_GRID_PATH = os.path.join("data", "daily_demand_grid.csv")
KPI_SUMMARY_PATH = os.path.join("data", "kpi_summary.json")


def clean_currency_series(series: pd.Series) -> pd.Series:
    """
    Sanitizes currency strings by stripping '$' and ',' symbols and converting to float.
    """
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """
    Robustly parses mixed date formats (e.g. DD/MM/YY vs DD/MM/YYYY) with dayfirst=True.
    """
    return pd.to_datetime(series, format="mixed", dayfirst=True)


def load_and_sanitize_data(csv_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw sales data, cleans currency fields, and parses dates.
    """
    logger.info(f"Loading raw sales data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Sanitize Currency
    logger.info("Sanitizing Unit Cost and Unit Price fields...")
    df["Unit Cost"] = clean_currency_series(df["Unit Cost"])
    df["Unit Price"] = clean_currency_series(df["Unit Price"])

    # 2. Robust Date Parsing
    date_cols = ["ProcuredDate", "OrderDate", "ShipDate", "DeliveryDate"]
    logger.info(f"Parsing mixed date formats across: {date_cols}...")
    for col in date_cols:
        df[col] = parse_mixed_dates(df[col])

    # 3. Supply Chain KPI Feature Engineering
    logger.info("Computing retail supply chain KPIs...")
    df["total_fulfillment_days"] = (df["DeliveryDate"] - df["OrderDate"]).dt.days
    df["procurement_lead_days"] = (df["ShipDate"] - df["ProcuredDate"]).dt.days
    df["shipping_days"] = (df["DeliveryDate"] - df["ShipDate"]).dt.days

    df["gross_revenue"] = df["Order Quantity"] * df["Unit Price"]
    df["discount_amount"] = df["gross_revenue"] * df["Discount Applied"]
    df["net_revenue"] = df["gross_revenue"] - df["discount_amount"]
    df["total_cost"] = df["Order Quantity"] * df["Unit Cost"]
    df["gross_profit"] = df["net_revenue"] - df["total_cost"]
    df["profit_margin_pct"] = (df["gross_profit"] / df["net_revenue"]) * 100

    return df


def calculate_network_kpis(df: pd.DataFrame) -> dict:
    """
    Calculates executive-level network KPIs across revenue, profit, and fulfillment.
    """
    tot_gross_rev = float(df["gross_revenue"].sum())
    tot_discount = float(df["discount_amount"].sum())
    tot_net_rev = float(df["net_revenue"].sum())
    tot_cost = float(df["total_cost"].sum())
    tot_profit = float(df["gross_profit"].sum())
    operating_margin = float((tot_profit / tot_net_rev) * 100)
    avg_fulfillment = float(df["total_fulfillment_days"].mean())

    warehouse_fulfillment = (
        df.groupby("WarehouseCode")["total_fulfillment_days"]
        .agg(["mean", "count"])
        .reset_index()
    )
    warehouse_kpis = {
        row["WarehouseCode"]: {
            "avg_fulfillment_days": round(float(row["mean"]), 2),
            "order_count": int(row["count"]),
        }
        for _, row in warehouse_fulfillment.iterrows()
    }

    slowest_warehouse = max(
        warehouse_kpis.items(), key=lambda x: x[1]["avg_fulfillment_days"]
    )

    kpis = {
        "total_orders": len(df),
        "total_units_sold": int(df["Order Quantity"].sum()),
        "gross_revenue": round(tot_gross_rev, 2),
        "discount_amount": round(tot_discount, 2),
        "net_revenue": round(tot_net_rev, 2),
        "total_cost": round(tot_cost, 2),
        "gross_profit": round(tot_profit, 2),
        "operating_margin_pct": round(operating_margin, 2),
        "avg_fulfillment_days": round(avg_fulfillment, 2),
        "slowest_warehouse": {
            "code": slowest_warehouse[0],
            "avg_fulfillment_days": slowest_warehouse[1]["avg_fulfillment_days"],
        },
        "warehouse_metrics": warehouse_kpis,
        "date_range": {
            "min_date": str(df["OrderDate"].min().date()),
            "max_date": str(df["OrderDate"].max().date()),
            "total_calendar_days": int((df["OrderDate"].max() - df["OrderDate"].min()).days + 1),
        },
        "unique_skus_count": int(df["_ProductID"].nunique()),
    }
    return kpis


def create_continuous_calendar_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Solves the non-consecutive date issue by reindexing daily SKU sales onto
    a complete continuous calendar date range, filling days without orders with 0.0.
    Ensures true chronological sequence continuity for neural network modeling.
    """
    logger.info("Generating continuous calendar zero-filled demand grid...")

    min_date = df["OrderDate"].min()
    max_date = df["OrderDate"].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    skus = sorted(df["_ProductID"].unique())

    logger.info(
        f"Continuous calendar spans {len(full_date_range)} days ({min_date.date()} to {max_date.date()}) "
        f"across {len(skus)} SKUs -> {len(full_date_range) * len(skus)} total sequence points."
    )

    # Aggregate daily demand per SKU
    daily_sales = (
        df.groupby(["_ProductID", "OrderDate"])["Order Quantity"]
        .sum()
        .reset_index()
    )

    # Create full Cartesian grid (all SKUs x all Calendar Dates)
    grid_index = pd.MultiIndex.from_product(
        [skus, full_date_range], names=["_ProductID", "OrderDate"]
    )
    grid_df = pd.DataFrame(index=grid_index).reset_index()

    # Merge actual daily sales onto the continuous grid and zero-fill
    merged = pd.merge(grid_df, daily_sales, on=["_ProductID", "OrderDate"], how="left")
    merged["Order Quantity"] = merged["Order Quantity"].fillna(0.0)

    # Compute SKU average price and unit cost for downstream inventory modeling
    sku_price_map = (
        df.groupby("_ProductID")
        .agg(
            avg_unit_price=("Unit Price", "mean"),
            avg_unit_cost=("Unit Cost", "mean"),
            total_historical_demand=("Order Quantity", "sum"),
        )
        .reset_index()
    )

    merged = pd.merge(merged, sku_price_map, on="_ProductID", how="left")

    # Add rolling lag features useful for both DL and inventory statistics
    merged["rolling_7d_mean"] = (
        merged.groupby("_ProductID")["Order Quantity"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    merged["rolling_30d_mean"] = (
        merged.groupby("_ProductID")["Order Quantity"]
        .transform(lambda x: x.rolling(30, min_periods=1).mean())
    )
    merged["rolling_30d_std"] = (
        merged.groupby("_ProductID")["Order Quantity"]
        .transform(lambda x: x.rolling(30, min_periods=1).std().fillna(0.0))
    )

    return merged


def run_pipeline(csv_path: str = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Runs the complete preprocessing pipeline, computes KPIs, and saves files.
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df_cleaned = load_and_sanitize_data(csv_path)
    kpis = calculate_network_kpis(df_cleaned)
    df_grid = create_continuous_calendar_grid(df_cleaned)

    logger.info(f"Saving cleaned transaction data to {CLEANED_DATA_PATH}...")
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)

    logger.info(f"Saving continuous daily demand grid to {DAILY_GRID_PATH}...")
    df_grid.to_csv(DAILY_GRID_PATH, index=False)

    logger.info(f"Saving KPI summary to {KPI_SUMMARY_PATH}...")
    with open(KPI_SUMMARY_PATH, "w") as f:
        json.dump(kpis, f, indent=4)

    logger.info("Data preprocessing and KPI computation completed successfully!")
    return df_cleaned, df_grid, kpis


if __name__ == "__main__":
    df_cleaned, df_grid, kpis = run_pipeline()
    print("\n" + "=" * 50)
    print("RETAIL SUPPLY CHAIN KPI VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total Transactions : {kpis['total_orders']}")
    print(f"Net Revenue        : ${kpis['net_revenue']/1e6:.2f} Million")
    print(f"Gross Profit       : ${kpis['gross_profit']/1e6:.2f} Million")
    print(f"Operating Margin   : {kpis['operating_margin_pct']:.2f}%")
    print(f"Avg Fulfillment    : {kpis['avg_fulfillment_days']} days")
    print(f"Bottleneck Hub     : {kpis['slowest_warehouse']['code']} ({kpis['slowest_warehouse']['avg_fulfillment_days']} days)")
    print(f"Unique SKUs        : {kpis['unique_skus_count']}")
    print(f"Daily Grid Points  : {len(df_grid)} records across {kpis['date_range']['total_calendar_days']} days")
    print("=" * 50)
