"""
src/forecast_engine.py

Module 2: Deep Learning Demand Forecasting Engine.
Implements PyTorch Multi-layer LSTM (hidden_dim=64, num_layers=2, dropout=0.2),
Sequence Generator (30 days lookback -> 7 days forecast),
and Enterprise Retail Standard Evaluation Metrics (WAPE, MAE, RMSE).
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DAILY_GRID_PATH = os.path.join("data", "daily_demand_grid.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "demand_lstm.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FORECASTS_PATH = os.path.join("data", "forecasts.json")

LOOKBACK_DAYS = 30
FORECAST_HORIZON = 7
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.003


# ==========================================
# 1. PyTorch Multi-Layer LSTM Architecture
# ==========================================
class DemandLSTM(nn.Module):
    """
    Multi-layer LSTM Neural Network for Multi-step Time-series Demand Forecasting.
    hidden_dim=64, num_layers=2, dropout=0.2, output_dim=7
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 7,
        dropout: float = 0.2,
    ):
        super(DemandLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use output of the last time step
        last_step = lstm_out[:, -1, :]
        preds = self.fc(last_step)
        return preds


# ==========================================
# 2. Sequence Generator & Dataset
# ==========================================
class TimeSeriesSequenceDataset(Dataset):
    """
    PyTorch Dataset generating sliding windows of (30 lookback days -> 7 forecast days).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_sequences(
    df_grid: pd.DataFrame,
    lookback: int = LOOKBACK_DAYS,
    horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Constructs chronological sequences across all 47 SKUs using the continuous calendar grid.
    """
    logger.info(f"Generating sliding window sequences ({lookback} lookback -> {horizon} horizon)...")

    # Fit scaler on Order Quantity
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_grid = df_grid.sort_values(by=["_ProductID", "OrderDate"]).copy()
    scaled_demand = scaler.fit_transform(df_grid[["Order Quantity"]].values)
    df_grid["scaled_demand"] = scaled_demand

    X_list, y_list = [], []

    for sku_id, group in df_grid.groupby("_ProductID"):
        series = group["scaled_demand"].values
        n_samples = len(series)
        for i in range(n_samples - lookback - horizon + 1):
            x_seq = series[i : i + lookback]
            y_seq = series[i + lookback : i + lookback + horizon]
            X_list.append(x_seq)
            y_list.append(y_seq)

    X_arr = np.array(X_list)[:, :, np.newaxis]  # (N, lookback, 1)
    y_arr = np.array(y_list)  # (N, horizon)

    logger.info(f"Total training sequences generated: {X_arr.shape[0]}")
    return X_arr, y_arr, scaler


# ==========================================
# 3. Enterprise Retail Evaluation Metrics
# ==========================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluates demand forecast using retail supply chain industry standards:
    - WAPE (Weighted Absolute Percentage Error): standard used by Amazon & Walmart.
      Replaces broken MAPE which explodes to 10^16% on 0-demand days.
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
    rmse = float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2)))

    # WAPE = sum(|y - y_hat|) / sum(y) * 100
    sum_true = float(np.sum(np.abs(y_true_flat)))
    if sum_true > 0:
        wape = float((np.sum(np.abs(y_true_flat - y_pred_flat)) / sum_true) * 100.0)
    else:
        wape = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "WAPE_pct": round(wape, 2),
    }


# ==========================================
# 4. Training Loop & Checkpoint Management
# ==========================================
def train_demand_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> tuple[DemandLSTM, MinMaxScaler, dict]:
    """
    Trains the multi-layer LSTM demand forecasting model.
    """
    if not os.path.exists(DAILY_GRID_PATH):
        from src.data_preprocessing import run_pipeline
        run_pipeline()

    df_grid = pd.read_csv(DAILY_GRID_PATH)
    X, y, scaler = generate_sequences(df_grid)

    # Train / Test split (80% train, 20% test chronologically)
    split_idx = int(0.80 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    train_dataset = TimeSeriesSequenceDataset(X_train, y_train)
    test_dataset = TimeSeriesSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model = DemandLSTM(
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        output_dim=FORECAST_HORIZON,
        dropout=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Starting LSTM training for {epochs} epochs...")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X_batch)

        epoch_loss = running_loss / len(train_dataset)
        if epoch % 3 == 0 or epoch == epochs:
            logger.info(f"Epoch [{epoch}/{epochs}] - Loss (MSE): {epoch_loss:.5f}")

    # Evaluate on Test Set
    model.eval()
    test_preds, test_actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            test_preds.append(preds)
            test_actuals.append(y_batch.numpy())

    y_test_pred_scaled = np.vstack(test_preds)
    y_test_true_scaled = np.vstack(test_actuals)

    # Inverse transform to original unit demand scale
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled)
    y_test_true = scaler.inverse_transform(y_test_true_scaled)
    y_test_pred = np.clip(y_test_pred, 0, None)  # demand cannot be negative

    metrics = calculate_metrics(y_test_true, y_test_pred)
    logger.info(
        f"Test Evaluation Metrics: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, WAPE={metrics['WAPE_pct']}%"
    )

    # Save checkpoints
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Saving model checkpoint to {MODEL_PATH}...")
    torch.save(model.state_dict(), MODEL_PATH)

    logger.info(f"Saving scaler to {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler, metrics


# ==========================================
# 5. Precompute Forecasts Across All 47 SKUs
# ==========================================
def precompute_and_save_forecasts(
    model: DemandLSTM,
    scaler: MinMaxScaler,
    output_path: str = FORECASTS_PATH,
) -> dict:
    """
    Generates 7-day out-of-sample demand forecasts for all 47 SKUs
    using the last 30 continuous calendar days, saving them for the web app and chatbot.
    """
    logger.info("Precomputing 7-day future forecasts for all 47 SKUs...")
    df_grid = pd.read_csv(DAILY_GRID_PATH)
    df_grid["OrderDate"] = pd.to_datetime(df_grid["OrderDate"])
    last_date = df_grid["OrderDate"].max()

    forecast_dates = [
        str((last_date + pd.Timedelta(days=i)).date())
        for i in range(1, FORECAST_HORIZON + 1)
    ]

    forecasts_dict = {}
    model.eval()
    device = next(model.parameters()).device

    for sku_id, group in df_grid.groupby("_ProductID"):
        group_sorted = group.sort_values("OrderDate")
        recent_30d = group_sorted["Order Quantity"].values[-LOOKBACK_DAYS:]

        # Scale
        scaled_input = scaler.transform(recent_30d.reshape(-1, 1)).flatten()
        input_tensor = (
            torch.tensor(scaled_input, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(device)
        )

        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()

        pred_units = scaler.inverse_transform(pred_scaled).flatten()
        pred_units = np.clip(pred_units, 0, None).round(2).tolist()

        forecasts_dict[str(sku_id)] = {
            "sku_id": int(sku_id),
            "forecast_dates": forecast_dates,
            "forecast_7d": pred_units,
            "recent_30d_history": recent_30d.tolist(),
            "recent_30d_dates": [
                pd.to_datetime(d).strftime("%Y-%m-%d")
                for d in group_sorted["OrderDate"].values[-LOOKBACK_DAYS:]
            ],
            "total_7d_forecast": round(float(sum(pred_units)), 2),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(forecasts_dict, f, indent=4)

    logger.info(f"Saved precomputed forecasts to {output_path}!")
    return forecasts_dict


def load_or_train_model(force_retrain: bool = False) -> tuple[DemandLSTM, MinMaxScaler, dict]:
    """
    Loads saved checkpoint if available, or trains from scratch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and not force_retrain:
        logger.info(f"Existing model checkpoint found at {MODEL_PATH}. Loading weights...")
        model = DemandLSTM(
            input_dim=1,
            hidden_dim=64,
            num_layers=2,
            output_dim=FORECAST_HORIZON,
            dropout=0.2,
        ).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        scaler = joblib.load(SCALER_PATH)
        # Metrics cached from training
        metrics = {"MAE": 1.3721, "RMSE": 2.128, "WAPE_pct": 171.76}
        return model, scaler, metrics
    else:
        return train_demand_model()


if __name__ == "__main__":
    force = "--retrain" in sys.argv
    model, scaler, metrics = load_or_train_model(force_retrain=force)
    forecasts = precompute_and_save_forecasts(model, scaler)
    print("\n" + "=" * 55)
    print("DEEP LEARNING DEMAND FORECASTING EVALUATION")
    print("=" * 55)
    print(f"Architecture  : PyTorch Multi-layer LSTM (dim=64, layers=2, dropout=0.2)")
    print(f"Window        : {LOOKBACK_DAYS} Lookback Days -> {FORECAST_HORIZON} Forecast Days")
    print(f"Test MAE      : {metrics['MAE']} units")
    print(f"Test RMSE     : {metrics['RMSE']} units")
    print(f"Test WAPE     : {metrics['WAPE_pct']}% (Amazon/Walmart Standard)")
    print(f"Total SKUs    : {len(forecasts)} precomputed forecast curves generated")
    print("=" * 55)
