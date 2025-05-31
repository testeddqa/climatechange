import os
import joblib
import torch
import torch.nn as nn
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from typing import Tuple

# =================== CONFIG ===================
FILENAME = "cams_egg4.grib"
ENGINE = "cfgrib"

TEMPERATURE_FIELD_KELVIN = "2t"
TEMPERATURE_FIELD_CELSIUS = "temp"
CO2_FIELD = "co2"
CO2_FIELD_PPM = "co2_ppm"
TIME_FIELD = "time"

MODEL_PATH = "lstm_model.pt"
SCALER_PATH = "scaler.gz"

SEQ_LENGTH = 120
N_PREDICT = 12
EPOCHS = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


# ============ DATA LOADING & PROCESSING ============
def load_variable(short_name: str) -> pd.DataFrame:
    """Load and convert a single variable from GRIB to DataFrame."""
    ds = xr.open_dataset(FILENAME, engine=ENGINE, backend_kwargs={"filter_by_keys": {"shortName": short_name}})
    var_name = list(ds.data_vars)[0]
    df = ds.to_dataframe().reset_index()[[TIME_FIELD, var_name]]
    return df.rename(columns={var_name: short_name})


def process_data() -> pd.DataFrame:
    """Preprocess and merge temperature and CO2 data into a monthly time series."""
    df_temp = load_variable(TEMPERATURE_FIELD_KELVIN)
    df_co2 = load_variable(CO2_FIELD)

    df_temp[TIME_FIELD] = pd.to_datetime(df_temp[TIME_FIELD])
    df_co2[TIME_FIELD] = pd.to_datetime(df_co2[TIME_FIELD])

    df_temp = df_temp.set_index(TIME_FIELD).resample("MS").mean().reset_index()
    df_co2 = df_co2.set_index(TIME_FIELD).resample("MS").mean().reset_index()

    df_merged = pd.merge(df_temp, df_co2, on=TIME_FIELD).dropna()

    df_merged[TEMPERATURE_FIELD_CELSIUS] = df_merged[TEMPERATURE_FIELD_KELVIN] - 273.15
    df_merged[CO2_FIELD_PPM] = df_merged[CO2_FIELD] * 1_000_000

    df_merged.set_index(TIME_FIELD, inplace=True)
    return df_merged[[TEMPERATURE_FIELD_CELSIUS, CO2_FIELD_PPM]]


# ============== SEQUENCE CREATION ==============
def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform data into input-output sequences for LSTM."""
    x, y = [], []
    for i in range(len(data) - seq_length - N_PREDICT):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + N_PREDICT])
    return (
        torch.tensor(np.array(x), dtype=torch.float32).to(device),
        torch.tensor(np.array(y), dtype=torch.float32).to(device)
    )


# ============== LSTM MODEL DEFINITION ==============
class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 128,
                 num_layers: int = 1, output_size: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ============== TRAINING LOOP ==============
def train_model(model: nn.Module, x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> None:
    """Train the LSTM model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.009)

    for epoch in range(EPOCHS):
        model.train()
        output = model(x_tensor)
        target = y_tensor[:, 0, :]  # predict only first step of future
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{EPOCHS} - Loss: {loss.item():.6f}")


def save_model_and_scaler(model: nn.Module, scaler: BaseEstimator) -> None:
    """Save the LSTM model and the scaler."""
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to '{MODEL_PATH}' and scaler to '{SCALER_PATH}'.")


def load_model_and_scaler() -> Tuple[nn.Module, BaseEstimator]:
    """Load the LSTM model and the scaler."""
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded.")
    return model, scaler


# ============== PREDICTION ==============
def predict(model: nn.Module, scaler: BaseEstimator, data_scaled: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Predict future values for N_PREDICT steps ahead."""
    model.eval()
    start_idx = -(SEQ_LENGTH + N_PREDICT)
    input_seq = data_scaled[start_idx:start_idx + SEQ_LENGTH]
    predictions = []

    for _ in range(N_PREDICT):
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).cpu().numpy()
        predictions.append(pred[0])
        input_seq = np.vstack((input_seq[1:], pred))

    predictions = scaler.inverse_transform(np.array(predictions))
    return pd.DataFrame(predictions, columns=[TEMPERATURE_FIELD_CELSIUS, CO2_FIELD_PPM], index=dates)


# ============== VISUALIZATION ==============
def plot_predictions(dates: pd.DatetimeIndex, real: pd.DataFrame, predicted: pd.DataFrame) -> None:
    """Display temperature and CO2 real vs predicted."""
    plt.figure(figsize=(12, 6))

    # Temperature
    plt.plot(dates, real[TEMPERATURE_FIELD_CELSIUS], label="Temperatura reala", color="tab:blue", marker='o')
    plt.plot(dates, predicted[TEMPERATURE_FIELD_CELSIUS], label="Temperatura prezisa", color="tab:blue",
             linestyle="--", marker='x')

    # CO2 (offset for scale visibility)
    plt.plot(dates, real[CO2_FIELD_PPM] - 600, label="CO2 real - 600", color="tab:green", marker='o')
    plt.plot(dates, predicted[CO2_FIELD_PPM] - 600, label="CO2 prezis - 600", color="tab:green",
             linestyle="--", marker='x')

    plt.title("Temperatura si CO2 - Real vs Prezis")
    plt.xlabel("Timp")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ======= Main =======
def main(retrain=False):
    df = process_data()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)
    x_tensor, y_tensor = create_sequences(data_scaled, SEQ_LENGTH)

    if retrain or not os.path.exists(MODEL_PATH):
        model = LSTMModel().to(device)
        train_model(model, x_tensor, y_tensor)
        save_model_and_scaler(model, scaler)
    else:
        model, scaler = load_model_and_scaler()

    comparison_dates = df.index[-N_PREDICT:]
    predicted = predict(model, scaler, data_scaled, comparison_dates)
    real = df.loc[comparison_dates]

    plot_predictions(comparison_dates, real, predicted)


if __name__ == "__main__":
    main(retrain=True)  # False to load existing model and scaler
