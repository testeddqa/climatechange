import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
from itertools import product

# Config
FILENAME = "cams_egg4.grib"
ENGINE = "cfgrib"
TEMPERATURE_FIELD_KELVIN = '2t'
TEMPERATURE_FIELD_CELSIUS = 'temp'
CO2_FIELD = 'co2'
CO2_FIELD_PPM = 'co2_ppm'
TIME_FIELD = 'time'
SEQ_LENGTH = 120
N_PREDICT = 12

# Grid Search params
GRID_PARAMS = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2],
    'learning_rate': [0.001, 0.005, 0.009, 0.009],
    'epochs': [100, 200, 300, 350, 400],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


# Load data
def load_variable(short_name):
    dataset = xr.open_dataset(FILENAME, engine=ENGINE, backend_kwargs={"filter_by_keys": {"shortName": short_name}})
    var_name = list(dataset.data_vars)[0]
    data_frame = dataset.to_dataframe().reset_index()[[TIME_FIELD, var_name]]
    return data_frame.rename(columns={var_name: short_name})


def process_data():
    df_temp = load_variable(TEMPERATURE_FIELD_KELVIN)
    df_co2 = load_variable(CO2_FIELD)
    df_temp[TIME_FIELD] = pd.to_datetime(df_temp[TIME_FIELD])
    df_co2[TIME_FIELD] = pd.to_datetime(df_co2[TIME_FIELD])
    df_temp = df_temp.set_index(TIME_FIELD).resample("MS").mean().reset_index()
    df_co2 = df_co2.set_index(TIME_FIELD).resample("MS").mean().reset_index()
    data_frame = pd.merge(df_temp, df_co2, on=TIME_FIELD)
    data_frame = data_frame.dropna()
    data_frame[TEMPERATURE_FIELD_CELSIUS] = data_frame[TEMPERATURE_FIELD_KELVIN] - 273.15
    data_frame[CO2_FIELD_PPM] = data_frame[CO2_FIELD] * 1_000_000
    return data_frame[[TIME_FIELD, TEMPERATURE_FIELD_CELSIUS, CO2_FIELD_PPM]].set_index(TIME_FIELD)


def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length - N_PREDICT):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + N_PREDICT])
    return torch.tensor(np.array(x), dtype=torch.float32).to(device), torch.tensor(np.array(y), dtype=torch.float32).to(
        device)


class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def display(dates, real, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, real[TEMPERATURE_FIELD_CELSIUS], label="Temperatura reala", color="tab:blue", marker='o')
    plt.plot(dates, predicted[TEMPERATURE_FIELD_CELSIUS], label="Temperatura prezisa", color="tab:blue", linestyle="--",
             marker='x')
    plt.plot(dates, real[CO2_FIELD_PPM] - 600, label="CO2 real - 600", color="tab:green", marker='o')
    plt.plot(dates, predicted[CO2_FIELD_PPM] - 600, label="CO2 prezis - 600", color="tab:green", linestyle="--",
             marker='x')
    plt.title("Temperatura si CO2 - real vs prezis")
    plt.xlabel("Timp")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def time_series_cross_validation(data, model_params, epochs, lr, splits=3):
    tscv = TimeSeriesSplit(n_splits=splits)
    losses = []

    x_all, y_all = create_sequences(data, SEQ_LENGTH)

    for train_idx, val_idx in tscv.split(x_all):
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_val, y_val = x_all[val_idx], y_all[val_idx]

        model = LSTMModel(**model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(epochs):
            model.train()
            output = model(x_train)
            target = y_train[:, 0, :]
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(x_val)
            val_target = y_val[:, 0, :]
            val_loss = criterion(val_output, val_target).item()
            losses.append(val_loss)

    return np.mean(losses)


def grid_search_time_series(data):
    param_combinations = list(product(*GRID_PARAMS.values()))
    best_score = float("inf")
    best_model = None
    best_params = None

    for hidden_size, num_layers, lr, epochs in param_combinations:
        print(f"Trying: hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}, epochs={epochs}")
        params = {'hidden_size': hidden_size, 'num_layers': num_layers}
        avg_loss = time_series_cross_validation(data, params, epochs, lr)
        print(f"Avg CV loss: {avg_loss:.6f}")

        if avg_loss < best_score:
            best_score = avg_loss
            best_params = (hidden_size, num_layers, lr, epochs)
            best_model = LSTMModel(**params).to(device)
            optimizer = torch.optim.Adam(best_model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            x_tensor, y_tensor = create_sequences(data, SEQ_LENGTH)
            for _ in range(epochs):
                best_model.train()
                output = best_model(x_tensor)
                target = y_tensor[:, 0, :]
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print("Best parameters:", best_params)
    return best_model


def predict(data, dates, model, scaler):
    model.eval()
    predictions = []
    input_seq = data[-SEQ_LENGTH:]
    for _ in range(N_PREDICT):
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()
        predictions.append(prediction[0])
        input_seq = np.vstack((input_seq[1:], prediction))
    predictions = np.array(predictions)
    prediction_real = scaler.inverse_transform(predictions)
    return pd.DataFrame(prediction_real, columns=[TEMPERATURE_FIELD_CELSIUS, CO2_FIELD_PPM], index=dates)


if __name__ == "__main__":
    dataframe = process_data()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(dataframe.values)

    model = grid_search_time_series(data_scaled)

    prediction_dates = dataframe.index[-N_PREDICT:]
    real_values = dataframe.loc[prediction_dates].copy()
    predicted_values = predict(data_scaled, prediction_dates, model, scaler)

    display(prediction_dates, real_values, predicted_values)
