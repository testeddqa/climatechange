import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

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
EPOCHS = 350

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
    data_frame[TEMPERATURE_FIELD_CELSIUS] = data_frame[
                                                TEMPERATURE_FIELD_KELVIN] - 273.15  # Convert temperature Kelvin -> Celsius
    data_frame[CO2_FIELD_PPM] = data_frame[
                                    CO2_FIELD] * 1_000_000  # Convert CO2 concentration 1 mol/mol = 1.000.000 ppm (part per million)
    data_frame = data_frame[[TIME_FIELD, TEMPERATURE_FIELD_CELSIUS, CO2_FIELD_PPM]].set_index(TIME_FIELD)
    return data_frame


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
        out = self.fc(out[:, -1, :])
        return out


def train(model):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(EPOCHS):
        model.train()
        output = model(x_tensor)
        target = y_tensor[:, 0, :]
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


def predict(dates):
    # Predict for last known N_PREDICT months
    start_index = - (N_PREDICT + SEQ_LENGTH)
    input_seq = data_scaled[start_index:start_index + SEQ_LENGTH]
    predictions = []
    model.eval()
    for _ in range(N_PREDICT):
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).cpu().numpy()
        predictions.append(pred[0])
        input_seq = np.vstack((input_seq[1:], pred))
    predictions = np.array(predictions)
    predicted_real = scaler.inverse_transform(predictions)
    return pd.DataFrame(predicted_real, columns=[TEMPERATURE_FIELD_CELSIUS, CO2_FIELD_PPM], index=dates)


def display(dates, real, predicted):
    # Combined plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, real[TEMPERATURE_FIELD_CELSIUS], label="Temperatura reala", color="tab:blue", marker='o')
    plt.plot(dates, predicted[TEMPERATURE_FIELD_CELSIUS], label="Temperatura prezisa", color="tab:blue",
             linestyle="--", marker='x')
    plt.plot(dates, real[CO2_FIELD_PPM] - 600, label="CO2 real - 600", color="tab:green", marker='o')
    plt.plot(dates, predicted[CO2_FIELD_PPM] - 600, label="CO2 prezis - 600", color="tab:green",
             linestyle="--",
             marker='x')
    plt.title("Temperatura si CO2 - real vs prezis")
    plt.xlabel("Timp")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataframe = process_data()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(dataframe.values)
    x_tensor, y_tensor = create_sequences(data_scaled, SEQ_LENGTH)
    model = LSTMModel().to(device)

    train(model)

    comparison_dates = dataframe.index[-N_PREDICT:]
    predicted_values = predict(comparison_dates)
    real_values = dataframe.loc[comparison_dates].copy()

    display(comparison_dates, real_values, predicted_values)
