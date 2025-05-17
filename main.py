import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

FILENAME = "data2.grib"
SEQ_LENGTH = 240  # Number of monthly mean temperature values used as input for the RNN network
N_PREDICT = 12  # Number of predicted monthly mean temperature values
EPOCHS = 500
TEMPERATURE_FIELD_KELVIN = 't2m'
TEMPERATURE_FIELD_CELSIUS = 't2m_c'
TIME_FIELD = 'time'


def prepareDataframe(dataset):
    # Transform to DataFrame and convert the temperature from Kelvin to Celsius
    dataframe = dataset[TEMPERATURE_FIELD_KELVIN].to_dataframe().reset_index()
    dataframe[TEMPERATURE_FIELD_CELSIUS] = dataframe[TEMPERATURE_FIELD_KELVIN] - 273.15
    dataframe[TIME_FIELD] = pd.to_datetime(dataframe[TIME_FIELD])
    dataframe = dataframe.set_index(TIME_FIELD)
    return dataframe


def processMeanTemperature(dataframe):
    plt.figure(figsize=(14, 8))
    colors = plt.cm.get_cmap('tab10', 12)  # 12 colors
    for month in range(1, 13):
        df_month = dataframe[dataframe.index.month == month]
        monthly_avg_by_year = df_month.groupby(df_month.index.year)[TEMPERATURE_FIELD_CELSIUS].mean()

        years = monthly_avg_by_year.index.values.reshape(-1, 1)

        monthly_regression_model = LinearRegression()
        monthly_regression_model.fit(years, monthly_avg_by_year.values)
        trend = monthly_regression_model.predict(years)

        temp_1940 = monthly_regression_model.predict(np.array([[1940]]))[0]
        temp_2025 = monthly_regression_model.predict(np.array([[2025]]))[0]
        delta = temp_2025 - temp_1940

        month_name = pd.to_datetime(f'2023-{month:02d}-01').strftime('%B')
        label = f"{month:02d} - {month_name} (Δ = {delta:+.2f}°C)"

        plt.plot(years.flatten(), trend, label=label, color=colors(month - 1))
        plt.text(years[-1][0] + 0.2, trend[-1], f'{month:02d}', fontsize=8, color=colors(month - 1))

    annual_avg_by_year = dataframe.groupby(dataframe.index.year)[TEMPERATURE_FIELD_CELSIUS].mean()
    years = annual_avg_by_year.index.values.reshape(-1, 1)
    yearly_regression_model = LinearRegression()
    yearly_regression_model.fit(years, annual_avg_by_year.values)
    trend = yearly_regression_model.predict(years)
    temp_1940 = yearly_regression_model.predict(np.array([[1940]]))[0]
    temp_2025 = yearly_regression_model.predict(np.array([[2025]]))[0]
    delta = temp_2025 - temp_1940
    plt.plot(years.flatten(), trend, label=f"00 - Annual (Δ = {delta:+.2f}°C)", color='black', linewidth=2,
             linestyle='--')
    plt.text(years[-1][0] + 0.2, trend[-1], '0', fontsize=11, color='black')
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))  # Y axis: step 1
    plt.title('Mean monthly and yearly temperature evolution (1940–2025)')
    plt.xlabel('Year')
    plt.ylabel('Mean temperature (°C)')
    plt.grid(True)
    plt.legend(title="Month (Δ = variation starting at 1940 to 2025)", ncol=3, fontsize='small')
    plt.tight_layout()
    plt.show()


def prepareData(frame):
    monthly_avg = frame[TEMPERATURE_FIELD_CELSIUS].resample('M').mean()
    data = monthly_avg.values.reshape(-1, 1)
    data_normalized = scaler.fit_transform(data)
    # Split data: train + test
    train_data = data_normalized[:-N_PREDICT]
    test_data = data_normalized[-(N_PREDICT + SEQ_LENGTH):]
    x, y = [], []
    for i in range(len(train_data) - SEQ_LENGTH):
        x.append(train_data[i:i + SEQ_LENGTH])
        y.append(train_data[i + SEQ_LENGTH])

    x_train, y_train = np.array(x), np.array(y)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_train_tensor = x_train_tensor.view(x_train_tensor.shape[0], x_train_tensor.shape[1], 1)
    return x_train_tensor, y_train_tensor, test_data, monthly_avg


def builModel():
    # LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, criterion, optimizer


def trainModel(model, criterion, optimizer, x_train_tensor, y_train_tensor):
    for epoch in range(EPOCHS):
        model.train()
        output = model(x_train_tensor)
        loss = criterion(output, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


def predict(model, test_data, monthly_avg):
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for i in range(N_PREDICT):
            input_seq = test_data[i:i + SEQ_LENGTH]
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).view(1, SEQ_LENGTH, 1).to(device)
            pred = model(input_tensor).cpu().numpy()
            test_predictions.append(pred[0][0])
    real_values = scaler.inverse_transform(test_data[SEQ_LENGTH:]).flatten()
    predicted_values = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
    timestamps = monthly_avg.index[-N_PREDICT:]
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, real_values, label="Actual temperatures", marker="o")
    plt.plot(timestamps, predicted_values, label="Predicted temperatures", linestyle="--", marker="x", color="red")
    plt.title("Monthly average temperatures")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    mae = mean_absolute_error(real_values, predicted_values)
    mse = mean_squared_error(real_values, predicted_values)
    rmse = np.sqrt(mse)

    print(f"MAE  = {mae:.2f} °C")
    print(f"MSE  = {mse:.2f}")
    print(f"RMSE = {rmse:.2f} °C")


if __name__ == "__main__":
    # Open GRIB file
    dataset = xr.open_dataset(FILENAME, engine="cfgrib")
    scaler = MinMaxScaler()
    df = prepareDataframe(dataset)
    processMeanTemperature(df)
    x_train_tensor, y_train_tensor, test_data, monthly_avg = prepareData(df)
    model, criterion, optimizer = builModel()
    trainModel(model, criterion, optimizer, x_train_tensor, y_train_tensor)
    predict(model, test_data, monthly_avg)
