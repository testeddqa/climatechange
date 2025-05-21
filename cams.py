import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.ticker as ticker

filename = "cams_egg4.grib"


# Load variable value by shortName
def load_variable(short_name):
    ds = xr.open_dataset(
        filename,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": short_name}}
    )
    var_name = list(ds.data_vars)[0]
    df = ds.to_dataframe().reset_index()[["time", var_name]]
    return df.rename(columns={var_name: short_name})


df_temp = load_variable("2t")  # extract temperature - average hourly values
df_co2 = load_variable("co2")  # extract CO2 - monthly values
# Compute temperature monthly values
df_co2["time"] = pd.to_datetime(df_co2["time"])
df_co2_monthly = df_co2.set_index("time").resample("MS").mean().reset_index()

# DateTime conversion and sorting by time
df_temp["time"] = pd.to_datetime(df_temp["time"])
df_temp = df_temp.sort_values("time")
df_co2_monthly = df_co2_monthly.sort_values("time")

# Merge temperature and CO2 in the same dataset
df = pd.merge(df_temp, df_co2_monthly, on="time")

df["2t"] = df["2t"] - 273.15  # Convert temperature Kelvin -> Celsius
df["co2_ppm"] = df["co2"] * 1000000  # Convert CO2 concentration 1 mol/mol = 1.000.000 ppm (part per million)

df["year"] = df["time"].dt.year
annual_avg_temp = df.groupby("year")["2t"].mean().reset_index()
annual_avg_co2 = df.groupby("year")["co2_ppm"].mean().reset_index()

combined_df = pd.merge(annual_avg_temp, annual_avg_co2, on="year")
combined_df.columns = ["year", "temp", "co2"]

X = combined_df["year"].values.reshape(-1, 1)

# Temperature linear regression
temp_model = LinearRegression().fit(X, combined_df["temp"])
temp_trend = temp_model.predict(X)

# CO2 linear regression
co2_model = LinearRegression().fit(X, combined_df["co2"])
co2_trend = co2_model.predict(X)

# Compute temperature and CO2 correlation
r, p_value = pearsonr(combined_df["temp"], combined_df["co2"])
print(f"Coeficient Pearson: r = {r:.3f}")
print(f"Valoare p (semnificatie statistica): p = {p_value:.3e}")

plt.figure(figsize=(14, 6))

# Temperature
plt.plot(combined_df["year"], combined_df["temp"], label="Temp. medie anuala (°C)", color="tab:red", marker="o")
plt.plot(combined_df["year"], temp_trend, linestyle="--", color="tab:red",
         label=f"Trend Temp. (Δ={temp_model.coef_[0]:.3f}°C/an)")

# CO2
plt.plot(combined_df["year"], combined_df["co2"]/100, label="CO2 mediu anual (ppm)/100", color="tab:green", marker="o")
plt.plot(combined_df["year"], co2_trend/100, linestyle="--", color="tab:green",
         label=f"Trend CO2 (Δ={co2_model.coef_[0]:.2f} ppm/an)")

# Correlation
# textstr = f"Coef. Pearson: {r:.3f}\nValoare p: {p_value:.1e}"
plt.text(combined_df["year"].min() + 0.5, combined_df["temp"].max() - 0.5,
         f"Coef. Pearson: {r:.3f}\nValoare p: {p_value:.1e}",
         fontsize=11, bbox=dict(facecolor='white', edgecolor='gray'))

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xticks(rotation=45)
plt.title("Evolutia temperaturii si concentratiei CO2 (medii anuale)")
plt.xlabel("An")
plt.ylabel("Valoare")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
