import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings 
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from pathlib import Path
import os
import pickle
import json


warnings.filterwarnings("ignore")
import re

def make_safe_filename(s):
    s = s.replace("→", "to")
    s = s.replace("|", "_")
    s = re.sub(r"\s+", "_", s)  # mehrere Leerzeichen zu einem Unterstrich
    s = re.sub(r"[^a-zA-Z0-9_\(\)\-]", "", s)  # nur alphanumerische, Unterstrich, Klammern, Bindestrich behalten
    s = re.sub(r"_+", "_", s)  # mehrere Unterstriche zu einem
    return s.strip("_")

def load_historical_data(file_path):
    #Load combined CSV and filter for historical years (2022 and 2023).
    
    df = pd.read_csv(file_path)
    historical = df[df["YEAR"].isin([2022, 2023])].copy()
    historical["DATE"] = pd.to_datetime(historical["YEAR"].astype(str) + "-" + 
                                        historical["MONTH"].astype(str).str.zfill(2) + "-01")
    historical["LOAD_FACTOR"] = historical["PASSENGERS"] / historical["SEATS"]

    return historical



def prepare_forecast_data(data, selected_route, selected_airline):
    df = data.copy()
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]
    df = df[df["ROUTE"] == selected_route]

    # If no airline selected, or "all"
    if not selected_airline or selected_airline.lower() == "all":
        # Aggregate all airlines on a monthly basis
        df = (
            df.groupby("DATE", as_index=False)
            .agg({"PASSENGERS": "sum", "SEATS": "sum"})
        )
        df["LOAD_FACTOR"] = df["PASSENGERS"] / df["SEATS"]
        df["YEAR"] = df["DATE"].dt.year
        df["MONTH"] = df["DATE"].dt.month

        return df
    
    if "_" in selected_airline:
        carrier_code, aircraft_type = selected_airline.split("_")
        df = df[
            (df["UNIQUE_CARRIER_NAME"] == carrier_code)
            & (df["AIRCRAFT_TYPE"].astype(str) == aircraft_type)
        ]
    if df.empty:
        raise ValueError(f"No data for airline '{selected_airline}' on this route.")

    df["LOAD_FACTOR"] = df["PASSENGERS"] / df["SEATS"]
    return df



def forecast_load_factor(df, periods=12):
    df = df.sort_values("DATE")
    df = df.set_index("DATE")
    df.index.freq = 'MS'

    ts = df["LOAD_FACTOR"]

    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(periods)
    forecast_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')

    forecast_df = pd.DataFrame({
        "DATE": forecast_dates,
        "FORECAST_LOAD_FACTOR": forecast.values
    })
    return forecast_df


def forecast_passengers(df, periods=12):
    # using Holt-Winters exponential smoothing.
    df = df.sort_values("DATE")
    df = df.set_index("DATE")
    df.index.freq = 'MS'
    

    ts = df["PASSENGERS"]

    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(periods)
    forecast_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')

    forecast_df = pd.DataFrame({
        "DATE": forecast_dates,
        "FORECAST_PASSENGERS": forecast.values
    })
    
    return forecast_df

def get_forecast_for_year(df, target_year, periods=12):
    
    #Given historical data with columns including 'DATE', 'PASSENGERS', 'LOAD_FACTOR', generate forecast for a target year.
    #Only data before target_year is used to forecast.
    #Returns merged DataFrame with forecasted passengers and load factor for target_year.
    
    historical = df[df['DATE'].dt.year < target_year].copy()
    
    pax_forecast_df = forecast_passengers(historical[['DATE', 'PASSENGERS']], periods=periods)
    lf_forecast_df = forecast_load_factor(historical[['DATE', 'LOAD_FACTOR']], periods=periods)

    # Filter forecast to only the target year months (in case forecast spills beyond)
    pax_forecast_df = pax_forecast_df[pax_forecast_df['DATE'].dt.year == target_year]
    lf_forecast_df = lf_forecast_df[lf_forecast_df['DATE'].dt.year == target_year]

    forecast_df = pax_forecast_df.merge(lf_forecast_df, on='DATE', how='left')
    return forecast_df

# SARIMAX

def sarima_forecast_load_factor(df, forecast_year, periods=12):
    df = df.copy()
    df = df.sort_values('DATE')
    df.index = pd.to_datetime(df['DATE'])
    df.index.freq = 'MS'

    # Only historical data vor dem Vorhersagejahr
    ts = df[df.index.year < forecast_year]['LOAD_FACTOR']

    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fit = model.fit(disp=False)

    forecast_index = pd.date_range(start=f"{forecast_year}-01-01", periods=periods, freq='MS')
    forecast_values = fit.get_forecast(steps=periods).predicted_mean

    return pd.DataFrame({
        "DATE": forecast_index,
        "FORECAST_LOAD_FACTOR": forecast_values
    })



def sarima_forecast(df, forecast_year, route=None, airline=None,aircraft_type=None, periods=12, save=False):
    df = df.sort_values('DATE').reset_index(drop=True)
    df["DATE"] = pd.to_datetime(df["DATE"])
    train_df = df[df['DATE'].dt.year < forecast_year].copy()
    ts = train_df.set_index("DATE")["PASSENGERS"].asfreq("MS")

    forecast_df = pd.DataFrame(columns=["DATE", "VALUE", "TYPE"])
    fallback_triggered = False

    # Try AutoARIMA first
    try:
        train_arima = train_df[['DATE', 'PASSENGERS']].copy()
        train_arima.columns = ['ds', 'y']
        train_arima['unique_id'] = 'series'
        train_arima = train_arima[['unique_id', 'ds', 'y']]

        
        sf = StatsForecast(models=[AutoARIMA(season_length=12, stepwise=True, seasonal=True, approximation=False, max_order=10)], freq='MS')
        forecast = sf.forecast(df=train_arima, h=periods)
        
        # Check if forecast is flat → then fallback (nunique() <= 1: all values are exactly the same,std() < 1e-3: values are nearly identical (low variation))
        if forecast['AutoARIMA'].nunique() <= 3 or forecast['AutoARIMA'].std() < 1e-3:
            fallback_triggered = True
            raise Exception("Flat forecast, fallback to SARIMA")

        # Build forecast_df from AutoARIMA
        forecast_dates = pd.date_range(start=ts.index.max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
        forecast_df = pd.DataFrame({
            "DATE": forecast_dates,
            "VALUE": forecast['AutoARIMA'].values,
            "TYPE": f"Forecast {forecast_year} (AutoARIMA)"
        })
        
    except:
        # Try fallback with predefined SARIMA parameters
        try:
            with open("custom_sarima_params.json") as f:
                param_config = json.load(f)
            key = f"{route} | {airline}" if airline else route
            if key in param_config:
                order = tuple(param_config[key]["order"])
                seasonal_order = tuple(param_config[key]["seasonal_order"])
            else:
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 12)
        except:
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)

        try:
            model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
            fit = model.fit(disp=False)
            forecast_index = pd.date_range(start=f"{forecast_year}-01-01", periods=periods, freq='MS')
            forecast_values = fit.get_forecast(steps=periods).predicted_mean
            forecast_df = pd.DataFrame({
                "DATE": forecast_index,
                "VALUE": forecast_values,
                "TYPE": f"SARIMA Fallback {forecast_year}"
            })
        except:
            forecast_df = pd.DataFrame(columns=["DATE", "VALUE", "TYPE"])

    if forecast_year == 2024 and save:
        try:
            valid_data = df[df['DATE'].dt.year == 2024]
            train_data = df[df['DATE'].dt.year < 2024]
            forecast_valid = forecast_df.copy()
            
   
            # Nur relevanten Forecast-Fehler-Speicher
            route_key = f"{route} | {airline}| {aircraft_type}" if airline else route 
            safe_route_key = make_safe_filename(route_key)
            folder = "saved_forecasts"
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = f"saved_forecasts/{safe_route_key}_2024.pkl"
            with open(save_path, "wb") as f:
                pickle.dump({
                    "train_data": train_data,
                    "valid_data": valid_data,
                    "forecast_valid": forecast_valid
                }, f)
        except Exception as e:
            print("Error saving forecast evaluation data:", e)

    
    return forecast_df.reset_index(drop=True), f"Fallback used: {fallback_triggered}"
    
'''
# Auto ARIMA
def sarima_forecast_load_factor(df, forecast_year, periods=12):
    df = df.copy()
    df = df.sort_values('DATE')
    df.index = pd.to_datetime(df['DATE'])
    df.index.freq = 'MS'
    try:
        ts = df[df.index.year < forecast_year][['LOAD_FACTOR']].copy()
        ts = ts.reset_index()
        ts.columns = ['ds', 'y']
        ts['unique_id'] = 'series'
        ts = ts[['unique_id', 'ds', 'y']]
        sf = StatsForecast(models=[AutoARIMA(season_length=12, stepwise=True,seasonal =True, approximation=False, max_order=10)], freq='MS')
        forecast = sf.forecast(df=ts, h=periods)
        forecast_values = forecast['AutoARIMA'].values
        forecast_index = pd.date_range(start=f"{forecast_year}-01-01", periods=periods, freq='MS')
        return pd.DataFrame({"DATE": forecast_index, "FORECAST_LOAD_FACTOR": forecast_values})
    except Exception as e:
        return pd.DataFrame(columns=["DATE", "FORECAST_LOAD_FACTOR"])
'''
#if __name__ == "__main__":

"""
# Beispielhafte Testdaten generieren
date_rng = pd.date_range(start="2022-01-01", end="2024-12-01", freq='MS')
test_df = pd.DataFrame({
    "DATE": date_rng,
    "PASSENGERS": np.random.randint(10000, 50000, size=len(date_rng))
})
# Optional: Lade zusätzlich den SEATS und berechne LOAD_FACTOR, falls notwendig
test_df["SEATS"] = test_df["PASSENGERS"] * 1.2
test_df["LOAD_FACTOR"] = test_df["PASSENGERS"] / test_df["SEATS"]

# Funktion aufrufen
train, valid, fc_2024, fc_2025, err = sarima_forecast(test_df) 

# Ergebnisse ausgeben
print("\n--- TEST ---")
print("Fehlermeldung / Status:", err)
print("Forecast 2024:\n", fc_2024.head())
print("Forecast 2025:\n", fc_2025.head())

    #Testing
import matplotlib.pyplot as plt


# Testdaten nochmal setzen
forecast_year = 2025
test_df["DATE"] = pd.to_datetime(test_df["DATE"])
test_df = test_df.sort_values("DATE")
test_df = test_df.set_index("DATE")
test_df.index.freq = 'MS'

# Zeitreihe extrahieren
ts = test_df[test_df.index.year < forecast_year]['LOAD_FACTOR']

# Standardabweichung und Plot
print("\nStandardabweichung LOAD_FACTOR:", ts.std())
ts.plot(title="LOAD_FACTOR Zeitreihe")
plt.xlabel("Datum")
plt.ylabel("Load Factor")
plt.grid(True)
plt.show()

"""
'''
#Auto Arima Parameter
if __name__ == "__main__":
    # 1. Define the path to the input CSV file with historical flight connection data.
    date_rng = pd.date_range(start="2022-01-01", end="2023-12-01", freq='MS')
    test_df = pd.DataFrame({
        "DATE": date_rng,
        "PASSENGERS": np.random.randint(10000, 50000, size=len(date_rng))
    })
    test_df["SEATS"] = test_df["PASSENGERS"] * 1.2
    test_df["LOAD_FACTOR"] = test_df["PASSENGERS"] / test_df["SEATS"]

    # Testvorhersage
    forecast_df, msg, order, seasonal_order = sarima_forecast(test_df, forecast_year=2024)

    print("\n--- Forecast 2024 ---")
    print(msg)
    print("AutoARIMA order:", order)
    print("AutoARIMA seasonal_order:", seasonal_order)
    print(forecast_df.head())

if __name__ == "__main__":
    # 1. Define the path to the input CSV file with historical flight connection data.
    file_path = "Data/Grouped_All_Valid_Connections.csv"  # Datei mit Flugverbindungen

    # 2. Choose a specific route and airline to forecast.
    route = "FRA → JFK"
    airline = "all"  # oder None bzw. "all" für alle Airlines

    # 3. Load the historical data (only years 2022 and 2023 are considered).
    data = load_historical_data(file_path)

    # 4. Filter and prepare the data for the selected route and airline.
    # If airline = "all", data is aggregated monthly across all airlines.
    prepared = prepare_forecast_data(data, selected_route=route, selected_airline=airline)

    # 5. Run the forecast for the year 2024 using the sarima_forecast function.
    # Internally, this first tries AutoARIMA and only falls back to manual SARIMA if needed.
    forecast_df, msg, order, seasonal_order = sarima_forecast(
        prepared,                # the time series data to forecast on
        forecast_year=2024,      # the target year for which we want predictions
        route=route,             # used for saving results and fallback config
        airline=airline          # used for fallback key
    )

    # 6. Print the results
    print("\n--- Forecast 2024 für echte Route ---")
    print("Route:", route)
    print("Airline:", airline)
    print(msg)
    print("AutoARIMA order:", order)
    print("AutoARIMA seasonal_order:", seasonal_order)
    print(forecast_df.head())
'''