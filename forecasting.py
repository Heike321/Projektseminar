import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings 
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from pathlib import Path

import json


warnings.filterwarnings("ignore")

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
    
    # If an airline is explicitly selected
    df = df[df["UNIQUE_CARRIER_NAME"] == selected_airline]

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

'''
# AutoARIMA
def sarima_forecast(df, start_train='2022-01-01', valid_start='2024-01-01', pred_start='2025-01-01', periods=12):
    # Sort and reset index for consistency
    df = df.sort_values('DATE').reset_index(drop=True)
    train_initial = df[df['DATE'] < valid_start]
    valid_2024 = df[(df['DATE'] >= valid_start) & (df['DATE'] < pred_start)]
    full_train = df[df['DATE'] < pred_start]

    try:
        # Forecast validation period (2024)
        df_valid_train = train_initial[['DATE', 'PASSENGERS']].copy()
        df_valid_train.columns = ['ds', 'y']
        df_valid_train['unique_id'] = 'series'
        df_valid_train = df_valid_train[['unique_id', 'ds', 'y']]

        autoarima_config = AutoARIMA(
            season_length=12,
            stepwise=True,  #nicht alle Modellkombinationen ausprobieren sondern heuristischer Suchalgorithmus
            approximation=False,
            max_order=15, # summe der parameter nicht über 15 , kein überfitten
            start_p=1, max_p=3, # 0 ausgeschlossen keine flache Linie
            start_q=0, max_q=3,
            max_d=1, # maximal eine differnzierung - verjhindert mehrfache und dass e zu glatt wird
            start_P=1, max_P=2, # saisonale Muster berücksichtigen
            start_Q=0, max_Q=2,
            max_D=1
        )        

        # Konfiguration anwenden
        sf_valid = StatsForecast(models=[autoarima_config], freq='MS')
        #sf_valid = StatsForecast(models=[autoarima_config], freq='MS', trace=True)

        sf_final = StatsForecast(models=[autoarima_config], freq='MS')

        # Forecast für Validierung
        forecast_valid = sf_valid.forecast(df=df_valid_train, h=len(valid_2024))
        forecast_df_2024 = forecast_valid.rename(columns={'ds': 'DATE', 'AutoARIMA': 'VALUE'})
        forecast_df_2024['TYPE'] = 'Forecast 2024'


    
        # sf_valid = StatsForecast(models=[AutoARIMA(season_length=12, stepwise=True, approximation=False, max_order=10)], freq='MS')
        #forecast_valid = sf_valid.forecast(df=df_valid_train, h=len(valid_2024))
        #forecast_df_2024 = forecast_valid.rename(columns={'ds': 'DATE', 'AutoARIMA': 'VALUE'})
        #forecast_df_2024['TYPE'] = 'Forecast 2024'

        # Calculate validation errors
        mae = mean_absolute_error(valid_2024['PASSENGERS'].values, forecast_valid['AutoARIMA'].values[:len(valid_2024)])
        rmse = np.sqrt(mean_squared_error(valid_2024['PASSENGERS'].values, forecast_valid['AutoARIMA'].values[:len(valid_2024)]))
        error_text = f"📏 MAE (2024): {mae:.0f} passengers | RMSE: {rmse:.0f}"

        # Forecast future period (2025)
        df_full_train = full_train[['DATE', 'PASSENGERS']].copy()
        df_full_train.columns = ['ds', 'y']
        df_full_train['unique_id'] = 'series'
        df_full_train = df_full_train[['unique_id', 'ds', 'y']]
        #sf_final = StatsForecast(models=[AutoARIMA(season_length=12, stepwise=True, approximation=False, max_order=10)], freq='MS')
        forecast_2025 = sf_final.forecast(df=df_full_train, h=periods)
        forecast_df_2025 = forecast_2025.rename(columns={'ds': 'DATE', 'AutoARIMA': 'VALUE'})
        forecast_df_2025['TYPE'] = 'Forecast 2025'

    except Exception as e:
        # Handle any errors during model fitting or forecasting
        forecast_df_2024 = pd.DataFrame(columns=['DATE', 'VALUE', 'TYPE'])
        forecast_df_2025 = pd.DataFrame(columns=['DATE', 'VALUE', 'TYPE'])
        error_text = f"Error during model fitting or forecasting: {e}"

    real_train = train_initial.rename(columns={'PASSENGERS': 'VALUE'}).assign(TYPE='Training data')
    real_valid = valid_2024.rename(columns={'PASSENGERS': 'VALUE'}).assign(TYPE='Actual 2024')
    return real_train.reset_index(), real_valid.reset_index(), forecast_df_2024, forecast_df_2025, error_text
'''

# SARIMAX
'''

def sarima_forecast_load_factor(df, forecast_year, periods=12):
    df = df.copy()
    df = df.sort_values('DATE')
    df.index = pd.to_datetime(df['DATE'])
    df.index.freq = 'MS'

    # Nur historische Daten vor dem Vorhersagejahr
    ts = df[df.index.year < forecast_year]['LOAD_FACTOR']

    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fit = model.fit(disp=False)

    forecast_index = pd.date_range(start=f"{forecast_year}-01-01", periods=periods, freq='MS')
    forecast_values = fit.get_forecast(steps=periods).predicted_mean

    return pd.DataFrame({
        "DATE": forecast_index,
        "FORECAST_LOAD_FACTOR": forecast_values
    })
'''


def sarima_forecast(df, forecast_year, route=None, airline=None, periods=12):
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

        sf = StatsForecast(models=[AutoARIMA(season_length=12, stepwise=True, approximation=False, max_order=10)], freq='MS')
        forecast = sf.forecast(df=train_arima, h=periods)
        # Check if forecast is flat → then fallback
        if forecast['AutoARIMA'].nunique() <= 1:
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

    
    return forecast_df.reset_index(drop=True), f"Fallback used: {fallback_triggered}"

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
        sf = StatsForecast(models=[AutoARIMA(season_length=12, stepwise=True, approximation=False, max_order=10)], freq='MS')
        forecast = sf.forecast(df=ts, h=periods)
        forecast_values = forecast['AutoARIMA'].values
        forecast_index = pd.date_range(start=f"{forecast_year}-01-01", periods=periods, freq='MS')
        return pd.DataFrame({"DATE": forecast_index, "FORECAST_LOAD_FACTOR": forecast_values})
    except Exception as e:
        return pd.DataFrame(columns=["DATE", "FORECAST_LOAD_FACTOR"])

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
