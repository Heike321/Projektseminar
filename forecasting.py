# Data handling
import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
import re

# Time series models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")


def make_safe_filename(s):
    s = s.replace("→", "to")
    s = s.replace("|", "_")
    s = re.sub(r"\s+", "_", s)  
    s = re.sub(r"[^a-zA-Z0-9_\(\)\-]", "", s)  
    s = re.sub(r"_+", "_", s)  
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

    # Only historical data before the forecast year
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

    # Try AutoARIMA as the primary forecasting model
    try:
        train_arima = train_df[['DATE', 'PASSENGERS']].copy()
        train_arima.columns = ['ds', 'y']
        train_arima['unique_id'] = 'series'
        train_arima = train_arima[['unique_id', 'ds', 'y']]

        #Initialize and fit AutoARIMA model
        sf = StatsForecast(models=[AutoARIMA(season_length=12, stepwise=True, seasonal=True, approximation=False, max_order=10)], freq='MS')
        forecast = sf.forecast(df=train_arima, h=periods)
        
        # Check if forecast is flat → then fallback (std() < 1e-3: values are nearly identical (low variation))
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
        # If AutoARIMA fails, fallback to manually parameterized SARIMA
        try:
            with open("custom_sarima_params.json") as f:
                param_config = json.load(f)
            key = f"{route} | {airline}| {aircraft_type}" if airline else route 
            safe_key = make_safe_filename(key)
            if safe_key in param_config:
                order = tuple(param_config[safe_key]["order"])
                seasonal_order = tuple(param_config[safe_key]["seasonal_order"])
            else:
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 12)
        except:
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)

        try:
            # Fit SARIMA model and generate forecast
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
            # In case even SARIMA fails, return an empty DataFrame
            forecast_df = pd.DataFrame(columns=["DATE", "VALUE", "TYPE"])

    # If forecast year is 2024 and save=True → store forecast, training and validation data to file
    if forecast_year == 2024 and save:
        try:
            valid_data = df[df['DATE'].dt.year == 2024]
            train_data = df[df['DATE'].dt.year < 2024]
            forecast_valid = forecast_df.copy()
            
   
            # Create a filename-safe key for the route/airline/aircraft combination
            route_key = f"{route} | {airline}| {aircraft_type}" if airline else route 
            safe_route_key = make_safe_filename(route_key)
            folder = "Generated/saved_forecasts"
            if not os.path.exists(folder):
                os.makedirs(folder)
            # Save data as pickle file for future evaluation    
            save_path = f"Generated/saved_forecasts/{safe_route_key}_2024.pkl"
            with open(save_path, "wb") as f:
                pickle.dump({
                    "train_data": train_data,
                    "valid_data": valid_data,
                    "forecast_valid": forecast_valid
                }, f)
        except Exception as e:
            print("Error saving forecast evaluation data:", e)

    
    return forecast_df.reset_index(drop=True), f"Fallback used: {fallback_triggered}"
    