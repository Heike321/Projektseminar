import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings 

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




def sarima_forecast(df, start_train='2022-01-01', valid_start='2024-01-01', pred_start='2025-01-01', periods=12):
   
    # Sort and reset index for consistency
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Split data into training (before validation), validation, and full training (before prediction)
    train_initial = df[df['DATE'] < valid_start]
    valid_2024 = df[(df['DATE'] >= valid_start) & (df['DATE'] < pred_start)]
    full_train = df[df['DATE'] < pred_start]

    try:
        # Fit SARIMA model on initial training data
        model_initial = SARIMAX(train_initial['PASSENGERS'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit_initial = model_initial.fit(disp=False)

        # Forecast validation period (2024)
        forecast_valid = model_fit_initial.get_forecast(steps=len(valid_2024))
        forecast_df_2024 = pd.DataFrame({
            'DATE': valid_2024['DATE'].values,
            'VALUE': forecast_valid.predicted_mean,
            'TYPE': 'Forecast 2024'
        })

        # Calculate validation errors
        mae = mean_absolute_error(valid_2024['PASSENGERS'], forecast_valid.predicted_mean)
        rmse = np.sqrt(mean_squared_error(valid_2024['PASSENGERS'], forecast_valid.predicted_mean))
        error_text = f"📏 MAE (2024): {mae:.0f} passengers | RMSE: {rmse:.0f}"

        # Retrain SARIMA model on full training data including validation period
        model_final = SARIMAX(full_train['PASSENGERS'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit_final = model_final.fit(disp=False)

        # Forecast future period (2025)
        forecast_2025 = model_fit_final.get_forecast(steps=periods)
        forecast_df_2025 = pd.DataFrame({
            'DATE': pd.date_range(start=pred_start, periods=periods, freq='MS'),
            'VALUE': forecast_2025.predicted_mean,
            'TYPE': 'Forecast 2025'
        })

    except Exception as e:
        # Handle any errors during model fitting or forecasting
        forecast_df_2024 = pd.DataFrame(columns=['DATE', 'VALUE', 'TYPE'])
        forecast_df_2025 = pd.DataFrame(columns=['DATE', 'VALUE', 'TYPE'])
        error_text = f"Error during model fitting or forecasting: {e}"

    # Prepare actual training and validation data with unified format
    real_train = train_initial.rename(columns={'PASSENGERS': 'VALUE'}).assign(TYPE='Training data')
    real_valid = valid_2024.rename(columns={'PASSENGERS': 'VALUE'}).assign(TYPE='Actual 2024')

    # Combine all data for plotting or further processing
    combined_df = pd.concat([real_train, real_valid, forecast_df_2024, forecast_df_2025], ignore_index=True)

    #return combined_df, error_text
    return real_train.reset_index(), real_valid.reset_index(), forecast_df_2024, forecast_df_2025, error_text



