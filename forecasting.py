import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def load_historical_data(file_path):
    #Load combined CSV and filter for historical years (2022 and 2023).
    
    df = pd.read_csv(file_path)
    historical = df[df["YEAR"].isin([2022, 2023])].copy()
    historical["DATE"] = pd.to_datetime(historical["YEAR"].astype(str) + "-" + 
                                        historical["MONTH"].astype(str).str.zfill(2) + "-01")
    historical["LOAD_FACTOR"] = historical["PASSENGERS"] / historical["SEATS"]

    return historical

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

