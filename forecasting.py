import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_passengers(df, periods=12):
    # using Holt-Winters exponential smoothing.
    df = df.sort_values("DATE")
    df = df.set_index("DATE")
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

def load_historical_data(file_path):
    #Load combined CSV and filter for historical years (2022 and 2023).
    
    df = pd.read_csv(file_path)
    historical = df[df["YEAR"].isin([2022, 2023])].copy()
    historical["DATE"] = pd.to_datetime(historical["YEAR"].astype(str) + "-" + 
                                        historical["MONTH"].astype(str).str.zfill(2) + "-01")
    return historical

def forecast_all_connections(file_path, periods=12):
    #Load historical data from file_path and generate forecasts for all connections.
    
    historical_df = load_historical_data(file_path)
    all_forecasts = []
    
    connections = historical_df[["ORIGIN", "DEST"]].drop_duplicates()
    
    for _, row in connections.iterrows():
        origin = row["ORIGIN"]
        dest = row["DEST"]
        
        conn_data = historical_df[(historical_df["ORIGIN"] == origin) & (historical_df["DEST"] == dest)]
        ts_df = conn_data[["DATE", "PASSENGERS"]].sort_values("DATE")
        
        # Skip connections with too few data points
        if len(ts_df) < 24:
            continue
        
        forecast_df = forecast_passengers(ts_df, periods=periods)
        forecast_df["ORIGIN"] = origin
        forecast_df["DEST"] = dest
        
        all_forecasts.append(forecast_df)
    
    result_df = pd.concat(all_forecasts, ignore_index=True)
    return result_df

if __name__ == "__main__":
    data_file = "Data/Grouped_All_Valid_Connections.csv"
    forecasts = forecast_all_connections(data_file)
    
