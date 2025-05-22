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
