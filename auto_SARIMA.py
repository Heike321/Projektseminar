import pandas as pd
import plotly.express as px
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

# Load airport metadata
airports_df = pd.read_csv("airports.dat")

# Extract top N most frequent routes by passenger volume
def compute_top_routes(df, top_n=10):
    parts = df['con_key'].str.split('-', expand=True)
    df['ORIGIN'] = parts[2]
    df['DEST'] = parts[3]

    if "ORIGIN_NAME" in df.columns and "DEST_NAME" in df.columns:
        df['ROUTE'] = df['ORIGIN_NAME'] + " → " + df['DEST_NAME']
    else:
        df['ROUTE'] = df['ORIGIN'] + " → " + df['DEST']

    monthly_data = df.groupby(['YEAR', 'MONTH', 'ROUTE'])['PASSENGERS'].sum().reset_index()
    total_passengers = monthly_data.groupby('ROUTE')['PASSENGERS'].sum().reset_index()
    return total_passengers.sort_values(by='PASSENGERS', ascending=False).head(top_n)

# Plot long-term trend in passenger data
def get_trend_plot(df):
    fig = go.Figure()
    df = df.copy()
    df = df.sort_values('DATE')
    df = df.set_index('DATE')

    if df['PASSENGERS'].isnull().any():
        df = df.dropna(subset=['PASSENGERS'])

    ts = df['PASSENGERS']

    if len(ts) < 24:
        # Add warning if data is insufficient for decomposition
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Not enough data to compute trend (need at least 24 months)",
            showarrow=False,
            font=dict(size=16),
            xref="paper",
            yref="paper"
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    model_type = "additive" if (ts <= 0).any() else "multiplicative"
    decomposition = seasonal_decompose(ts, model=model_type, period=12)

    fig.add_trace(go.Scatter(
        x=decomposition.trend.index,
        y=decomposition.trend.values,
        mode="lines",
        name="Trend",
        line=dict(color='#9467bd'),
    ))

    fig.update_layout(
        title="Passenger Trend",
        xaxis_title="Date",
        xaxis=dict(showgrid=False, zeroline=False, showline=True),
        yaxis_title="Passengers",
        yaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)', gridwidth=1, griddash='dot',
                   zeroline=False, showline=False),
        plot_bgcolor="#222222",
        paper_bgcolor="#111111",
        font_color="white"
    )

    return fig

# Visualize seasonality using monthly box plots
def get_seasonality_plot(df):
    monthly = df.groupby(["YEAR", "MONTH"])["PASSENGERS"].sum().reset_index()
    fig = px.box(monthly, x="MONTH", y="PASSENGERS", title="Seasonal Pattern of Passengers by Month")
    return fig

# Identify statistical outliers based on IQR
def get_outliers_plot(df):
    q1 = df["PASSENGERS"].quantile(0.25)
    q3 = df["PASSENGERS"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df["OUTLIER"] = (df["PASSENGERS"] < lower) | (df["PASSENGERS"] > upper).copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["DATE"], y=df["PASSENGERS"], mode='lines+markers', name='Passengers'))
    fig.add_trace(go.Scatter(x=df[df["OUTLIER"]]["DATE"], y=df[df["OUTLIER"]]["PASSENGERS"],
                             mode='markers', name='Outliers', marker=dict(color='red', size=10)))
    fig.update_layout(title="Outliers in Passengers", xaxis_title="Date", yaxis_title="Passengers")
    return fig

# Perform forecast evaluation per route using Holt-Winters and AutoARIMA
def generate_route_insights(df):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    insights = []

    df["DATE"] = pd.to_datetime(df["DATE"])
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]

    all_routes = df["ROUTE"].unique()

    for route in all_routes:
        route_df = df[df["ROUTE"] == route].sort_values("DATE")

        if route_df["PASSENGERS"].isnull().any() or len(route_df) < 36:
            continue

        # Compute linear trend slope
        y = route_df["PASSENGERS"].values
        x = np.arange(len(y))
        slope, *_ = np.polyfit(x, y, 1)

        # STL decomposition for seasonality and residual analysis
        ts = route_df.set_index("DATE")["PASSENGERS"]
        stl = STL(ts, period=12)
        res = stl.fit()

        avg_passengers = ts.mean()
        season_amp = res.seasonal.max() - res.seasonal.min()
        season_amp_pct = (season_amp / avg_passengers) * 100

        resid = res.resid
        q1, q3 = np.percentile(resid, [25, 75])
        iqr = q3 - q1
        outliers = ((resid < (q1 - 1.5 * iqr)) | (resid > (q3 + 1.5 * iqr))).sum()

        # Holt-Winters forecasting
        try:
            train_hw = route_df[route_df["DATE"].dt.year < 2024]
            valid_hw = route_df[route_df["DATE"].dt.year == 2024]
            ts_hw = train_hw.set_index("DATE")["PASSENGERS"]
            ts_hw.index.freq = 'MS'

            if len(ts_hw) < 36:
                print(f"Not enough data for Holt-Winters: {route}")

            model_hw = ExponentialSmoothing(ts_hw, trend='add', seasonal='add', seasonal_periods=12)
            fit_hw = model_hw.fit()
            forecast_hw = fit_hw.forecast(12)

            mae_hw = mean_absolute_error(valid_hw["PASSENGERS"], forecast_hw)

        except Exception as e:
            print(f"Holt-Winters error for {route}: {e}")
            mae_hw = np.nan

        # AutoARIMA forecasting with StatsForecast
        try:
            train_sarima = route_df[route_df["DATE"] < "2024-01-01"]
            valid_sarima = route_df[(route_df["DATE"] >= "2024-01-01") & (route_df["DATE"] < "2025-01-01")]

            ts_sf = train_sarima.copy()
            ts_sf = ts_sf.rename(columns={"DATE": "ds", "PASSENGERS": "y"})
            ts_sf["unique_id"] = route

            sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="MS", n_jobs=1)
            forecast_df = sf.forecast(df=ts_sf, h=12)

            # Auto-detect column name for the forecast
            forecast_column = forecast_df.columns.difference(["unique_id", "ds"])[0]
            forecast_sarima = forecast_df[forecast_column]

            if len(forecast_sarima) == len(valid_sarima):
                mae_sarima = mean_absolute_error(valid_sarima["PASSENGERS"].values, forecast_sarima.values)
            else:
                mae_sarima = np.nan
        except Exception as e:
            print(f"AutoARIMA error for {route}: {e}")
            mae_sarima = np.nan

        # Append insights for the route
        insights.append({
            "route": route,
            "trend_slope": round(slope, 2),
            "season_amp_pct": round(season_amp_pct, 1),
            "outlier_count": int(outliers),
            "mae_holt": round(mae_hw, 1) if not np.isnan(mae_hw) else np.nan,
            "mae_sarima": round(mae_sarima, 1) if not np.isnan(mae_sarima) else np.nan,
            "quotient_holt": round(mae_hw / slope, 3) if (not np.isnan(mae_hw) and slope != 0) else np.nan,
            "quotient_sarima": round(mae_sarima / slope, 3) if (not np.isnan(mae_sarima) and slope != 0) else np.nan
        })

    df_result = pd.DataFrame(insights)
    df_result = df_result.sort_values("trend_slope", ascending=False).reset_index(drop=True)
    df_result.to_csv("Data/precomputed_route_insights.csv", index=False)
    return df_result

# Main execution entrypoint for loading and processing data
if __name__ == "__main__":
    df = pd.read_csv("Data/Grouped_All_Valid_Connections.csv", low_memory=False)
    df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01")
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]
    generate_route_insights(df)
