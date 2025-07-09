import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from forecasting import sarima_forecast, forecast_passengers


 

airports_df = pd.read_csv("airports.dat")  


def compute_top_routes(df, top_n=10):
    # Split connection key to extract origin and destination codes
    parts = df['con_key'].str.split('-', expand=True)
    df['ORIGIN'] = parts[2]
    df['DEST'] = parts[3]

    # Use full airport names if available; otherwise fall back to IATA codes
    if "ORIGIN_NAME" in df.columns and "DEST_NAME" in df.columns:
        df['ROUTE'] = df['ORIGIN_NAME'] + " → " + df['DEST_NAME']
    else:
        df['ROUTE'] = df['ORIGIN'] + " → " + df['DEST']

    # Step 1: Aggregate monthly passenger totals per route
    monthly_data = df.groupby(['YEAR', 'MONTH', 'ROUTE'])['PASSENGERS'].sum().reset_index()

    # Step 2: Sum across all months and years
    total_passengers = monthly_data.groupby('ROUTE')['PASSENGERS'].sum().reset_index()

    # Step 3: Return the top N routes sorted by total passenger volume
    return total_passengers.sort_values(by='PASSENGERS', ascending=False).head(top_n)

#EDA:
def get_trend_plot(df):
    fig = go.Figure()

    df = df.copy()
    df = df.sort_values('DATE')
    df = df.set_index('DATE')

    if df['PASSENGERS'].isnull().any():
        df = df.dropna(subset=['PASSENGERS'])

    ts = df['PASSENGERS']

    # Check for minimum data length
    if len(ts) < 24:
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

    # Compute decomposition
    if (ts <= 0).any():
        model_type = "additive"
    else:
        model_type = "multiplicative"
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
        xaxis=dict(
            showgrid=False,          # no vertical lines
            zeroline=False,
            showline=True),
        yaxis_title="Passengers",
        yaxis=dict(
            showgrid=True,           # only horizontal lines
            gridcolor='rgba(200, 200, 200, 0.3)',  # semi-transparent
            gridwidth=1,
            griddash='dot',          # dashed lines
            zeroline=False,
            showline=False
            
        ),
        
        plot_bgcolor="#222222",
        paper_bgcolor="#111111",
        font_color="white"
    )

    return fig



def get_seasonality_plot(df):
    monthly = df.groupby(["YEAR", "MONTH"])["PASSENGERS"].sum().reset_index()
    fig = px.box(monthly, x="MONTH", y="PASSENGERS", title="Seasonal Pattern of Passengers by Month")
    return fig

def get_outliers_plot(df):
    q1 = df["PASSENGERS"].quantile(0.25)
    q3 = df["PASSENGERS"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df["OUTLIER"] = (df["PASSENGERS"] < lower) | (df["PASSENGERS"] > upper).copy()
    if df["OUTLIER"].sum() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No anomalies identified based on IQR method.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["DATE"], y=df["PASSENGERS"], mode='lines+markers', name='Passengers'))
    fig.add_trace(go.Scatter(x=df[df["OUTLIER"]]["DATE"], y=df[df["OUTLIER"]]["PASSENGERS"],
                             mode='markers', name='Outliers', marker=dict(color='red', size=10)))
    fig.update_layout(title="Outliers in Passengers", xaxis_title="Date", yaxis_title="Passengers")
    return fig
    
    
def generate_route_insights(df):
    

    insights = []

    df["DATE"] = pd.to_datetime(df["DATE"])
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]

    all_routes = df["ROUTE"].unique()

    for route in all_routes:
        route_df = df[df["ROUTE"] == route].sort_values("DATE")

        # Skip routes with missing values or too little data
        if route_df["PASSENGERS"].isnull().any() or len(route_df) < 36:
            continue
        # STL decomposition
        
        ts = route_df.set_index("DATE")["PASSENGERS"]
        stl = STL(ts, period=12)
        res = stl.fit()

        # Use only the trend component for slope estimation
        trend = res.trend.dropna()
        if len(trend) >= 12:
            x = np.arange(len(trend))
            y = trend.values
            slope, *_ = np.polyfit(x, y, 1)
        else:
            slope = 0


        # STL decomposition for seasonality and outliers
        ts = route_df.set_index("DATE")["PASSENGERS"]
        stl = STL(ts, period=12)
        res = stl.fit()

        avg_passengers = ts.mean()
        season_amp = res.seasonal.max() - res.seasonal.min()
        season_amp_pct = (season_amp / avg_passengers) * 100

        
        # IQR method to detect outliers in residuals
        resid = res.resid
        q1, q3 = np.percentile(resid, [25, 75])
        iqr = q3 - q1
        outliers = ((resid < (q1 - 1.5 * iqr)) | (resid > (q3 + 1.5 * iqr))).sum()
        
        # Forecast Error: Holt-Winters (2024) 
        try:
            train_hw = route_df[route_df["DATE"].dt.year < 2024]
            valid_hw = route_df[route_df["DATE"].dt.year == 2024]
            ts_hw = train_hw.set_index("DATE")["PASSENGERS"]
            ts_hw.index.freq = 'MS'

            model_hw = ExponentialSmoothing(ts_hw, trend='add', seasonal='add', seasonal_periods=12)
            fit_hw = model_hw.fit()
            forecast_hw = fit_hw.forecast(12)

            mae_hw = (mean_absolute_error(valid_hw["PASSENGERS"], forecast_hw))/ np.mean(valid_hw["PASSENGERS"])
        except:
            mae_hw = np.nan
        #Forecast Error: Sarima/Autoarima
        for route in all_routes:
            route_df = df[df["ROUTE"] == route]
            airlines = route_df["UNIQUE_CARRIER_NAME"].unique()

            for airline in airlines:
                sub_df = route_df[route_df["UNIQUE_CARRIER_NAME"] == airline].sort_values("DATE")

                if sub_df["PASSENGERS"].isnull().any() or len(sub_df) < 24:
                    continue

                route_key = f"{route} | {airline}"
        
        file_path = f"saved_forecasts/{route_key}_2024.pkl"

        try:
            with open(file_path, "rb") as f:
                saved = pickle.load(f)
            
            train_data = saved["train_data"]
            valid_data = saved["valid_data"]
            forecast_valid = saved["forecast_valid"]

            mae_sarima = mean_absolute_error(valid_data["PASSENGERS"], forecast_valid["VALUE"]) / valid_data["PASSENGERS"].mean()

        except FileNotFoundError:
            print(f"No saved forecast for {route_key} in 2024.")
            mae_sarima = None

        # Collect results
        insights.append({
            "route": route,
            "trend_slope": round(slope, 2),
            "season_amp_pct": round(season_amp_pct, 1),
            "outlier_count": int(outliers),
            "mae_holt": round(mae_hw, 1) if not np.isnan(mae_hw) else np.nan,
            "mae_sarima": round(mae_sarima, 1) if not np.isnan(mae_sarima) else np.nan,
            #"quotient_holt": round(mae_hw / slope, 3) if (not np.isnan(mae_hw) and slope != 0) else np.nan,
            #"quotient_sarima": round(mae_sarima / slope, 3) if (not np.isnan(mae_sarima) and slope != 0) else np.nan
        })

    df_result = pd.DataFrame(insights)
    df_result = df_result.sort_values("trend_slope", ascending=False).reset_index(drop=True)
    
    # Save the insights to CSV for dashboard use
    df_result.to_csv("Data/precomputed_route_insights.csv", index=False)
    
    return df_result


def generate_combined_route_score(top_n=10):
    
    # Load precomputed insights
    insights_df = pd.read_csv("Data/precomputed_route_insights.csv")

    # Load raw route data to compute volume and load factor
    route_df = pd.read_csv("Data/Grouped_All_Valid_Connections.csv", dtype={14: str})
    route_df = route_df[(route_df["SEATS"] > 0) & (route_df["PASSENGERS"] > 0)].copy()
    route_df["ROUTE"] = route_df["ORIGIN"] + " → " + route_df["DEST"]
    route_df["LOAD_FACTOR"] = route_df["PASSENGERS"] / route_df["SEATS"]

    # Aggregate total passengers and load factor per route
    agg_df = route_df.groupby("ROUTE", as_index=False).agg({
        "PASSENGERS": "sum",
        "SEATS": "sum"
    })
    agg_df["LOAD_FACTOR"] = agg_df["PASSENGERS"] / agg_df["SEATS"]

    # Merge with insights
    merged_df = pd.merge(insights_df, agg_df, left_on="route", right_on="ROUTE", how="inner")

    # Normalize selected columns
    features = ["trend_slope", "season_amp_pct", "mae_holt", "mae_sarima", "PASSENGERS", "LOAD_FACTOR"]
    scaler = MinMaxScaler()
    merged_df[[f + "_scaled" for f in features]] = scaler.fit_transform(merged_df[features])

    # Apply weighted scoring
    weights = {
        "trend_slope_scaled": 0.3,
        "season_amp_pct_scaled": -0.1,
        "mae_holt_scaled": -0.1,
        "mae_sarima_scaled": -0.15,
        "PASSENGERS_scaled": 0.45,
        "LOAD_FACTOR_scaled": 0.5
    }

    merged_df["score"] = sum(merged_df[col] * weight for col, weight in weights.items())

    # Return top N routes sorted by score
    return merged_df.sort_values("score", ascending=False).head(top_n)


if __name__ == "__main__":
    
    df = pd.read_csv("Data/Grouped_All_Valid_Connections.csv",low_memory=False)#, dtype={14: str})
    df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01")
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]
    
    '''
    # One-time forecast save run for 2024
    from forecasting import sarima_forecast

    all_routes = df["ROUTE"].unique()

    for route in all_routes:
        route_df = df[df["ROUTE"] == route]
        airlines = route_df["UNIQUE_CARRIER_NAME"].unique()

        for airline in airlines:
            sub_df = route_df[route_df["UNIQUE_CARRIER_NAME"] == airline].copy()
            #Skip if there is insufficient data oer missing values
            if len(sub_df) < 24 or sub_df["PASSENGERS"].isnull().any():
                print(f"Not enough data for {route} | {airline}")
                continue

            print(f"Saving forecast: {route} | {airline}")
            try:
                sarima_forecast(
                    df=sub_df,
                    forecast_year=2024,
                    route=route,
                    airline=airline,
                    save=True  
                )
            except Exception as e:
                print(f"Error with {route} | {airline}: {e}")
    '''

    generate_route_insights(df)
    
