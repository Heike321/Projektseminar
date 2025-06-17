import pandas as pd
import plotly.express as px
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
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

        y = route_df["PASSENGERS"].values
        x = np.arange(len(y))

        # Linear trend estimation
        slope, *_ = np.polyfit(x, y, 1)

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

        '''
        # Forecast Error: SARIMA (2024)
        
        try:
            train_sarima = route_df[route_df["DATE"] < "2024-01-01"]
            valid_sarima = route_df[(route_df["DATE"] >= "2024-01-01") & (route_df["DATE"] < "2025-01-01")]
            ts_sarima = train_sarima.set_index("DATE")["PASSENGERS"]
            ts_sarima.index.freq = 'MS'

            model_sarima = SARIMAX(ts_sarima, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fit_sarima = model_sarima.fit(disp=False)
            forecast_sarima = fit_sarima.get_forecast(steps=12).predicted_mean

            
        except:
            mae_sarima = np.nan
        '''
         # Forecast Error: AutoARIMA (2024)
        try:
            train_auto = route_df[route_df["DATE"] < "2024-01-01"]
            valid_auto = route_df[(route_df["DATE"] >= "2024-01-01") & (route_df["DATE"] < "2025-01-01")]

            if len(train_auto) < 24 or len(valid_auto) == 0:
                raise ValueError("Not enough data for AutoARIMA forecast")

            df_train = train_auto[["DATE", "PASSENGERS"]].copy()
            df_train.columns = ["ds", "y"]
            df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")
            df_train = df_train.dropna(subset=["y"])
            df_train["unique_id"] = "series"
            df_train = df_train[["unique_id", "ds", "y"]]

            # Korrekte Initialisierung
            sf_model = StatsForecast(models=[AutoARIMA(season_length=12)], freq='MS')
            forecast_df = sf_model.forecast(df=df_train, h=12)



            forecast_values = forecast_df[forecast_df["unique_id"] == "series"]["AutoARIMA"].values
            forecast_values = forecast_values[:len(valid_auto)]
            true_values = valid_auto["PASSENGERS"].values[:len(forecast_values)]

            mae_sarima = mean_absolute_error(true_values, forecast_values) / np.mean(true_values)

        except Exception as e:
            print(f"⚠️ AutoARIMA failed: {e}")
            mae_sarima = np.nan
        
        '''
        #Forecast-Error 
        #Ich versuche hier die Fehlerberechnung zu vereinfachen, sodass hier nur die Werte von forecasting.py verwendet werden und nicht neue Vorhersagen gemacht werden...
        # Filter train and validation data (same as in SARIMA)
        train_data = df[df['DATE'].dt.year < valid_year]
        valid_data = df[df['DATE'].dt.year == valid_year]

        # Holt-Winters Forecast auf Basis train_data
        forecast_df = forecast_passengers(train_data, periods=len(valid_data))

        # MAE mit valid_data berechnen
        mae_holt = mean_absolute_error(valid_data['PASSENGERS'], forecast_df['FORECAST_PASSENGERS']) / valid_data['PASSENGERS'].mean()

        # Forecast und Validierungsdaten aus forecasting.py holen
        train_data, valid_data, forecast_valid, _ = sarima_forecast(route_df, valid_year=valid_year, forecast_year=valid_year+1)

        # MAE berechnen 
        mae_sarima = mean_absolute_error(valid_data["PASSENGERS"], forecast_valid["VALUE"]) / valid_data["PASSENGERS"].mean()
        '''




        
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

if __name__ == "__main__":
    
    df = pd.read_csv("Data/Grouped_All_Valid_Connections.csv",low_memory=False)#, dtype={14: str})
    df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01")
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]
    
    generate_route_insights(df)

