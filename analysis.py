import pandas as pd
import plotly.express as px
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from scipy import stats

df = pd.read_csv("Data/Grouped_All_Valid_Connections.csv", dtype={14: str})
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

    # ❗ Check for minimum data length
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
    decomposition = seasonal_decompose(ts, model="additive", period=12)

    fig.add_trace(go.Scatter(
        x=decomposition.trend.index,
        y=decomposition.trend.values,
        mode="lines",
        name="Trend"
    ))

    fig.update_layout(
        title="Passenger Trend",
        xaxis_title="Date",
        yaxis_title="Passengers",
        plot_bgcolor="#222222",
        paper_bgcolor="#111111",
        font_color="white"
    )

    return fig



def get_seasonality_plot(df):
    monthly = df.groupby(["YEAR", "MONTH"])["PASSENGERS"].sum().reset_index()
    fig = px.box(monthly, x="MONTH", y="PASSENGERS", title="Seasonal Pattern of Passengers by Month")
    return fig
'''
def get_outliers_plot(df):
    monthly = df.groupby("DATE")["PASSENGERS"].sum().reset_index()
    monthly["z_score"] = stats.zscore(monthly["PASSENGERS"].fillna(0))
    outliers = monthly[np.abs(monthly["z_score"]) > 3]
    fig = px.line(monthly, x="DATE", y="PASSENGERS", title="Outlier Detection in Monthly Passenger Data")
    fig.add_scatter(x=outliers["DATE"], y=outliers["PASSENGERS"], mode="markers", marker=dict(color="red", size=8), name="Outliers")
    return fig
'''
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
   