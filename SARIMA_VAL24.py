import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("Data/Grouped_All_Valid_Connections.csv")
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))  # Create a datetime column from YEAR and MONTH
df['ROUTE'] = df['ORIGIN'] + " → " + df['DEST']  # Create readable route names
routes = df['ROUTE'].unique()

# Initialize Dash app and layout
app = Dash(__name__)
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': 'white', 'padding': '20px'}, children=[
    html.H1("📈 SARIMA Forecast: Validation 2024 & Prediction 2025", style={'textAlign': 'center'}),
    html.Label("Select a route:", style={'fontSize': '18px'}),
    dcc.Dropdown(
        id='route-dropdown',
        options=[{'label': route, 'value': route} for route in sorted(routes)],
        value=routes[0],
        style={'color': 'black'}
    ),
    dcc.Graph(id='forecast-graph'),
    html.Div(id='error-metrics', style={'marginTop': '20px', 'fontSize': '16px'})
])

@app.callback(
    Output('forecast-graph', 'figure'),
    Output('error-metrics', 'children'),
    Input('route-dropdown', 'value')
)
def update_graph(selected_route):
    # Filter the data to only include the selected route
    route_df = df[df['ROUTE'] == selected_route]
    if route_df.empty:
        return go.Figure().update_layout(title="No data available"), "No data available"

    # Group data monthly by date
    grouped = route_df.groupby('DATE')['PASSENGERS'].sum().reset_index().sort_values('DATE')

    # === Step 1: Initial training on 2022–2023 ===
    train_initial = grouped[grouped['DATE'] < '2024-01-01']
    valid_2024 = grouped[(grouped['DATE'] >= '2024-01-01') & (grouped['DATE'] < '2025-01-01')]

    try:
        # Train SARIMA model on 2022–2023 data
        model_initial = SARIMAX(train_initial['PASSENGERS'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit_initial = model_initial.fit(disp=False)

        # Forecast the months of 2024 for validation purposes
        forecast_valid = model_fit_initial.get_forecast(steps=len(valid_2024))

        # Build dataframe for 2024 forecasted values
        forecast_df_2024 = pd.DataFrame({
            'DATE': valid_2024['DATE'].values,
            'VALUE': forecast_valid.predicted_mean,
            'TYPE': 'Forecast 2024'
        })

        # Calculate validation error on 2024 (compared to actuals)
        mae = mean_absolute_error(valid_2024['PASSENGERS'], forecast_valid.predicted_mean)
        rmse = np.sqrt(mean_squared_error(valid_2024['PASSENGERS'], forecast_valid.predicted_mean))
        error_text = f"📏 MAE (2024): {mae:.0f} passengers | RMSE: {rmse:.0f}"

        # === Step 2: Retrain on 2022–2024 for 2025 prediction ===
        full_train = grouped[grouped['DATE'] < '2025-01-01']

        # Train new model including 2024
        model_final = SARIMAX(full_train['PASSENGERS'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit_final = model_final.fit(disp=False)

        # Forecast next 12 months (2025)
        forecast_2025 = model_fit_final.get_forecast(steps=12)
        forecast_df_2025 = pd.DataFrame({
            'DATE': pd.date_range(start='2025-01-01', periods=12, freq='MS'),
            'VALUE': forecast_2025.predicted_mean,
            'TYPE': 'Forecast 2025'
        })

    except:
        # In case of failure during modeling
        forecast_df_2024 = pd.DataFrame(columns=['DATE', 'VALUE', 'TYPE'])
        forecast_df_2025 = pd.DataFrame(columns=['DATE', 'VALUE', 'TYPE'])
        error_text = "❌ Error during model fitting or forecasting"

    # Prepare actual training and validation data for plotting
    real_train = train_initial.rename(columns={'PASSENGERS': 'VALUE'}).assign(TYPE='Training data')
    real_valid = valid_2024.rename(columns={'PASSENGERS': 'VALUE'}).assign(TYPE='Actual 2024')

    # Combine all data into one plot-friendly dataframe
    plot_df = pd.concat([real_train, real_valid, forecast_df_2024, forecast_df_2025], ignore_index=True)

    # Create line plot
    fig = go.Figure()
    for typ in plot_df['TYPE'].unique():
        sub = plot_df[plot_df['TYPE'] == typ]
        fig.add_trace(go.Scatter(
            x=sub['DATE'], y=sub['VALUE'],
            mode='lines+markers',
            name=typ,
            line=dict(width=2)
        ))

    # Style the plot
    fig.update_layout(
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white'),
        hovermode='x unified',
        xaxis_title="Month",
        yaxis_title="Number of passengers",
        legend=dict(x=0, y=1.1, orientation='h')
    )

    return fig, error_text

# Run the Dash app
if __name__ == "__main__":
    app.run(debug=True)
