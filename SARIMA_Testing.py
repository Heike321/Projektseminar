import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# CSV laden und vorbereiten
df = pd.read_csv("Data/Grouped_All_Valid_Connections.csv")
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
df['ROUTE'] = df['ORIGIN'] + " → " + df['DEST']
routes = df['ROUTE'].unique()

# Dash-App
app = Dash(__name__)
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': 'white', 'padding': '20px'}, children=[
    html.H1("📈 SARIMA Prognose pro Flugverbindung", style={'textAlign': 'center'}),
    html.Label("Wähle eine Route:", style={'fontSize': '18px'}),
    dcc.Dropdown(
        id='route-dropdown',
        options=[{'label': route, 'value': route} for route in sorted(routes)],
        value=routes[0],
        style={'color': 'black'}
    ),
    dcc.Graph(id='forecast-graph')
])

@app.callback(
    Output('forecast-graph', 'figure'),
    Input('route-dropdown', 'value')
)
def update_graph(selected_route):
    route_df = df[df['ROUTE'] == selected_route]
    if route_df.empty:
        return go.Figure().update_layout(title="Keine Daten verfügbar")

    grouped = route_df.groupby('DATE')['PASSENGERS'].sum().reset_index()

    try:
        model = SARIMAX(grouped['PASSENGERS'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)

        forecast = model_fit.get_forecast(steps=12)
        forecast_index = pd.date_range(start=grouped['DATE'].max() + pd.offsets.MonthBegin(), periods=12, freq='MS')
        forecast_df = pd.DataFrame({'DATE': forecast_index, 'VALUE': forecast.predicted_mean, 'TYPE': 'Prognose'})
    except:
        forecast_df = pd.DataFrame(columns=['DATE', 'VALUE', 'TYPE'])

    real_df = grouped.rename(columns={'PASSENGERS': 'VALUE'}).assign(TYPE='Echt')
    plot_df = pd.concat([real_df, forecast_df], ignore_index=True)

    fig = go.Figure()
    for typ in plot_df['TYPE'].unique():
        sub = plot_df[plot_df['TYPE'] == typ]
        fig.add_trace(go.Scatter(
            x=sub['DATE'], y=sub['VALUE'],
            mode='lines+markers',
            name=typ,
            line=dict(width=2)
        ))

    fig.update_layout(
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white'),
        hovermode='x unified',
        xaxis_title="Monat",
        yaxis_title="Passagierzahl",
        legend=dict(x=0, y=1.1, orientation='h')
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True)
