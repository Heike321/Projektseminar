import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px

# Daten laden
data = pd.read_csv("Data/Grouped_All_Valid_Connections.csv")
data["DATE"] = pd.to_datetime(data["YEAR"].astype(str) + "-" + data["MONTH"].astype(str) + "-01")

with open("Data/valid_connections.json") as f:
    route_options = json.load(f)

# App initialisieren
app = dash.Dash(__name__)
app.title = "Flight Dashboard"

# Layout der App
app.layout = html.Div(
    style={'backgroundColor': '#111111', 'color': 'white', 'padding': '20px'},
    children=[
        html.H1("Flight Connection Dashboard ✈️", style={'textAlign': 'center'}),

        html.Div(style={'display': 'flex'}, children=[

            html.Div(style={'flex': 2, 'marginRight': '40px'}, children=[
                html.Div([
                    html.Label("Select a route:"),
                    dcc.Dropdown(
                        id='route-selector',
                        options=route_options,
                        placeholder="Choose a route",
                        style={'width': '100%', 'backgroundColor': 'white', 'color': 'black'}
                    )
                ], style={'marginBottom': '20px'}),

                html.Div([
                    html.Div([
                        html.Label("Select airline:"),
                        dcc.Dropdown(
                            id='airline-selector',
                            options=[{"label": "All Airlines", "value": "all"}],
                            value="all",
                            style={'width': '100%', 'backgroundColor': 'white', 'color': 'black'}
                        )
                    ], style={'flex': 1, 'marginRight': '10px'}),

                    html.Div([
                        html.Label("Select year:"),
                        dcc.Dropdown(
                            id='year-selector',
                            options=[
                                {"label": "All years", "value": "all"},
                                {"label": "2022", "value": 2022},
                                {"label": "2023", "value": 2023},
                                {"label": "2024", "value": 2024}
                            ],
                            value="all",
                            clearable=False,
                            style={'width': '100%', 'backgroundColor': 'white', 'color': 'black'}
                        )
                    ], style={'flex': 1})
                ], style={'display': 'flex', 'marginBottom': '20px'}),

                dcc.Graph(id='lf-graph')
            ]),

            html.Div(style={'flex': 1}, children=[
                html.H2("Top 10 Routes", style={'textAlign': 'center'}),

                html.Label("Select year:"),
                dcc.Dropdown(
                    id='top-routes-year-selector',
                    options=[
                        {"label": "All years", "value": "all"},
                        {"label": "2022", "value": 2022},
                        {"label": "2023", "value": 2023},
                        {"label": "2024", "value": 2024}
                    ],
                    value="all",
                    style={'width': '100%', 'backgroundColor': 'white', 'color': 'black'},
                    clearable=False
                ),

                html.Label("Select month:"),
                dcc.Dropdown(
                    id='top-routes-month-selector',
                    options=[{"label": "All month", "value": "all"}] +
                            [{"label": str(m), "value": m} for m in range(1, 13)],
                    value=1,
                    style={'width': '100%', 'backgroundColor': 'white', 'color': 'black'},
                    clearable=False
                ),

                html.Br(),
                dcc.Graph(id='top-routes-bar'),
                html.Div(id='top-routes-table')
            ])
        ])
    ]
)

# Callback 1: Airline-Dropdown aktualisieren
@app.callback(
    Output('airline-selector', 'options'),
    Output('airline-selector', 'value'),
    Input('route-selector', 'value')
)
def update_airline_options(selected_route):
    if not selected_route:
        return [], "all"

    try:
        parts = selected_route.split("-")
        origin_code = parts[2]
        dest_code = parts[3]
    except Exception as e:
        print("Fehler beim Zerlegen:", e)
        return [], "all"

    filtered = data[(data["ORIGIN"] == origin_code) & (data["DEST"] == dest_code)]
    airlines = sorted(filtered["UNIQUE_CARRIER_NAME"].dropna().unique())

    options = [{"label": airline, "value": airline} for airline in airlines]
    options.append({"label": "All Airlines", "value": "all"})

    return options, "all"

# Callback 2: Graph aktualisieren
@app.callback(
    Output('lf-graph', 'figure'),
    Input('route-selector', 'value'),
    Input('year-selector', 'value'),
    Input('airline-selector', 'value')
)
def update_graph(route_value, selected_year, selected_airline):
    if not route_value:
        return go.Figure()

    try:
        parts = route_value.split("-")
        origin_code = parts[2]
        dest_code = parts[3]
    except Exception as e:
        print("Fehler beim Zerlegen der Route:", e)
        return go.Figure()

    filtered = data[(data["ORIGIN"] == origin_code) & (data["DEST"] == dest_code)]

    if selected_year != "all":
        filtered = filtered[filtered["YEAR"] == int(selected_year)]

    if selected_airline != "all":
        filtered = filtered[filtered["UNIQUE_CARRIER_NAME"] == selected_airline]
    else:
        filtered = filtered.groupby(["YEAR", "MONTH"], as_index=False).agg({
            "PASSENGERS": "sum",
            "SEATS": "sum",
            "DEPARTURES_PERFORMED": "sum"
        })
        filtered["DATE"] = pd.to_datetime(filtered["YEAR"].astype(str) + "-" + filtered["MONTH"].astype(str) + "-01")
        filtered["LOAD_FACTOR"] = filtered.apply(
            lambda row: row["PASSENGERS"] / row["SEATS"] if row["SEATS"] > 0 else 0,
            axis=1
        )

    if filtered.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["DATE"],
        y=filtered["LOAD_FACTOR"],
        mode='lines+markers',
        name='Load Factor'
    ))

    fig.update_layout(
        title=f"Load Factor for {origin_code} → {dest_code}",
        xaxis_title="Date",
        yaxis_title="Load Factor",
        plot_bgcolor='#222222',
        paper_bgcolor='#111111',
        font_color='white'
    )

    return fig

# Callback 3: Top-Routen-Tabelle & Balkendiagramm
@app.callback(
    [Output('top-routes-bar', 'figure'),
     Output('top-routes-table', 'children')],
    [Input('top-routes-year-selector', 'value'),
     Input('top-routes-month-selector', 'value')]
)
def update_top_routes_visuals(selected_year, selected_month):
    df_filtered = data.copy()

    if selected_year != "all":
        df_filtered = df_filtered[df_filtered["YEAR"] == int(selected_year)]

    if selected_month != "all":
        df_filtered = df_filtered[df_filtered["MONTH"] == int(selected_month)]

    df_filtered = df_filtered[df_filtered["SEATS"] > 0]
    df_filtered["ROUTE"] = df_filtered["ORIGIN"] + " → " + df_filtered["DEST"]
    df_filtered["LOAD_FACTOR"] = df_filtered["PASSENGERS"] / df_filtered["SEATS"]

    top_routes = df_filtered.groupby("ROUTE", as_index=False).agg({
        "PASSENGERS": "sum",
        "SEATS": "sum"
    })
    top_routes["LOAD_FACTOR"] = top_routes["PASSENGERS"] / top_routes["SEATS"]
    top_routes = top_routes.sort_values("PASSENGERS", ascending=False).head(10)

    fig = px.bar(
        top_routes,
        x="ROUTE",
        y="PASSENGERS",
        title="Top 10 Routes by Passengers",
        labels={"PASSENGERS": "Number of Passengers"},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='#222222',
        paper_bgcolor='#111111',
        font_color='white'
    )

    table = dash_table.DataTable(
        columns=[
            {"name": "Route", "id": "ROUTE"},
            {"name": "Passengers", "id": "PASSENGERS", "type": "numeric", "format": {"specifier": ","}},
            {"name": "Seats", "id": "SEATS", "type": "numeric", "format": {"specifier": ","}},
            {"name": "Load Factor", "id": "LOAD_FACTOR", "type": "numeric", "format": {"specifier": ".2%"}},
        ],
        data=top_routes.to_dict("records"),
        style_table={'overflowX': 'auto'},
        style_cell={'backgroundColor': '#111111', 'color': 'white', 'padding': '8px'},
        style_header={'backgroundColor': '#222222', 'fontWeight': 'bold'}
    )

    return fig, table

# App starten
if __name__ == '__main__':
    app.run(debug=True)