import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import json
import plotly.graph_objects as go
from analysis import compute_top_routes  

# Load and preprocess data
data = pd.read_csv("Data/Grouped_All_Valid_Connections.csv")
data["DATE"] = pd.to_datetime(data["YEAR"].astype(str) + "-" + data["MONTH"].astype(str) + "-01")

with open("Data/valid_routes.json") as f:
    route_options = json.load(f)

# Initialize Dash app 
app = dash.Dash(__name__)
app.title = "Flight Dashboard"

# App layout 
app.layout = html.Div(
    style={'backgroundColor': '#111111', 'color': 'white', 'padding': '20px'},
    children=[
        html.H1("Flight Connection Dashboard", style={'textAlign': 'center'}),

        html.Div(style={'display': 'flex'}, children=[
            # LEFT SIDE: Graph and controls
            html.Div(style={'flex': 2, 'marginRight': '40px'}, children=[

                # Route dropdown (alone at top)
                html.Div([
                    html.Label("Select a route:"),
                    dcc.Dropdown(
                        id='route-selector',
                        options=route_options,
                        placeholder="Choose a route",
                        style={'width': '100%', 'backgroundColor': 'white', 'color': 'black'}
                    )
                ], style={'marginBottom': '20px'}),  # <- Achtung: "marginBottom", nicht "marginButtom"

                # Airline and Year dropdowns side-by-side
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

                # Graph
                dcc.Graph(id='lf-graph')
            ]),

            # RIGHT SIDE: Top Routes Table
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
                    clearable=False,
                    #style={'backgroundColor': 'white', 'color': 'black'}
                ),

                html.Br(),

                html.Div(id='top-routes-table')
            ])
        ])
    ])
'''
        # Graph and top routes side-by-side
        html.Div([
            #html.Div([
                # Dropdown section (route, airline, year)
                #html.Div([
            html.Div([
                html.Label("Select a route:"),
                dcc.Dropdown(
                    id='route-selector',
                    options=route_options,
                    placeholder="Choose a route",
                    style={'width': '100%', 'backgroundColor': 'white', 'color': 'black'}
                )
            ], style={'marginButtom': '20px'}),

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

            dcc.Graph(id='lf-graph')  # Load factor time series
        ], style={'flex': 2, 'marginRight': '40px'}),

            html.Div([
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
                    clearable=False,
                    style={'backgroundColor': 'white', 'color': 'black'}
                ),

                html.Br(),
                html.Div(id='top-routes-table')  # DataTable will be rendered here
            ], style={'flex': 1})
        ], style={'display': 'flex'})
        
    ]
)
'''
# Callback: Update airline dropdown based on selected route
@app.callback(
    Output('airline-selector', 'options'),
    Output('airline-selector', 'value'),
    Input('route-selector', 'value')
)
def update_airline_options(selected_route):
    if not selected_route:
        return [], "all"

    origin, dest = selected_route.split('-')
    filtered = data[(data["ORIGIN"] == origin) & (data["DEST"] == dest)]
    airlines = sorted(filtered["UNIQUE_CARRIER_NAME"].dropna().unique())

    options = [{"label": airline, "value": airline} for airline in airlines]
    options.append({"label": "All Airlines", "value": "all"})

    return options, "all"

# Callback: Update load factor graph
@app.callback(
    Output('lf-graph', 'figure'),
    Input('route-selector', 'value'),
    Input('year-selector', 'value'),
    Input('airline-selector', 'value')
)
def update_graph(route_value, selected_year, selected_airline):
    if not route_value:
        return go.Figure()

    origin, dest = route_value.split('-')
    filtered = data[(data["ORIGIN"] == origin) & (data["DEST"] == dest)]

    if selected_year != "all":
        filtered = filtered[filtered["YEAR"] == int(selected_year)]

    if selected_airline != "all":
        filtered = filtered[filtered["UNIQUE_CARRIER_NAME"] == selected_airline]
    else:
        # Aggregate over all airlines
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["DATE"],
        y=filtered["LOAD_FACTOR"],
        mode='lines+markers',
        name='Load Factor'
    ))

    fig.update_layout(
        title=f"Load Factor for {origin} → {dest}",
        xaxis_title="Date",
        yaxis_title="Load Factor",
        plot_bgcolor='#222222',
        paper_bgcolor='#111111',
        font_color='white'
    )

    return fig

# Callback: Update top 10 routes table
@app.callback(
    Output('top-routes-table', 'children'),
    Input('top-routes-year-selector', 'value')
)
def update_top_routes_table(selected_year):
    df_filtered = data if selected_year == "all" else data[data["YEAR"] == int(selected_year)]
    top_routes = compute_top_routes(df_filtered)

    return dash_table.DataTable(
        columns=[
            {"name": "Route", "id": "ROUTE"},
            {"name": "Passengers", "id": "PASSENGERS", "type": "numeric", "format": {"specifier": ","}}
        ],
        data=top_routes.to_dict("records"),
        style_table={'overflowX': 'auto'},
        style_cell={'backgroundColor': '#111111', 'color': 'white', 'padding': '8px'},
        style_header={'backgroundColor': '#222222', 'fontWeight': 'bold'}
    )

# Run app
if __name__ == '__main__':
    app.run(debug=True)
