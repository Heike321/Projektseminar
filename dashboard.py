import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from analysis import compute_top_routes  
from forecasting import forecast_passengers
from preprocess import iata_to_name

# Load and preprocess data
data = pd.read_csv("Data/Grouped_All_Valid_Connections.csv")
data["DATE"] = pd.to_datetime(data["YEAR"].astype(str) + "-" + data["MONTH"].astype(str) + "-01")

with open("Data/valid_routes.json") as f:
    route_options = json.load(f)

iata_codes = data["ORIGIN"].dropna().unique()

# Initialize Dash app 
app = dash.Dash(__name__)
app.title = "Flight Dashboard"

# App layout 
app.layout = html.Div(
    style={'backgroundColor': '#111111', 'color': 'white', 'padding': '20px'},
    children=[
        html.H1("Flight Connection Dashboard ✈️", style={'textAlign': 'center'}),

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

                # Airline and Year dropdowns
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
                               # {"label": "All years with forecast for 2025", "value": "all_2025"},
                                {"label": "2022", "value": 2022},
                                {"label": "2023", "value": 2023},
                                {"label": "2024", "value": 2024},
                                {"label": "Forecast 2024", "value": "forecast_2024"},
                                {"label": "Forecast 2025", "value": "forecast_2025"},

                            ],
                            value="all",
                            clearable=False,
                            style={'width': '100%', 'backgroundColor': 'white', 'color': 'black'}
                        )
                    ], style={'flex': 1})
                ], style={'display': 'flex', 'marginBottom': '20px'}),

                # Graph
                dcc.Graph(id='lf-graph'),
                dcc.Graph(id='passenger-graph')
            ]),

            # RIGHT SIDE: Top Routes Table
            
   



            html.Div(style={'flex': 1}, children=[
                html.H2("Route Map", style={'textAlign': 'center'}),
                html.Label("Select "),
                dcc.Dropdown(
                    id="origin-dropdown",
                    options = [{"label": f"{iata_to_name.get(iata, iata)} ({iata})", "value": iata} for iata in sorted(iata_codes)],
                   # options=[{"label": f"{name} ({iata})", "value": iata} for iata, name in iata_to_name.items()],
                    placeholder="Select origin airport",
                    clearable=True,
                    style={'width': '100%', 'backgroundColor': 'white', 'color': 'black'}
                    
                ),
                dcc.Graph(id='route-map'),
                html.H3("Top 10 Routes", style={'textAlign': 'center'}),

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
                    clearable=False,
                    #style={'backgroundColor': 'white', 'color': 'black'}
                ),
                html.Label("Select month:"),
                dcc.Dropdown(
                    id='top-routes-month-selector',
                    options=[{"label": "All month", "value": "all"}]+
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
    ])

@app.callback(
    Output("route-map", "figure"),
    Input("origin-dropdown", "value")
)
def update_map(selected_origin):
    
    fig = go.Figure()

    # Always show the world map
    fig.add_trace(go.Scattergeo(
        lon=[0],
        lat=[0],
        mode='markers',
        marker=dict(size=0, color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # If a departure airport is selected:
    if selected_origin:
        filtered = data[data["ORIGIN"] == selected_origin]

        # Mark starting point (visible)
        start_row = filtered.iloc[0] if not filtered.empty else None
        if start_row is not None:
            fig.add_trace(go.Scattergeo(
                lon=[start_row["ORIGIN_LON"]],
                lat=[start_row["ORIGIN_LAT"]],
                mode='markers',
                showlegend=False,
                marker=dict(size=10, color='limegreen'),
                name="Start"
            ))

        for _, row in filtered.iterrows():
            # Line
            fig.add_trace(go.Scattergeo(
                lon=[row["ORIGIN_LON"], row["DEST_LON"]],
                lat=[row["ORIGIN_LAT"], row["DEST_LAT"]],
                mode='lines',
                line=dict(width=1, dash='dot', color='cyan'),
                opacity=0.6,
                hoverinfo='text',
                showlegend=False,
                text=f"{iata_to_name.get(row['ORIGIN'], row['ORIGIN'])} → {iata_to_name.get(row['DEST'], row['DEST'])}"
            ))
            '''
            # Zielmarker mit Legende für jedes Ziel (Funktioniert noch nicht... die Legende wiederholt sich immer wieder)
            filtered = filtered.copy()
            filtered["DEST_LAT"] = filtered["DEST_LAT"].round(3)
            filtered["DEST_LON"] = filtered["DEST_LON"].round(3)
            unique_dests = filtered.drop_duplicates(subset=["DEST", "DEST_LAT", "DEST_LON"])

            for _, dest_row in unique_dests.iterrows():
                fig.add_trace(go.Scattergeo(
                    lon=[dest_row["DEST_LON"]],
                    lat=[dest_row["DEST_LAT"]],
                    mode='markers',
                    marker=dict(size=8, symbol='star', color='red'),
                    name=f"Ziel: {iata_to_name.get(dest_row['DEST'], dest_row['DEST'])}",
                    showlegend=True,
                    hoverinfo='skip'
                ))
            '''
            # Target marker
            fig.add_trace(go.Scattergeo(
                lon=[row["DEST_LON"]],
                lat=[row["DEST_LAT"]],
                mode='markers',
                marker=dict(size=8, symbol='star', color='red'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
    # Geo settings (no border, no labels)
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="gray",
        showcountries=False,
        showocean=False,
        showlakes=False,
        showcoastlines=False,
        lataxis_showgrid=False,
        lonaxis_showgrid=False,
        resolution=50,
        visible=True
    )

    # Layout transparent
    fig.update_layout(
        geo_bgcolor='rgba(0,0,0,0)',     
        paper_bgcolor='rgba(0,0,0,0)',   
        plot_bgcolor='rgba(0,0,0,0)',    
        margin={"r":0, "t":0, "l":0, "b":0}
    )

    return fig

    
    

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

# Callback: Update top 10 routes table
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

    if selected_month!= "all":
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


@app.callback(
    Output('lf-graph', 'figure'),
    Output('passenger-graph', 'figure'),
    Input('route-selector', 'value'),
    Input('year-selector', 'value'),
    Input('airline-selector', 'value')
)
def update_graphs(route_value, selected_year, selected_airline):
    if not route_value:
        return go.Figure(), go.Figure()

    origin, dest = route_value.split('-')
    filtered = data[(data["ORIGIN"] == origin) & (data["DEST"] == dest)]

    if selected_airline != "all":
        filtered = filtered[filtered["UNIQUE_CARRIER_NAME"] == selected_airline]
    else:
        filtered = filtered.groupby(["YEAR", "MONTH"], as_index=False).agg({
            "PASSENGERS": "sum",
            "SEATS": "sum"
        })

    filtered["DATE"] = pd.to_datetime(filtered["YEAR"].astype(str) + "-" + filtered["MONTH"].astype(str) + "-01")
    filtered["LOAD_FACTOR"] = filtered.apply(
        lambda row: row["PASSENGERS"] / row["SEATS"] if row["SEATS"] > 0 else 0, axis=1)

    lf_fig = go.Figure()
    pax_fig = go.Figure()

    if selected_year == "forecast_2024":
        filtered_2024 = filtered[filtered["YEAR"] == 2024]
        filtered_before_2024 = filtered[filtered["YEAR"] < 2024]

        forecast_df = forecast_passengers(filtered_before_2024[["DATE", "PASSENGERS"]], periods=12)
        avg_seat_ratio = filtered_before_2024["SEATS"].sum() / filtered_before_2024["PASSENGERS"].sum()
        forecast_df["SEATS"] = forecast_df["FORECAST_PASSENGERS"] * avg_seat_ratio
        forecast_df["LOAD_FACTOR"] = forecast_df["FORECAST_PASSENGERS"] / forecast_df["SEATS"]

        # Load Factor
        lf_fig.add_trace(go.Scatter(
            x=filtered_2024["DATE"], y=filtered_2024["LOAD_FACTOR"],
            mode='lines+markers', name="Actual 2024"
        ))
        lf_fig.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["LOAD_FACTOR"],
            mode='lines+markers', name="Forecast 2024", line=dict(dash='dot')
        ))

        # Passengers
        pax_fig.add_trace(go.Scatter(
            x=filtered_2024["DATE"], y=filtered_2024["PASSENGERS"],
            mode='lines+markers', name="Actual Passengers 2024"
        ))
        pax_fig.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["FORECAST_PASSENGERS"],
            mode='lines+markers', name="Forecast Passengers", line=dict(dash='dot')
        ))

    elif selected_year == "forecast_2025":
        forecast_df = forecast_passengers(filtered[["DATE", "PASSENGERS"]], periods=12)
        avg_seat_ratio = filtered["SEATS"].sum() / filtered["PASSENGERS"].sum()
        forecast_df["SEATS"] = forecast_df["FORECAST_PASSENGERS"] * avg_seat_ratio
        forecast_df["LOAD_FACTOR"] = forecast_df["FORECAST_PASSENGERS"] / forecast_df["SEATS"]

        lf_fig.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["LOAD_FACTOR"],
            mode='lines+markers', name="Forecast 2025", line=dict(dash='dash')
        ))
        pax_fig.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["FORECAST_PASSENGERS"],
            mode='lines+markers', name="Forecast Passengers", line=dict(dash='dash')
        ))

    else:
        if selected_year != "all":
            filtered = filtered[filtered["YEAR"] == int(selected_year)]

        lf_fig.add_trace(go.Scatter(
            x=filtered["DATE"], y=filtered["LOAD_FACTOR"],
            mode='lines+markers', name="Load Factor"
        ))
        pax_fig.add_trace(go.Scatter(
            x=filtered["DATE"], y=filtered["PASSENGERS"],
            mode='lines+markers', name="Passengers"
        ))

    
    # Layout
    for fig in [lf_fig, pax_fig]:
        fig.update_layout(
            plot_bgcolor='#222222',
            paper_bgcolor='#111111',
            font_color='white'
        )

    lf_fig.update_layout(
        title=f"Load Factor for {origin} → {dest}",
        xaxis_title="Date",
        yaxis_title="Load Factor"
    )

    pax_fig.update_layout(
        title=f"Passenger Volume for {origin} → {dest}",
        xaxis_title="Date",
        yaxis_title="Passengers"
    )

    return lf_fig, pax_fig

'''

@app.callback(
    Output('lf-graph', 'figure'),
    Output('passenger-graph', 'figure'),
    Input('route-selector', 'value'),
    Input('year-selector', 'value'),
    Input('airline-selector', 'value')
)
def update_graphs(route_value, selected_year, selected_airline):
    if not route_value:
        return go.Figure(), go.Figure()

    origin, dest = route_value.split('-')
    df = data[(data["ORIGIN"] == origin) & (data["DEST"] == dest)]

    if selected_airline != "all":
        df = df[df["UNIQUE_CARRIER_NAME"] == selected_airline]

    # Aggregate data
    if selected_airline == "all":
        df = df.groupby(["YEAR", "MONTH"], as_index=False).agg({
            "PASSENGERS": "sum",
            "SEATS": "sum"
        })

    df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01")
    df["LOAD_FACTOR"] = df.apply(lambda row: row["PASSENGERS"] / row["SEATS"] if row["SEATS"] > 0 else 0, axis=1)

    fig_lf = go.Figure()
    fig_passengers = go.Figure()

    if selected_year in [2022, 2023, 2024]:
        df_year = df[df["YEAR"] == int(selected_year)]

        fig_lf.add_trace(go.Scatter(
            x=df_year["DATE"], y=df_year["LOAD_FACTOR"],
            mode='lines+markers', name=f"Load Factor {selected_year}"
        ))

        fig_passengers.add_trace(go.Scatter(
            x=df_year["DATE"], y=df_year["PASSENGERS"],
            mode='lines+markers', name=f"Passengers {selected_year}"
        ))

    elif selected_year == "all":
        fig_lf.add_trace(go.Scatter(
            x=df["DATE"], y=df["LOAD_FACTOR"],
            mode='lines+markers', name="Load Factor (2022–2024)"
        ))

        fig_passengers.add_trace(go.Scatter(
            x=df["DATE"], y=df["PASSENGERS"],
            mode='lines+markers', name="Passengers (2022–2024)"
        ))

    elif selected_year == "forecast_2024":
        forecast_df = forecast_passengers(df, forecast_year=2024)
        fig_lf.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["LOAD_FACTOR"],
            mode='lines+markers', name="Forecast 2024 (LF)",
            line=dict(dash="dot", color="orange")
        ))
        fig_passengers.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["PASSENGERS"],
            mode='lines+markers', name="Forecast 2024 (Passengers)",
            line=dict(dash="dot", color="orange")
        ))

    elif selected_year == "forecast_2025":
        forecast_df = forecast_passengers(df, forecast_year=2025)
        fig_lf.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["LOAD_FACTOR"],
            mode='lines+markers', name="Forecast 2025 (LF)",
            line=dict(dash="dot", color="lightblue")
        ))
        fig_passengers.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["PASSENGERS"],
            mode='lines+markers', name="Forecast 2025 (Passengers)",
            line=dict(dash="dot", color="lightblue")
        ))

    elif selected_year == "all_2025":
        fig_lf.add_trace(go.Scatter(
            x=df["DATE"], y=df["LOAD_FACTOR"],
            mode='lines+markers', name="Load Factor (2022–2024)"
        ))
        fig_passengers.add_trace(go.Scatter(
            x=df["DATE"], y=df["PASSENGERS"],
            mode='lines+markers', name="Passengers (2022–2024)"
        ))

        forecast_df = forecast_passengers(df, forecast_year=2025)
        fig_lf.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["LOAD_FACTOR"],
            mode='lines+markers', name="Forecast 2025 (LF)",
            line=dict(dash="dot", color="lightblue")
        ))
        fig_passengers.add_trace(go.Scatter(
            x=forecast_df["DATE"], y=forecast_df["PASSENGERS"],
            mode='lines+markers', name="Forecast 2025 (Passengers)",
            line=dict(dash="dot", color="lightblue")
        ))

    # Style for both graphs
    for fig in [fig_lf, fig_passengers]:
        fig.update_layout(
            plot_bgcolor='#222222',
            paper_bgcolor='#111111',
            font_color='white',
            xaxis_title="Date"
        )

    fig_lf.update_layout(
        yaxis_title="Load Factor",
        title=f"Load Factor for {origin} → {dest}"
    )

    fig_passengers.update_layout(
        yaxis_title="Passengers",
        title=f"Passenger Volume for {origin} → {dest}"
    )

    return fig_lf, fig_passengers
'''
# Run app
if __name__ == '__main__':
    app.run(debug=True)
