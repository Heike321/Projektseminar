# Dash framework for interactive web apps
import dash
from dash import dcc, html, Input, Output, dash_table

# Data handling and stats
import pandas as pd
from scipy import stats
import json
import datetime
import warnings

# Plotly for visualization
import plotly.graph_objects as go
import plotly.express as px

# Custom analysis and forecasting modules
from analysis import (
    compute_top_routes,
    get_outliers_plot,
    get_seasonality_plot,
    get_trend_plot,
    generate_route_insights,
    generate_combined_route_score,
)
from forecasting import (
    forecast_passengers,
    forecast_load_factor,
    get_forecast_for_year,
    sarima_forecast,
    prepare_forecast_data,
    sarima_forecast_load_factor,
)
from preprocess import iata_to_name

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)



# Load and preprocess data
data = pd.read_csv("Data/Grouped_All_Valid_Connections.csv", dtype={14: str})
data["DATE"] = pd.to_datetime(data["YEAR"].astype(str) + "-" + data["MONTH"].astype(str) + "-01")

with open("Data/valid_routes.json") as f:
    route_options = json.load(f)

# Load precomputed route insights and select the top 10 routes with the highest increasing trend
route_insights_df = pd.read_csv("Data/precomputed_route_insights.csv")
top_routes_df = route_insights_df.sort_values("trend_slope", ascending=False).head(10)


# Get all unique origin IATA codes used in the dataset
iata_codes = data["ORIGIN"].dropna().unique()

# Initialize Dash app 
app = dash.Dash(__name__)
app.title = "Flight Dashboard"

# App layout 
app.layout = html.Div(
    #'backgroundColor': '#111111'black
    style={
    'backgroundImage': 'linear-gradient(to bottom right,#e0f7fa, #0288d1)',
    'color': '#003344',
    'padding': '20px',
    'minHeight': '100vh',
    'fontFamily': 'Segoe UI, Roboto, sans-serif',
    },

    children=[

        
        html.H1("Flight Insights Dashboard ✈️", style={
            'textAlign': 'center',
            'color': '#003344',
            'fontWeight': 'bold',
            'fontSize': 'clamp(20px, 4vw, 40px)'
            }),    
        
        #Dropdowns + KPIs 
        html.Div(style={'display': 'flex'}, children=[
            # LEFT SIDE: Graph and controls
            html.Div(style={'flex': 2, 'marginRight': '20px'}, children=[

                # DROPDOWNS + KPIs side by side
                html.Div(style={'display': 'flex', 'marginBottom': '20px'}, children=[

                    # DROPDOWNS (3/4 width)
                    html.Div(style={'flex': 3, 'marginRight': '10px'}, children=[

                        # Route dropdown (alone at top)
                        html.Div([
                            html.Label("Select a route:"),
                            dcc.Dropdown(
                                id='route-selector',
                                options=route_options,
                                placeholder="Choose a route",
                                style={
                                    'width': '100%',
                                    'backgroundColor': 'white',
                                    'color': 'black',
                                    'borderRadius': '8px',
                                    'boxShadow': '0 2px 6px rgba(0,0,0,0.2)',
                                    'padding': '5px'
                                    }

                            )
                        ], style={'marginBottom': '10px'}),  

                        # Airline and Year dropdowns
                        html.Div(style={'display': 'flex'}, children=[
                            html.Div([
                                html.Label("Select airline:"),
                                html.Span(" ⓘ", title="The Unique Carrier Entity is shown in parentheses only if an airline operates under multiple different carrier entities (e.g., subsidiaries or branches). Otherwise, it is not displayed.."),
                                    
                                dcc.Dropdown(
                                    id='airline-selector',
                                    options=[{"label": "All Airlines", "value": "all"}],
                                    value="all",
                                    style={'width': '100%', 'backgroundColor': 'white', 'color': 'black','borderRadius': '8px',
                                    'boxShadow': '0 2px 6px rgba(0,0,0,0.2)',
                                    'padding': '5px'}
                                )
                            ], style={'flex': 1, 'marginRight': '10px'}),

                            html.Div([
                            html.Label("Select year:"),
                            dcc.Dropdown(
                                id='year-selector',
                                options=[
                                    {"label": "Years 2022-2024", "value": "all"},
                                    {"label": "2022", "value": 2022},
                                    {"label": "2023", "value": 2023},
                                    {"label": "2024", "value": 2024},
                                    {"label": "Forecast 2024", "value": "forecast_2024"},
                                    {"label": "Forecast 2025", "value": "forecast_2025"},
                                    {"label": "Forecast all years", "value": "forecast_all"},

                                ],
                                value="all",
                                clearable=False,
                                style={'width': '100%', 'backgroundColor': 'white', 'color': 'black','borderRadius': '8px',
                                    'boxShadow': '0 2px 6px rgba(0,0,0,0.2)',
                                    'padding': '5px'}
                            )
                        ], style={'flex': 1})
                    ])   
                ]),
                
                # KPIs (RIGHT)
                html.Div(id='kpi-container', style={
                        'flex': 1,
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'space-between',
                        'backgroundColor': '#222222',
                        'padding': '12px',
                        'borderRadius': '10px',
                        'border': '1.5px solid #888888',
                        'minWidth': '140px',
                        'maxWidth': '160px',
                        'height': '140px',
                        'boxShadow': '0 2px 6px rgba(0,0,0,0.15)'
                    }),
                ]),
            
                # Graph
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label='Trend',
                            children=[dcc.Graph(id='trend-graph')],
                            style={
                                'color': 'white',
                                'backgroundColor': '#222222',
                                'borderRadius': '8px 8px 0 0',
                                'padding': '10px',
                                'marginRight': '5px',
                                'transition': 'background-color 0.3s ease',
                            },
                            selected_style={
                                'color': 'orange',
                                'fontWeight': 'bold',
                                'backgroundColor': '#333333',
                                'boxShadow': '0 4px 10px rgba(255, 165, 0, 0.5)',
                            },
                            className='custom-tab'
                        ),
                        dcc.Tab(
                            label='Seasonality',
                            children=[dcc.Graph(id='seasonality-graph')],
                            style={
                                'color': 'white',
                                'backgroundColor': '#222222',
                                'borderRadius': '8px 8px 0 0',
                                'padding': '10px',
                                'marginRight': '5px',
                                'transition': 'background-color 0.3s ease',
                            },
                            selected_style={
                                'color': 'orange',
                                'fontWeight': 'bold',
                                'backgroundColor': '#333333',
                                'boxShadow': '0 4px 10px rgba(255, 165, 0, 0.5)',
                            },
                            className='custom-tab'
                        ),
                        dcc.Tab(
                            label='Outliers',
                            children=[dcc.Graph(id='outliers-graph')],
                            style={
                                'color': 'white',
                                'backgroundColor': '#222222',
                                'borderRadius': '8px 8px 0 0',
                                'padding': '10px',
                                'transition': 'background-color 0.3s ease',
                            },
                            selected_style={
                                'color': 'orange',
                                'fontWeight': 'bold',
                                'backgroundColor': '#333333',
                                'boxShadow': '0 4px 10px rgba(255, 165, 0, 0.5)',
                            },
                            className='custom-tab'
                        ),
                        dcc.Tab(
                            label='Recommendation',
                            children=[
                                html.H4("Recommed top routes sorted by: "),
                                html.Div(
                                    id="focus-container",
                                    children=[
                                        html.Label("Strategic Focus for Combined Score:"),
                                        dcc.Dropdown(
                                            id="focus-profile",
                                            options=[
                                                {"label": "📈 Growth Focus", "value": "growth"},
                                                {"label": "⚙️ Efficiency Focus", "value": "efficiency"},
                                                {"label": "🛡️ Robustness Focus", "value": "robustness"}
                                            ],
                                            value="growth",
                                            clearable=False,
                                            style={'width': '300px'}
                                        ),
                                        html.Div(id="focus-description", style={"marginTop": "10px", "fontStyle": "italic", "color": "#555"})
                                    ],
                                    style={'display': 'none', 'marginBottom': '20px'}  # initial versteckt
                                ),
                                html.Div([
                                    html.Button("Highest Increasing Trend", 
                                        id="t-button", 
                                        n_clicks=0,
                                        style={
                                            'backgroundColor': '#9467bd',
                                            'color': 'white',
                                            'border': 'none',
                                            'padding': '10px 20px',
                                            'margin': '0 10px',
                                            'borderRadius': '5px',
                                            'cursor': 'pointer'
                                        }),
                                    html.Button("Holt-Winters", 
                                        id="hw-button", 
                                        n_clicks=0,
                                        style={
                                            'backgroundColor': '#ff7f0e',
                                            'color': 'white',
                                            'border': 'none',
                                            'padding': '10px 20px',
                                            'margin': '0 10px',
                                            'borderRadius': '5px',
                                            'cursor': 'pointer'
                                        }),
                                    html.Button("SARIMA", 
                                        id="sarima-button", 
                                        n_clicks=0,
                                        style={
                                            'backgroundColor': '#2ca02c',
                                            'color': 'white',
                                            'border': 'none',
                                            'padding': '10px 20px',
                                            'margin': '0 10px',
                                            'borderRadius': '5px',
                                            'cursor': 'pointer'
                                        }),
                                    html.Button("Combined Score",
                                        id="combined-button",
                                        n_clicks=0,
                                        style={
                                            'backgroundColor': '#1f77b4',
                                            'color': 'white',
                                            'border': 'none',
                                            'padding': '10px 20px',
                                            'margin': '0 10px',
                                            'borderRadius': '5px',
                                            'cursor': 'pointer'
                                        }),
                                    html.Span("ℹ️", title="The Combined Score combines several metrics (Trend strength, Seasonality amplitude, MAE Holt error, MAE SARIMA error, Number of passengers, Load factor). Positive weights increase the score, negative weights decrease it."),
                                  
                                ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '20px'}),
                                
                                
                                dash_table.DataTable(
                                    id='analysis-table',
                                    columns=[
                                        {"name": "Route", "id": "route"},
                                        {"name": "Airline", "id": "airline"},
                                        {"name": "Aircraft-Type", "id": "aircraft"},
                                        {"name": "Trend ↑", "id": "trend_slope"},
                                        {"name": "Seasonal Amp.(%)", "id": "season_amp_pct"},
                                        {"name": "Outliers(Residuals)", "id": "outlier_count"},
                                        {"name": "MAE Holt-Winters", "id": "mae_holt"},
                                        {"name": "MAE SARIMA", "id": "mae_sarima"},
                                        {"name": "Combined Score", "id": "score","type": "numeric", "format": {"specifier": ".3f"}}
                                    ],
                                    data=top_routes_df.to_dict("records"),
                                    style_table={'overflowX': 'auto'},
                                    style_cell={'textAlign': 'center'},
                                    style_header={'backgroundColor': '#eeeeee', 'fontWeight': 'bold'},
                                    page_size=5
                                )
                            ],
                            style={
                                'color': 'white',
                                'backgroundColor': '#222222',
                                'borderRadius': '8px 8px 0 0',
                                'padding': '10px',
                                'marginRight': '5px',
                                'transition': 'background-color 0.3s ease',
                            },
                            selected_style={
                                'color': 'orange',
                                'fontWeight': 'bold',
                                'backgroundColor': '#333333',
                                'boxShadow': '0 4px 10px rgba(255, 165, 0, 0.5)',
                            },

                            className='custom-tab'
                        )
                    ],
                    style={
                        'backgroundColor': '#111111',
                        'borderBottom': '2px solid #444444',
                        'paddingBottom': '5px',
                    },
                    colors={
                        'border': '#444444',
                        'primary': 'orange',
                        'background': '#111111',
                    }
                ),
            
                dcc.Store(id='active-sort-method', data='trend'),
                dcc.Graph(id='lf-graph'),
                dcc.Graph(id='passenger-graph')
            ]),
                
                
        #]),

            # RIGHT SIDE:
            html.Div(style={'flex': 1}, children=[
                html.H2("🗺️ Explore Routes from Selected Airport", style={
                    'textAlign': 'center',
                    'color': '#003344',
                    'marginBottom': '10px',
                    'fontSize': 'clamp(12px, 1.4vw, 20px)',
                    }),

                #html.H2("Route Map", style={'textAlign': 'center'}),
                #html.Label("Select origin airport:"),
                dcc.Dropdown(
                    id="origin-dropdown",
                    options = [{"label": f"{iata_to_name.get(iata, iata)} ({iata})", "value": iata} for iata in sorted(iata_codes)],
                    placeholder="Select origin airport",
                    clearable=True,
                    style={'width': '100%', 'backgroundColor': 'white', 'color': 'black','borderRadius': '8px',
                                        'boxShadow': '0 2px 6px rgba(0,0,0,0.2)',
                                        'padding': '5px'}
                        
                ),
                dcc.Graph(id='route-map'),
                html.H3("Top 3 Routes", style={'textAlign': 'center'}),

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
                    style={'width': '100%', 'backgroundColor': 'white', 'color': 'black','borderRadius': '8px',
                                        'boxShadow': '0 2px 6px rgba(0,0,0,0.2)',
                                        'padding': '5px'},
                    clearable=False,
                        
                ),
                html.Label("Select month:"),
                dcc.Dropdown(
                    id='top-routes-month-selector',
                    options=[{"label": "All month", "value": "all"}]+
                        [{"label": str(m), "value": m} for m in range(1, 13)],
                    value=1,
                    style={'width': '100%', 'backgroundColor': 'white', 'color': 'black','borderRadius': '8px',
                                        'boxShadow': '0 2px 6px rgba(0,0,0,0.2)',
                                        'padding': '5px'},
                    clearable=False
                ),


                html.Br(),
                dcc.Graph(id='top-routes-bar'),
                html.Div(id='top-routes-table'),

                html.Div(
                    f"📅 {datetime.datetime.now().strftime('%d.%m.%Y')} • Created by Melanie Schulz & Heike Leers",
                    style={
                        'position': 'absolute',
                        'fontSize': '14px',
                        'color': '#000000',
                        'padding': '10px',
                        'textAlign': 'right',
                        'marginTop': '30px'
                    }
                )
            ])
        ])
    ]
)            

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
            
            # Target marker
            fig.add_trace(go.Scattergeo(
                lon=[row["DEST_LON"]],
                lat=[row["DEST_LAT"]],
                mode='markers',
                marker=dict(size=8, color='red'),
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

    
# Global: Count number of entities per airline in the full dataset
global_entity_counts = data.groupby("UNIQUE_CARRIER_NAME")["UNIQUE_CARRIER_ENTITY"].nunique()
   

# Callback: Update airline dropdown based on selected route
@app.callback(
    Output('airline-selector', 'options'),
    Output('airline-selector', 'value'),
    Input('route-selector', 'value')
)
def update_airline_options(selected_route):
    if not selected_route:
        return [], "all"
    
    # Split route string into origin and destination
    origin, dest = selected_route.split('-')

    # Filter dataset for the selected route
    filtered = data[(data["ORIGIN"] == origin) & (data["DEST"] == dest)]

    

    # Funktion zum Erzeugen des Labels
    def make_label(row):
        airline = row["UNIQUE_CARRIER_NAME"]
        aircraft = str(row["AIRCRAFT_TYPE"])
        #carrier_name = str(row["UNIQUE_CARRIER_ENTITY"])
        entity = str(row["UNIQUE_CARRIER_ENTITY"])
        if global_entity_counts.get(airline,1) > 1:
            return f"{airline} ({aircraft}) [{entity}]"
        else:
            return f"{airline} ({aircraft})"
    
    #Create label column
    filtered["label"] = filtered.apply(make_label, axis=1)



    # Create combined label: Airline Name + Aircraft Type
    #filtered["label"] = (
    #    filtered["UNIQUE_CARRIER_NAME"] + " (" + filtered["AIRCRAFT_TYPE"].astype(str) + ")"+ " [" + filtered["CARRIER_NAME"].astype(str) + "]"
    #)
   
    filtered["value"] = (
        filtered["UNIQUE_CARRIER_NAME"] + "_" + filtered["AIRCRAFT_TYPE"].astype(str)
    )

    # Drop duplicates to get unique airline-aircraft combinations
    unique_labels = filtered[["label", "value"]].drop_duplicates()

    # Create dropdown options with label (airline + aircraft) and value (airline only)
    #options = [{"label": row["label"], "value": row["UNIQUE_CARRIER_NAME"]} for _, row in unique_labels.iterrows()]
    options = [{"label": row["label"], "value": row["value"]} for _, row in unique_labels.iterrows()]

    # Add "All Airlines" option
    options.append({"label": "All Airlines (Aircraft Type)", "value": "all"})
    
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
    top_routes = top_routes.sort_values("PASSENGERS", ascending=False).head(3)

    fig = px.bar(
        top_routes,
        x="ROUTE",
        y="PASSENGERS",
        #title="Top 3 Routes",
        labels={"PASSENGERS": "Number of Passengers", "ROUTE": "Flight route"},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',   
            gridwidth=1,
            griddash='dash'  # 'dash', 'dot', 'dashdot', 'longdash'
        ),
        xaxis=dict(
            showgrid=False  
        ),
        plot_bgcolor='#222222',
        paper_bgcolor='#111111',
        font_color='white'
    )
    fig.update_traces(
        marker=dict(
            color=top_routes["PASSENGERS"],
            colorscale="Blues",     
            #line=dict(width=0)
        ),
        marker_line_width=0,
        width=0.6
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

#Left: 
@app.callback(
    Output('trend-graph', 'figure'),
    Output('seasonality-graph', 'figure'),
    Output('outliers-graph', 'figure'),
    Output('lf-graph', 'figure'),
    Output('passenger-graph', 'figure'),
    Input('route-selector', 'value'),
    Input('airline-selector', 'value'),
    Input('year-selector', 'value')
)
def update_all_graphs(selected_route, selected_airline,  selected_year):
    # Initial empty figures
    trend_fig = no_forecast_figure("No forecast available!")
    seasonality_fig = no_forecast_figure("No forecast available!")
    outliers_fig = no_forecast_figure("No forecast available!")
    lf_fig = go.Figure()
    pax_fig = go.Figure()

    # Return early if no route selected
    if not selected_route:
        return trend_fig, seasonality_fig, outliers_fig, lf_fig, pax_fig
    #airline_name, aircraft_type = selected_airline.strip().rsplit(" (", 1)

    if selected_airline == "all":
        airline_name = None
        aircraft_type = None
    else:
        if "_" not in selected_airline:
            raise ValueError(f"Invalid airline format: {selected_airline}")
        airline_name, aircraft_type = selected_airline.split("_", 1)
    
    #aircraft_type = aircraft_type.rstrip(")")
    origin, dest = selected_route.split('-')

    route_key = f"{origin} → {dest}"
    filtered = prepare_forecast_data(data, route_key, selected_airline if selected_airline != "all" else None)
    
   
    # Add DATE column if not present
    if 'DATE' not in filtered.columns:
        filtered['DATE'] = pd.to_datetime(filtered['YEAR'].astype(str) + '-' + 
                                          filtered['MONTH'].astype(str).str.zfill(2) + '-01')

    # Calculate load factor safely
    filtered = filtered.copy()
    filtered['LOAD_FACTOR'] = filtered.apply(
        lambda row: row['PASSENGERS'] / row['SEATS'] if row['SEATS'] > 0 else 0, axis=1)

    # If forecast selected, generate forecast data and plot
    if isinstance(selected_year, str) and (selected_year.startswith("forecast_") or selected_year == "forecast_all"):
        if selected_year == "forecast_all":
            forecast_years = [2024, 2025]
        else:
            forecast_years = [int(selected_year.split('_')[1])]

        year_label = ', '.join(str(y) for y in forecast_years)
        
        
        # Holt Winter forecast:
        # Get forecast dataframe for the forecast_year
        
        forecast_df = pd.concat([get_forecast_for_year(filtered, year) for year in forecast_years])

        # Filter actual data for forecast year 
        actual_df = filtered.copy()
        
        
        # SARIMA forecast
        sarima_2024_df, err_2024 = sarima_forecast(filtered, forecast_year=2024, route=route_key, airline=airline_name, aircraft_type=aircraft_type)
        sarima_2025_df, err_2025 = sarima_forecast(filtered, forecast_year=2025, route=route_key, airline=airline_name, aircraft_type=aircraft_type)

        # Combine error text
        err = f"{err_2024} | {err_2025}"

        sarima_forecast_load_df = pd.concat([
            sarima_forecast_load_factor(filtered, year) for year in forecast_years])
         
        
        # Filter SARIMA Forecast for forecast_year
        
        sarima_parts = []
        if 2024 in forecast_years and not sarima_2024_df.empty:
            sarima_parts.append(
                sarima_2024_df[sarima_2024_df["TYPE"].str.contains("2024", case=False)]
            )

        if 2025 in forecast_years and not sarima_2025_df.empty:
            sarima_parts.append(
                sarima_2025_df[sarima_2025_df["TYPE"].str.contains("2025", case=False)]
            )

        if sarima_parts:
            sarima_forecast_df = pd.concat(sarima_parts, ignore_index=True)
        else:
            sarima_forecast_df = pd.DataFrame(columns=["DATE", "VALUE", "TYPE"])
        
        
        # Sort data before plotting
        actual_df = actual_df.sort_values('DATE')
        forecast_df = forecast_df.sort_values('DATE')
        sarima_forecast_df = sarima_forecast_df.sort_values('DATE')

        # Actual Load Factor figure - Blue
        lf_fig.add_trace(go.Scatter(
            x=actual_df['DATE'], y=actual_df['LOAD_FACTOR'],
            mode='lines+markers', name=f'Actual {year_label}', 
            line=dict(color='#1f77b4')
        ))

        # Holt-Winters Forecast Load Factor - Orange
        lf_fig.add_trace(go.Scatter(
            x=forecast_df['DATE'], y=forecast_df['FORECAST_LOAD_FACTOR'],
            mode='lines+markers', name=f'Holt-Winters Forecast {year_label}', 
            line=dict(color='#ff7f0e', dash='dot')
        ))

        # SARIMA Forecast Load Factor - Green
        
        lf_fig.add_trace(go.Scatter(
            x=sarima_forecast_load_df['DATE'], y=sarima_forecast_load_df['FORECAST_LOAD_FACTOR'],  
            mode='lines+markers', name=f'SARIMA Forecast Load Factor {year_label}',
            line=dict(color='#2ca02c', dash='dashdot')
        ))
        
        # Actual Passengers figure - Blue
        pax_fig.add_trace(go.Scatter(
            x=actual_df['DATE'], y=actual_df['PASSENGERS'],
            mode='lines+markers', name=f'Actual Passengers {year_label}',
            line=dict(color='#1f77b4')
        ))

        # Holt-Winters Passengers Forecast - Orange
        pax_fig.add_trace(go.Scatter(
            x=forecast_df['DATE'], y=forecast_df['FORECAST_PASSENGERS'],
            mode='lines+markers', name=f'Holt-Winters Forecast Passengers {year_label}',
            line=dict(color='#ff7f0e', dash='dot')
        ))
        
        # SARIMA or AutoARIMA Passengers Forecast - Green
        # Determine forecast source per year
        source_label_2024 = "SARIMA" if "True" in err_2024 else "AutoARIMA"
        source_label_2025 = "SARIMA" if "True" in err_2025 else "AutoARIMA"

        # Add seperate forecast traces per year
        if 2024 in forecast_years and not sarima_2024_df.empty:
            pax_fig.add_trace(go.Scatter(
                x=sarima_2024_df['DATE'], y=sarima_2024_df['VALUE'],
                mode='lines+markers',
                name=f'{source_label_2024} Forecast Passengers 2024',
                line=dict(color='#2ca02c', dash='dashdot')
            ))

        if 2025 in forecast_years and not sarima_2025_df.empty:
            pax_fig.add_trace(go.Scatter(
                x=sarima_2025_df['DATE'], y=sarima_2025_df['VALUE'],
                mode='lines+markers',
                name=f'{source_label_2025} Forecast Passengers 2025',
                line=dict(color='#2ca02c', dash='dashdot')
            ))



    else:
        # For historical years or 'all', filter accordingly
        if selected_year != 'all':
            try:
                year_int = int(selected_year)
                filtered = filtered[filtered['YEAR'] == year_int]
            except Exception:
                pass
        
        # Now create trend, seasonality, outliers plots 
        trend_fig = get_trend_plot(filtered)

        if selected_year == 'all' and len(filtered) >= 24:
            seasonality_fig = get_seasonality_plot(filtered)
        elif selected_year == 'all':
            seasonality_fig = no_forecast_figure("Not enough data for seasonality")
        else:
            seasonality_fig = no_forecast_figure("Seasonality only shown for all years")
        
        outliers_fig = get_outliers_plot(filtered)

        # Load Factor figure for historical data
        filtered_agg = filtered.groupby(['YEAR', 'MONTH'], as_index=False).agg({
            'PASSENGERS': 'sum',
            'SEATS': 'sum'
        })
        filtered_agg['DATE'] = pd.to_datetime(filtered_agg['YEAR'].astype(str) + '-' +
                                             filtered_agg['MONTH'].astype(str).str.zfill(2) + '-01')
        filtered_agg['LOAD_FACTOR'] = filtered_agg.apply(
            lambda row: row['PASSENGERS'] / row['SEATS'] if row['SEATS'] > 0 else 0, axis=1)

        lf_fig.add_trace(go.Scatter(
            x=filtered_agg['DATE'], y=filtered_agg['LOAD_FACTOR'],
            mode='lines+markers', name='Load Factor'
        ))
        pax_fig.add_trace(go.Scatter(
            x=filtered_agg['DATE'], y=filtered_agg['PASSENGERS'],
            mode='lines+markers', name='Passengers'
        ))

    # Set layout themes for lf and pax figures
    for fig in [lf_fig, pax_fig]:
        fig.update_layout(
            plot_bgcolor='#222222',
            paper_bgcolor='#111111',
            font_color='white',
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                #bgcolor='rgba(255,255,255,0.2)',  #semi-transparent legend
                bgcolor='#111111',   # full-coverage legend
                bordercolor='white',
                borderwidth=1
            )
        )

    lf_fig.update_layout(
        title=f"Load Factor for {origin} → {dest}",
        xaxis=dict(
            showgrid=False,          # no vertical lines
            zeroline=False,
            showline=True),
        xaxis_title='Date',
        yaxis=dict(
            showgrid=True,           # only horizontal lines
            gridcolor='rgba(200, 200, 200, 0.3)',  # semi-transparent
            gridwidth=1,
            griddash='dot',          # dashed lines
            zeroline=False,
            showline=False
        ),
        yaxis_title='Load Factor'
    )
    pax_fig.update_layout(
        title=f"Passenger Volume for {origin} → {dest}",
        xaxis_title='Date',
        xaxis=dict(
            showgrid=False,          # no vertical lines
            zeroline=False,
            showline=True),
        yaxis_title='Passengers',
        yaxis=dict(
            showgrid=True,           # only horizontal lines
            gridcolor='rgba(200, 200, 200, 0.3)',  # semi-transparent
            gridwidth=1,
            griddash='dot',          # dashed lines
            zeroline=False,
            showline=False
        ),
    )

    return trend_fig, seasonality_fig, outliers_fig, lf_fig, pax_fig


def no_forecast_figure(message="No forecast available"):
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        text=message,
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

#Callback to update the KPI section based on selected route, airline, and year. It calculates average load factor, maximum passengers on a single flight, 
#and total passengers for the selected criteria.
@app.callback(
    Output('kpi-container', 'children'),
    Input('route-selector', 'value'),
    Input('airline-selector', 'value'),
    Input('year-selector', 'value')
)
def update_kpis(route, airline, year):
    if not route:
        return []

    origin, dest = route.split('-')
    df = data[(data['ORIGIN'] == origin) & (data['DEST'] == dest)].copy()
    if airline != "all":
        carrier_name, aircraft_type = airline.rsplit('_', 1)
        df = df[(df['UNIQUE_CARRIER_NAME'] == carrier_name) & (df['AIRCRAFT_TYPE'].astype(str) == aircraft_type)]

    if year != "all" and isinstance(year, int):
        df = df[df['YEAR'] == year]

    df['LOAD_FACTOR'] = df.apply(lambda row: row['PASSENGERS'] / row['SEATS'] if row['SEATS'] > 0 else 0, axis=1)

    avg_lf = df['LOAD_FACTOR'].mean()
    max_pax = df['PASSENGERS'].max()
    total_passengers = df['PASSENGERS'].sum()

    def kpi_box(label, value, color):
        return html.Div([
            html.Div(label, style={
                'color': 'white',
                'fontSize': '10px',
                'textAlign': 'center',
                'marginBottom': '1px',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            }),
            html.Div(value, style={
                'color': color,
                'fontWeight': 'bold',
                'fontSize': 'clamp(9px, 0.9vw, 20px)',  # responsive font size scaling with viewport width, limited between 9px and 20px
                'textAlign': 'center',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            })
        ], style={
            'marginBottom': '10px',
            'width': '100%',
        })


    color_lf = '#4CAF50' if avg_lf > 0.8 else '#FF5722'
    color_max = '#2196F3' if max_pax > 10000 else '#aaaaaa'
    color_total = '#FFC107'

    return html.Div([
        kpi_box("Ø Load Factor", f"{avg_lf:.2%}", color_lf),
        kpi_box("Max Passengers", f"{max_pax:,}", color_max),
        kpi_box("Total Passengers", f"{total_passengers:,}", color_total)
    ], style={
        'backgroundColor': '#222',
        'padding': '4px',
        'borderRadius': '8px',
        'width': '100%',
        'maxWidth': '100%',
        'boxSizing': 'border-box',
        'overflow': 'hidden',
    })

#Callback to update the recommendation table based on which sorting button is clicked
@app.callback(
    Output("analysis-table", "data"),
    Output("t-button", "style"),
    Output("hw-button", "style"),
    Output("sarima-button", "style"),
    Output("combined-button", "style"),
    Output('analysis-table', 'style_data_conditional'),
    Input("active-sort-method", "data"),
    Input("focus-profile", "value"),
    prevent_initial_call=True
)

def update_recommendation_table(active, focus):
    
    # Sort table data based on selected method
    if active == 'hw':
        sorted_df = top_routes_df.sort_values('mae_holt')
    elif active == 'sarima':
        sorted_df = top_routes_df.sort_values('mae_sarima')
    elif active == 'combined':
        sorted_df = generate_combined_route_score(top_n=10, focus=focus)
    else:  # Default to trend sort
        sorted_df = top_routes_df.sort_values('trend_slope', ascending=False)

    # Define base style (gray for inactive buttons)
    base_style = {
        'backgroundColor': '#444',
        'color': 'white',
        'border': 'none',
        'padding': '10px 20px',
        'margin': '0 10px',
        'borderRadius': '5px',
        'cursor': 'pointer'
    }

    # Define custom active colors
    color_map = {
        'trend': '#9467bd',   # Purple
        'hw': '#ff7f0e',      # Orange
        'sarima': '#2ca02c',   # Green
        'combined': '#1f77b4'
    }

    # Generate button styles dynamically
    def get_style(button_name):
        return {**base_style, 'backgroundColor': color_map[button_name], 'color': 'black'} if active == button_name else base_style
    # Highlight the active sort column in light gray
    highlight_col = {
        'trend': 'trend_slope',
        'hw': 'mae_holt',
        'sarima': 'mae_sarima',
        'combined': 'score'
    }.get(active, 'trend_slope')

    style_data_conditional = [{
        "if": {"column_id": highlight_col},
        "backgroundColor": "#f0f0f0",  # Light gray
        "fontWeight": "bold"
    }]

    # Return updated table and styles
    return (
        sorted_df.to_dict('records'),
        get_style('trend'),
        get_style('hw'),
        get_style('sarima'),
        get_style('combined'),
        style_data_conditional
    )

@app.callback(
    Output("focus-description", "children"),
    Input("focus-profile", "value")
)
def update_focus_description(focus):
    descriptions = {
        "growth": "Prioritizes routes with increasing trends and high passenger volume.",
        "efficiency": "Focuses on routes with high load factor and low forecast error.",
        "robustness": "Emphasizes stable routes with low seasonality and fewer outliers."
    }
    return descriptions.get(focus, "")



@app.callback(
    Output("focus-container", "style"),
    Input("combined-button", "n_clicks"),
    prevent_initial_call=True
)
def show_focus_dropdown(n_clicks):
    if n_clicks and n_clicks > 0:
        return {'display': 'block', 'marginBottom': '20px'}
    return {'display': 'none'}

@app.callback(
    Output("active-sort-method", "data"),
    Input("t-button", "n_clicks"),
    Input("hw-button", "n_clicks"),
    Input("sarima-button", "n_clicks"),
    Input("combined-button", "n_clicks"),
    prevent_initial_call=True
)
def update_active_sort(trend, hw, sarima, combined):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return {
        "t-button": "trend",
        "hw-button": "hw",
        "sarima-button": "sarima",
        "combined-button": "combined"
    }.get(triggered_id, "trend")

# Run app
if __name__ == '__main__':
    app.run(debug=True)
