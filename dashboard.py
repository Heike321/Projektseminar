import dash
from dash import dcc, html, Input, Output
import pandas as pd
import json
import plotly.graph_objects as go
from analysis import compute_top_airlines, compute_top_routes

# Load preprocessed combined CSV data
data = pd.read_csv("Data/Grouped_All_Valid_Connections.csv")

# Load list of valid connections (filtered based on data availability and threshold)
with open("valid_connections.json") as f:
    valid_connections = json.load(f)


data["DATE"] = pd.to_datetime(data["YEAR"].astype(str) + "-" + data["MONTH"].astype(str) + "-01")

# Initialize the Dash web application
app = dash.Dash(__name__)

# Define the layout (UI structure) of the dashboard
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': 'white', 'padding': '20px'}, children=[
    html.H1("Flight Connection Dashboard", style={'textAlign': 'center'}),

    html.Div([
        # Dropdown to select a specific flight connection (based on con_key)
        html.Label("Select a connection:"),
        dcc.Dropdown(
            id='connection-selector',
            options=json.load(open("valid_connections.json")),
            placeholder="Choose a connection",
            style={'backgroundColor': 'white', 'color': 'black'}
        ),

       
        # Dropdown to select year
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
            style={'backgroundColor': 'white', 'color': 'black'}
        )


    ], style={'marginBottom': '20px'}),

    # Graph component where the time series plot will be rendered
    dcc.Graph(id='lf-graph'),

    html.Div([
        html.H2("Top 10 Routes"),
        dcc.Graph(id='top-routes-graph'),
        html.H2("Top 10 Airlines"),
        dcc.Graph(id='top-airlines-graph'),
    ])
])

# Define callback to update the graph based on user input
@app.callback(
    Output('lf-graph', 'figure'),                 # Output goes to the figure of the graph
    Input('connection-selector', 'value'),        # Triggered when a connection is selected
    Input('year-selector','value')                  # Triggered when a load factor variant is selected
)
def update_graph(con_key, selected_year):
    if not con_key:
        # If no connection is selected, return an empty figure
        return go.Figure()

    # Filter the dataset for the selected connection
    sub = data[data["con_key"] == con_key]

    # Further filter by selected year (if not 'all')
    if selected_year != "all":
        sub = sub[sub["YEAR"] == int(selected_year)]

    # Create a line chart 
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["DATE"],
        y=sub["LOAD_FACTOR"],
        mode='lines+markers',
        name="Load Factor"
    ))

    # Customize the appearance of the chart (dark theme)
    fig.update_layout(
        title=f"Load Factor for {con_key}",
        xaxis_title="Date",
        yaxis_title="Load Factor",
        plot_bgcolor='#222222',
        paper_bgcolor='#111111',
        font_color='white'
    )
    return fig

@app.callback(
    Output('top-routes-graph', 'figure'),
    Output('top-airlines-graph', 'figure'),
    Input('year-selector', 'value')
)
def update_top_charts(selected_year):
    if selected_year == "all":
        df = data
    else:
        df = data[data["YEAR"] == int(selected_year)]

    top_routes_df = compute_top_routes(df)
    top_airlines_df = compute_top_airlines(df)

    fig_routes = go.Figure()
    fig_routes.add_trace(go.Bar(
        x=top_routes_df["ROUTE"],
        y=top_routes_df["PASSENGERS"],
        marker_color='lightblue'
    ))
    fig_routes.update_layout(
        title="Top 10 Routes by Passenger Volume",
        xaxis_title="Route",
        yaxis_title="Passengers",
        plot_bgcolor='#222222',
        paper_bgcolor='#111111',
        font_color='white'
    )

    fig_airlines = go.Figure()
    fig_airlines.add_trace(go.Bar(
        x=top_airlines_df["AIRLINE_ID"].astype(str),
        y=top_airlines_df["PASSENGERS"],
        marker_color='orange'
    ))
    fig_airlines.update_layout(
        title="Top 10 Airlines by Passenger Volume",
        xaxis_title="Airline ID",
        yaxis_title="Passengers",
        plot_bgcolor='#222222',
        paper_bgcolor='#111111',
        font_color='white'
    )

    return fig_routes, fig_airlines
   

# Start the Dash web server when the script is run directly
if __name__ == '__main__':
    app.run(debug=True)
