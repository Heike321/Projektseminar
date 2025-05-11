import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import json
import plotly.graph_objects as go
from analysis import compute_top_routes

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
    html.Div(style={'display': 'flex', 'gap': '40px'}, children=[
    # Left
    html.Div(style={'flex': 2}, children=[
        dcc.Graph(id='lf-graph')
    ]),

    # right
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
            style={'backgroundColor': 'white', 'color': 'black'}
        ),

        html.Br(),

        # Tabelle
        html.Div(id='top-routes-table')
    ])
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
    Output('top-routes-table', 'children'),
    Input('top-routes-year-selector', 'value')
)
def update_top_routes_table(selected_year):
    if selected_year == "all":
        df_filtered = data
    else:
        df_filtered = data[data["YEAR"] == int(selected_year)]

    top_routes = compute_top_routes(df_filtered)

    return dash_table.DataTable(
        columns=[
            {"name": "Route", "id": "ROUTE"},
            {"name": "Passengers", "id": "PASSENGERS", "type": "numeric", "format": {"specifier": ","}}
        ],
        data=top_routes.to_dict("records"),
        style_table={'overflowX': 'auto'},
        style_cell={
            'backgroundColor': '#111111',
            'color': 'white',
            'padding': '8px',
        },
        style_header={
            'backgroundColor': '#222222',
            'fontWeight': 'bold'
        }
    )
   

# Start the Dash web server when the script is run directly
if __name__ == '__main__':
    app.run(debug=True)
