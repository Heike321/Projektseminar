# Dash
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Load the grouped flight data from the new Excel file
df = pd.read_excel("Data/Grouped_Valid_Connections.xlsx")

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Flight Connections Dashboard ✈️", style={'color': 'white'}),
    
    html.Div([
        html.Label("Select Year:", style={'color': 'white'}),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': y, 'value': y} for y in sorted(df['YEAR'].dropna().unique())],
            value=2022
        ),
    ], style={'width': '25%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Select Month:", style={'color': 'white'}),
        dcc.Dropdown(
            id='month-dropdown',
            options=[{'label': m, 'value': m} for m in sorted(df['MONTH'].dropna().unique())],
            value=1
        ),
    ], style={'width': '25%', 'display': 'inline-block', 'marginLeft': '2%'}),
    html.Div(style={'height': '40px'}),  # spacer for vertical spacing

    dcc.Graph(id='passenger-bar-chart'),
    dcc.Graph(id='loadfactor-bar-chart')
    
    

],
style={'backgroundColor': 'black', 'padding': '20px'})

# Callback to update the bar chart based on selected year and month
@app.callback(
    [Output('passenger-bar-chart', 'figure'),
     Output('loadfactor-bar-chart', 'figure')],
    [Input('year-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)
def update_graphs(selected_year, selected_month):
    # Filter
    filtered_df = df[(df['YEAR'] == selected_year) & (df['MONTH'] == selected_month)].copy()
    filtered_df = filtered_df[filtered_df['SEATS'] > 0]  # Schutz vor Division durch 0
    filtered_df['LOAD_FACTOR'] = filtered_df['PASSENGERS'] / filtered_df['SEATS']

    # Gruppierung
    grouped = filtered_df.groupby(['ORIGIN', 'DEST'], as_index=False).agg({
        'PASSENGERS': 'sum',
        'SEATS': 'sum'
    })
    grouped['ROUTE'] = grouped['ORIGIN'] + " → " + grouped['DEST']
    grouped['LOAD_FACTOR'] = grouped['PASSENGERS'] / grouped['SEATS']

    # Diagramm 1: Passagierzahlen
    fig_passengers = px.bar(
        grouped.sort_values('PASSENGERS', ascending=False).head(20),
        x='ROUTE',
        y='PASSENGERS',
        title=f"Top 20 Routes by Number of Passengers ({selected_year}-{selected_month})"
    )
    fig_passengers.update_layout(xaxis_tickangle=-45)

    # Diagramm 2: Auslastung
    fig_loadfactor = px.bar(
        grouped.sort_values('LOAD_FACTOR', ascending=False).head(20),
        x='ROUTE',
        y='LOAD_FACTOR',
        title=f"Top 20 Routes by Load Factor ({selected_year}-{selected_month})",
        labels={'LOAD_FACTOR': 'Load Factor'}
    )
    fig_loadfactor.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")

    return fig_passengers, fig_loadfactor


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

