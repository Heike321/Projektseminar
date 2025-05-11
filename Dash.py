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

    dcc.Graph(id='passenger-bar-chart')

],
style={'backgroundColor': 'black', 'padding': '20px'})

# Callback to update the bar chart based on selected year and month
@app.callback(
    Output('passenger-bar-chart', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)
def update_graph(selected_year, selected_month):
    # Filter the data based on user input
    filtered_df = df[(df['YEAR'] == selected_year) & (df['MONTH'] == selected_month)]

    # Group by route and sum passengers
    grouped = filtered_df.groupby(['ORIGIN', 'DEST'], as_index=False)['PASSENGERS'].sum()
    grouped['ROUTE'] = grouped['ORIGIN'] + " → " + grouped['DEST']

    # Create bar chart
    fig = px.bar(
        grouped.sort_values('PASSENGERS', ascending=False).head(20),
        x='ROUTE',
        y='PASSENGERS',
        title=f"Top 20 Routes by Number of Passengers ({selected_year}-{selected_month})",
        labels={'PASSENGERS': 'Number of Passengers', 'ROUTE': 'Route'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
