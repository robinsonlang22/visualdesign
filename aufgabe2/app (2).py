import dash
from dash import dcc, html, Input, Output, State, dash_table
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

df = pd.read_csv("wein_cleaned.csv")
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

app = dash.Dash(__name__)

initial_leaderboard = []

# Layout
app.layout = html.Div([
    html.H1("linear regression of wine data"),
    html.H3("please select the two varibles to generate the graph"),

    html.Div([
        html.Label("varible x"),
        dcc.Dropdown(
            id='x-axis',
            options=[{'label': col, 'value': col} for col in numeric_columns],
            value=numeric_columns[0]
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.Label("varible y"),
        dcc.Dropdown(
            id='y-axis',
            options=[{'label': col, 'value': col} for col in numeric_columns],
            value=numeric_columns[1]
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    dcc.Graph(id='regression-plot'),

    html.H4("R² ranking ( score > 0.2 )"),
    dash_table.DataTable(
        id='leaderboard',
        columns=[
            {'name': 'x', 'id': 'X'},
            {'name': 'y', 'id': 'Y'},
            {'name': 'R² score', 'id': 'R2', 'type': 'numeric', 'format': {'specifier': '.4f'}}
        ],
        data=initial_leaderboard,
        sort_action='native',
        style_table={'width': '50%'},
        style_cell={'textAlign': 'center'},
    ),

    dcc.Store(id='leaderboard-store', data=initial_leaderboard)
])

# Callback
@app.callback(
    Output('regression-plot', 'figure'),
    Output('leaderboard', 'data'),
    Output('leaderboard-store', 'data'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    State('leaderboard-store', 'data')
)
def update_graph(x_col, y_col, leaderboard_data):

    x = df[[x_col]].values
    y = df[y_col].values

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = model.score(x, y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.flatten(), y=y, mode='markers', name='data marks', showlegend=False))
    fig.add_trace(go.Scatter(x=x.flatten(), y=y_pred, mode='lines', name='Fit'))

    fig.update_layout(
        title=f"liner regression：{x_col} vs {y_col} <br>R² score: {r2:.4f}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )

    new_entry = {"X": x_col, "Y": y_col, "R2": r2}
    if x_col != y_col and 0.2 < r2 < 1.0 and new_entry not in leaderboard_data:
        leaderboard_data.append(new_entry)

    leaderboard_data = sorted(leaderboard_data, key=lambda x: x['R2'], reverse=True)

    return fig, leaderboard_data, leaderboard_data

if __name__ == '__main__':
    app.run(debug=True)
