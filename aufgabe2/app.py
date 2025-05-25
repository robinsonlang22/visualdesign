import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

df = pd.read_csv("wein_cleaned.csv")
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("linear regression of wine data"),

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
])

# Callback
@app.callback(
    Output('regression-plot', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value')
)
def update_graph(x_col, y_col):
    x = df[[x_col]].values
    y = df[y_col].values

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.flatten(), y=y, mode='markers', name='data marks'))
    fig.add_trace(go.Scatter(x=x.flatten(), y=y_pred, mode='lines', name='linear regression'))

    fig.update_layout(
        title=f"liner regression：{x_col} → {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
