import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("wein_cleaned.csv")
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Wine Data Analysis"),

    # Auswahl für Lineare Regression
    html.Div([
        html.Label("Variable X"),
        dcc.Dropdown(
            id='x-axis',
            options=[{'label': col, 'value': col} for col in numeric_columns],
            value=numeric_columns[0]
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Variable Y"),
        dcc.Dropdown(
            id='y-axis',
            options=[{'label': col, 'value': col} for col in numeric_columns],
            value=numeric_columns[1]
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    dcc.Graph(id='regression-plot'),

    # Auswahl für K-Means Clustering
    html.Div([
        html.Label("Anzahl der Cluster"),
        dcc.Dropdown(
            id='num-clusters',
            options=[{'label': str(k), 'value': k} for k in range(2, 6)],
            value=3
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    dcc.Graph(id='clustering-plot'),
    dcc.Graph(id='scree-plot'),
])

# Callback für lineare Regression
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
    fig.add_trace(go.Scatter(x=x.flatten(), y=y, mode='markers', name='Datenpunkte'))
    fig.add_trace(go.Scatter(x=x.flatten(), y=y_pred, mode='lines', name='Lineare Regression'))

    fig.update_layout(
        title=f"Lineare Regression: {x_col} → {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )

    return fig

# Callback für K-Means Clustering und PCA
@app.callback(
    [Output('clustering-plot', 'figure'), Output('scree-plot', 'figure')],
    Input('num-clusters', 'value')
)
def update_clustering(n_clusters):
    data = df.select_dtypes(include=[np.number])
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)

    # PCA zur Dimensionsreduktion
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Scatterplot des Clusterings
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=reduced_data[:, 0], y=reduced_data[:, 1], mode='markers',
        marker=dict(color=labels, colorscale='viridis', size=8),
        name="Cluster-Punkte"
    ))
    fig1.update_layout(title="K-Means Clustering mit PCA", xaxis_title="PCA Komponente 1", yaxis_title="PCA Komponente 2")

    # Scree Plot der PCA
    explained_variance = pca.explained_variance_ratio_
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=list(range(1, len(explained_variance)+1)), y=explained_variance * 100))
    fig2.update_layout(title="Scree Plot der PCA", xaxis_title="PCA-Komponente", yaxis_title="Erklärte Varianz (%)")

    return fig1, fig2

if __name__ == '__main__':
    app.run(debug=True)
