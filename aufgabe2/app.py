import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
     dcc.Graph(id='davies-bouldin-plot'),
    dcc.Graph(id='silhouette-plot'),
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
    [Output('clustering-plot', 'figure'), Output('scree-plot', 'figure'),
     Output('davies-bouldin-plot', 'figure'), Output('silhouette-plot', 'figure')],
    Input('num-clusters', 'value')
)
def update_clustering(n_clusters):
    data = df.select_dtypes(include=[np.number])
    scaler =  StandardScaler()
    data = scaler.fit_transform(data)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_

    # PCA zur Dimensionsreduktion
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)



    # Scatterplot des Clusterings (existing code)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=reduced_data[:, 0], y=reduced_data[:, 1], mode='markers',
        marker=dict(color=labels, colorscale='viridis', size=8),
        name="Cluster-Punkte"
    ))
    fig1.update_layout(title="K-Means Clustering mit PCA", xaxis_title="PCA Komponente 1", yaxis_title="PCA Komponente 2")

    # Scree Plot der PCA (updated with cumulative variance and line plot)
    pca_full = PCA()
    pca_full.fit(data)
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    fig2 = go.Figure()

    # Explained variance ratio per component
    fig2.add_trace(go.Scatter(
    x=list(range(1, len(explained_variance_ratio)+1)),
    y=explained_variance_ratio * 100,
    mode='lines+markers',
    name='Erklärte Varianz (%)'
    ))

    # Cumulative variance
    fig2.add_trace(go.Scatter(
    x=list(range(1, len(cumulative_variance)+1)),
    y=cumulative_variance * 100,
    mode='lines+markers',
    name='Kumulative Varianz (%)'
  ))

    fig2.update_layout(
    title="Scree Plot der PCA",
    xaxis_title="PCA-Komponente",
    yaxis_title="Varianz (%)",
    legend_title="Legende"
)

    # Evaluate metrics across a range of cluster counts
    cluster_range = range(2, 5)
    dbi_scores = []
    silhouette_scores = []

    for k in cluster_range:
      km = KMeans(n_clusters=k, random_state=42)
      labels_k = km.fit_predict(data)
      dbi_scores.append(davies_bouldin_score(data, labels_k))
      silhouette_scores.append(silhouette_score(data, labels_k))

    # Davies-Bouldin Index Curve
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
      x=list(cluster_range),
      y=dbi_scores,
      mode='lines+markers',
      name='Davies-Bouldin Index',
      line=dict(color='blue')
      ))
    fig3.add_vline(x=n_clusters, line_dash="dash", line_color="red")
    fig3.update_layout(
    title="Davies-Bouldin Index über verschiedene Clusteranzahlen",
    xaxis_title="Anzahl der Cluster",
    yaxis_title="Davies-Bouldin Index"
    )

    # Silhouette Score Curve
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
      x=list(cluster_range),
      y=silhouette_scores,
      mode='lines+markers',
      name='Silhouette Score',
      line=dict(color='green')
    ))
    fig4.add_vline(x=n_clusters, line_dash="dash", line_color="red")
    fig4.update_layout(
      title="Silhouette Score über verschiedene Clusteranzahlen",
      xaxis_title="Anzahl der Cluster",
      yaxis_title="Silhouette Score"
    )

    return fig1, fig2, fig3, fig4


if __name__ == '__main__':
    app.run(debug=True)