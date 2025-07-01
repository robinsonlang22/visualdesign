import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import dash
from dash import dcc  
from dash import html 
import dash_table
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import scipy
import io
import base64
import matplotlib.pyplot as plt
scipy.interp = np.interp
import scikitplot as skplt

# 2.1 Datensatz laden
df = pd.read_csv("cleaned_pulsar_data.csv")
X = df.drop("target_class", axis=1)
y = df["target_class"]

# 2.2 Split in Training/Test (70/30) mit Stratifikation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2.3 Standardisieren
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 2.4 GridSearch für SVM-Hyperparameter
param_grid = {
    "C":      [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma":  ["scale", 0.01, 0.1, 1]
}

grid = GridSearchCV(
    estimator=SVC(probability=True),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    return_train_score=True
)
grid.fit(X_train_scaled, y_train)

# 2.5 Ergebnisse extrahieren
results_df = pd.DataFrame(grid.cv_results_)[
    ["param_C", "param_kernel", "param_gamma", "mean_train_score", "mean_test_score"]
]



from dash.dependencies import Input, Output
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

# 3.1 Dash-App initialisieren
app = dash.Dash(__name__)

# 3.2 Layout definieren
app.layout = html.Div([
    html.H1("SVM-Optimierung (Pulsar Data)"),

    dcc.Dropdown(
        id="param-dropdown",
        options=[
            {
                "label": f"C={row.param_C}, kernel={row.param_kernel}, gamma={row.param_gamma}",
                "value": idx
            }
            for idx, row in results_df.iterrows()
        ],
        value=0,
        clearable=False
    ),

    html.Div([
        html.Div([
            html.H3("ROC Curve"),
            html.Img(id="roc-image")
        ], style={"width": "48%", "display": "inline-block"}),

        html.Div([
            html.H3("Konfusionsmatrix"),
            dcc.Graph(id="cm-graph")
        ], style={"width": "48%", "display": "inline-block", "float": "right"})
    ]),

    html.H3("GridSearchCV Ergebnisse"),
    dash_table.DataTable(
        id="results-table",
        columns=[{"name": c, "id": c} for c in results_df.columns],
        data=results_df.to_dict("records"),
        page_size=10,
        style_table={"overflowX": "auto"}
    )
])

# 3.3 Callback für ROC (matplotlib→Base64) und Confusionsmatrix (Plotly)
@app.callback(
    [Output("roc-image", "src"),
     Output("cm-graph", "figure")],
    [Input("param-dropdown", "value")]
)
def update_plots(selected_idx):
    # 3.3.1 Modell nach ausgewählten Parametern trainieren
    params = results_df.iloc[selected_idx]
    model = SVC(
        C=params.param_C,
        kernel=params.param_kernel,
        gamma=params.param_gamma,
        probability=True
    ).fit(X_train_scaled, y_train)

    # 3.3.2 ROC-Kurve mit scikit-plot
    y_probas = model.predict_proba(X_test_scaled)
    fig, ax = plt.subplots(figsize=(6, 6))
    skplt.metrics.plot_roc(y_test, y_probas, ax=ax, plot_micro=False, plot_macro=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
    img_src = f"data:image/png;base64,{img_base64}"

    # 3.3.3 Konfusionsmatrix mit Plotly
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Nicht-Pulsar","Pulsar"],
        y=["Nicht-Pulsar","Pulsar"],
        colorscale="Blues",
        showscale=True
    ))
    cm_fig.update_layout(
        title="Konfusionsmatrix",
        xaxis_title="Vorhergesagt",
        yaxis_title="Tatsächlich"
    )

    return img_src, cm_fig

# 3.4 Server starten
if __name__ == "__main__":
    app.run(debug=True)
