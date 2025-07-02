# =============================================================================
# svm_dash_no_shap_swapped_axes_wide_fixed_scale.py
#
# Pulsar Data SVM mit
#  - GridSearchCV
#  - Test-Metriken
#  - statischen Heatmaps (Achsen getauscht), Lern‐/Validierungskurven, t-SNE
#  - Dash-App mit interaktiven ROC/PR‐Curve + Threshold
#  - Tabs für statische Plots, Heatmap extra breit/hoch, Color-Scale von 0–1 fix
# =============================================================================

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
scipy.interp = np.interp  # Monkey‐patch für scikit-plot
import scikitplot as skplt

from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    learning_curve, validation_curve
)
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score,
    f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output

sns.set()


# 1) Daten laden & splitten
df = pd.read_csv("cleaned_pulsar_data.csv")
X = df.drop("target_class", axis=1)
y = df["target_class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2) GridSearchCV
param_grid = {
    "C":      [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma":  ["scale", 0.01, 0.1, 1]
}
grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    scoring="accuracy",
    cv=5, n_jobs=-1,
    return_train_score=True
)
grid.fit(X_train, y_train)

results_df = pd.DataFrame(grid.cv_results_)[
    ["param_C","param_kernel","param_gamma","mean_train_score","mean_test_score"]
]
best_model = grid.best_estimator_

# 3) Test-Metriken
metrics = []
for _, r in results_df.iterrows():
    m = SVC(C=r.param_C, kernel=r.param_kernel, gamma=r.param_gamma, probability=True)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    proba = m.predict_proba(X_test)[:,1]
    metrics.append({
        "test_accuracy" : accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall"   : recall_score(y_test, pred),
        "test_f1"       : f1_score(y_test, pred),
        "test_auc"      : roc_auc_score(y_test, proba)
    })
metrics_df = pd.DataFrame(metrics)
results_df = pd.concat([results_df.reset_index(drop=True), metrics_df], axis=1)

# 4) Statische Plotly-Figuren

# 4.1 Heatmap (C on y, Gamma on x), fixe Color-Scale 0–1
pivot_val = results_df.pivot_table(
    index="param_C", columns="param_gamma",
    values="mean_test_score", aggfunc="mean"
)
# Get actual min/max values from the pivot
zmin = pivot_val.min().min()
zmax = pivot_val.max().max()

fig_heatmap = go.Figure(data=go.Heatmap(
    z=pivot_val.values,
    x=[str(x) for x in pivot_val.columns],
    y=[str(y) for y in pivot_val.index],
    colorscale="magma",
    zmin=zmin,
    zmax=zmax,
    colorbar=dict(title="Test Accuracy")
))

fig_heatmap.update_layout(
    title="Heatmap: Gamma vs. C (Test Accuracy)",
    autosize=False,
    width=1200,
    height=600,
    margin=dict(l=80, r=80, t=60, b=100),
    xaxis=dict(title="Gamma", tickangle=-45),
    yaxis=dict(title="C")
)




# 4.2 Lernkurve
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1,1.0,5),
    scoring="accuracy", n_jobs=-1
)
fig_lc = go.Figure([
    go.Scatter(x=train_sizes, y=train_scores.mean(axis=1),
               mode="lines+markers", name="Train"),
    go.Scatter(x=train_sizes, y=val_scores.mean(axis=1),
               mode="lines+markers", name="Validation")
])
fig_lc.update_layout(
    title="Lernkurve",
    xaxis_title="Trainingssamples",
    yaxis_title="Accuracy"
)

# 4.3 Validierungskurve für C
param_range = np.logspace(-2,2,5)
ts_train, ts_val = validation_curve(
    best_model, X_train, y_train,
    param_name="C", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1
)
fig_vc = go.Figure([
    go.Scatter(x=param_range, y=ts_train.mean(axis=1),
               mode="lines+markers", name="Train"),
    go.Scatter(x=param_range, y=ts_val.mean(axis=1),
               mode="lines+markers", name="Validation")
])
fig_vc.update_layout(
    title="Validierungskurve für C",
    xaxis_type="log", xaxis_title="C", yaxis_title="Accuracy"
)

# 4.4 t-SNE Projektion
X_emb = TSNE(n_components=2, random_state=42).fit_transform(X_test)
y_pred = best_model.predict(X_test)
fig_tsne = px.scatter(
    x=X_emb[:,0], y=X_emb[:,1],
    color=y_pred.astype(str),
    labels={"color":"Predicted"},
    title="t-SNE Projektion (Predicted Labels)"
)

# 5) Dash-App
app = Dash(__name__)
app.layout = html.Div([
    html.H1("SVM Explorer – Pulsar Data"),

    # Parameter Dropdown
    html.Div([
        html.Label("Hyperparameter"),
        dcc.Dropdown(
            id="param-dropdown",
            options=[
                {"label":f"C={r.param_C}, k={r.param_kernel}, γ={r.param_gamma}",
                 "value":i}
                for i,r in results_df.iterrows()
            ],
            value=0, clearable=False, style={"width":"60%"}
        )
    ], style={"margin":"20px 0"}),

    # ROC/PR und Threshold
    html.Div([
        dcc.RadioItems(
            id="curve-type",
            options=[
                {"label":"ROC Curve","value":"roc"},
                {"label":"Precision-Recall","value":"pr"}
            ],
            value="roc", labelStyle={"display":"inline-block","margin-right":"15px"}
        ),
        html.Br(),
        html.Label("Threshold"),
        dcc.Slider(
            id="threshold-slider",
            min=0, max=1, step=0.01, value=0.5,
            marks={0:"0.0",0.5:"0.5",1:"1.0"}
        ),
        html.Div(id="threshold-output", style={"margin-top":"5px"})
    ], style={"margin-bottom":"30px"}),

    # Interaktive Plots
    html.Div([
        dcc.Graph(id="perf-curve", style={"width":"48%","display":"inline-block"}),
        dcc.Graph(id="cm-graph",   style={"width":"48%","display":"inline-block"})
    ]),

    # Tabs für statische Auswertungen
    dcc.Tabs([
        dcc.Tab(label="Heatmap", children=[
            dcc.Graph(figure=fig_heatmap,
                      style={"width":"100%","height":"600px"})
        ]),
        dcc.Tab(label="Lernkurve", children=[dcc.Graph(figure=fig_lc)]),
        dcc.Tab(label="Validierungskurve", children=[dcc.Graph(figure=fig_vc)]),
        dcc.Tab(label="t-SNE", children=[dcc.Graph(figure=fig_tsne)])
    ]),

    # GridSearchCV Tabelle
    html.H3("GridSearchCV Results"),
    dash_table.DataTable(
        id="results-table",
        columns=[{"name":c,"id":c} for c in results_df.columns],
        data=results_df.to_dict("records"),
        page_size=10,
        style_cell={"textAlign":"left"},
        style_table={"overflowX":"auto"}
    )
])

@app.callback(
    Output("perf-curve","figure"),
    Output("cm-graph","figure"),
    Output("threshold-output","children"),
    Input("param-dropdown","value"),
    Input("curve-type","value"),
    Input("threshold-slider","value")
)
def update_plots(idx, curve_type, threshold):
    r = results_df.iloc[idx]
    m = SVC(C=r.param_C, kernel=r.param_kernel, gamma=r.param_gamma, probability=True)
    m.fit(X_train, y_train)
    y_proba = m.predict_proba(X_test)[:,1]

    # Performance-Kurve
    if curve_type == "roc":
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        perf_fig = go.Figure([
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.2f}"),
            go.Scatter(x=[0,1], y=[0,1], mode="lines",
                       line=dict(dash="dash"), name="Random")
        ])
        perf_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    else:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(rec, prec)
        perf_fig = go.Figure([
            go.Scatter(x=rec, y=prec, mode="lines", name=f"AUCPR={pr_auc:.2f}"),
            go.Scatter(x=[0,1], y=[y_test.mean()]*2, mode="lines",
                       line=dict(dash="dash"), name="Baseline")
        ])
        perf_fig.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")

    # Confusion Matrix
    pred_label = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred_label)
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Non-Pulsar","Pulsar"], y=["Non-Pulsar","Pulsar"],
        colorscale="Blues", showscale=True
    ))
    cm_fig.update_layout(title=f"Confusion Matrix (thr={threshold:.2f})",
                         xaxis_title="Predicted", yaxis_title="Actual")

    return perf_fig, cm_fig, f"Threshold = {threshold:.2f}"

if __name__ == "__main__":
    app.run(debug=True)