import pandas as pd
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"



df = pd.read_csv('titanic.csv')

y = df['Survived']

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Embarked']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

models = {
    'Logistische Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3)
}

def evaluate_cv(model, X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kf)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }

def evaluate_bootstrap(model, X, y, n_iterations=100):
    y_preds, y_tests = [], []
    X = pd.DataFrame(X)  # Make indexable
    y = pd.Series(y).reset_index(drop=True)

    for _ in range(n_iterations):
        X_resampled, y_resampled = resample(X, y, replace=True)
        mask = ~X.index.isin(X_resampled.index)
        if mask.sum() == 0:
            continue
        X_test, y_test = X.loc[mask], y.loc[mask]
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        y_preds.extend(y_pred)
        y_tests.extend(y_test)

    return {
        'accuracy': accuracy_score(y_tests, y_preds),
        'precision': precision_score(y_tests, y_preds),
        'recall': recall_score(y_tests, y_preds),
        'f1': f1_score(y_tests, y_preds),
        'confusion_matrix': confusion_matrix(y_tests, y_preds)
    }

cv_results = {}
bootstrap_results = {}

for name, model in models.items():
    # print(f"\nModel: {name}")
    cv_res = evaluate_cv(model, X_processed, y)
    boot_res = evaluate_bootstrap(model, X_processed, y)

    cv_results[name] = cv_res
    bootstrap_results[name] = boot_res

    # print("--- 10-fold Cross Validation ---")
    # print(f"Accuracy: {cv_res['accuracy']:.3f}, Precision: {cv_res['precision']:.3f}, Recall: {cv_res['recall']:.3f}, F1: {cv_res['f1']:.3f}")
    # print("Confusion Matrix:\n", cv_res['confusion_matrix'])

    # print("--- Bootstrapping (0.632) ---")
    # print(f"Accuracy: {boot_res['accuracy']:.3f}, Precision: {boot_res['precision']:.3f}, Recall: {boot_res['recall']:.3f}, F1: {boot_res['f1']:.3f}")
    # print("Confusion Matrix:\n", boot_res['confusion_matrix'])

tree_cv = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_cv.fit(X_processed, y)

cat_ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names) + ['Pclass']

plt.figure(figsize=(20, 10))
plot_tree(tree_cv, feature_names= feature_names, class_names=["Not Survived", "Survived"],
          filled=True, rounded=True)
plt.savefig("tree_cv.png")
plt.close()

cv_results = {
    'Logistische Regression': {
        'accuracy': 0.799,
        'precision': 0.762,
        'recall': 0.693,
        'f1': 0.726,
        'confusion_matrix': [[475, 74], [105, 237]]
    },
    'Decision Tree': {
        'accuracy': 0.780,
        'precision': 0.707,
        'recall': 0.728,
        'f1': 0.718,
        'confusion_matrix': [[446, 103], [93, 249]]
    },
    'KNN (k=3)': {
        'accuracy': 0.801,
        'precision': 0.755,
        'recall': 0.713,
        'f1': 0.734,
        'confusion_matrix': [[470, 79], [98, 244]]
    }
}

bootstrap_results = {
    'Logistische Regression': {
        'accuracy': 0.797,
        'precision': 0.750,
        'recall': 0.702,
        'f1': 0.725,
        'confusion_matrix': [[17455, 2927], [3745, 8803]]
    },
    'Decision Tree': {
        'accuracy': 0.769,
        'precision': 0.695,
        'recall': 0.708,
        'f1': 0.701,
        'confusion_matrix': [[16282, 3902], [3660, 8874]]
    },
    'KNN (k=3)': {
        'accuracy': 0.763,
        'precision': 0.689,
        'recall': 0.690,
        'f1': 0.690,
        'confusion_matrix': [[16297, 3889], [3868, 8618]]
    }
}

def create_metrics_df(results_dict):
    metric_list = ['accuracy', 'precision', 'recall', 'f1']
    data_for_chart = []
    for model_name, scores in results_dict.items():
        for metric in metric_list:
            data_for_chart.append({
                'Model': model_name,
                'Metric': metric.capitalize(),
                'Score': scores[metric]
            })
    return pd.DataFrame(data_for_chart)

def create_zoomable_image(src):
    fig = go.Figure()

    # Add the image to the layout
    fig.add_layout_image(
        dict(
            source=src,
            xref="paper",  # Relative positioning to the paper domain
            yref="paper",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="contain",  # Ensures the entire image is visible
            opacity=1,
            layer="below"
        )
    )

    # Hide axis visuals
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="zoom",  # Allows zooming via drag
        hovermode=False,
    )

    return fig

df_cv = create_metrics_df(cv_results)
df_bootstrap = create_metrics_df(bootstrap_results)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Model Evaluation Dashboard"),

    dcc.Tabs(id='tabs', value='cv', children=[
        dcc.Tab(label="Cross Validation (10-Fold)", value='cv'),
        dcc.Tab(label="Bootstrapping (0.632)", value='bootstrap'),
    ]),

    html.Div(id='metrics-graph'),

    html.H2("Confusion Matrix"),
    html.Div([
        html.Label("Select Model:"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': name, 'value': name} for name in cv_results.keys()],
            value='Logistische Regression',
            clearable=False,
            style={'width': '50%'}
        ),
        dcc.Graph(id='confusion-matrix-graph'),
    ]),

    html.H2("Decision Tree Visualization"),
    dcc.Graph(id='tree-zoomable', config={'scrollZoom': True})  # Enable scroll zoom
])
# Callback for updating the zoomable image
@app.callback(
    Output('tree-zoomable', 'figure'),
    Input('tabs', 'value')
)
def update_tree_zoomable(tab):
    # You can adjust this logic if needed based on the tab input
    src = encode_image("tree_cv.png")
    fig = create_zoomable_image(src)
    return fig

def update_tree_image(tab):
    return encode_image("tree_cv.png")

# Callback to update metrics bar chart
@app.callback(
    Output('metrics-graph', 'children'),
    Input('tabs', 'value')
)

def update_metrics_graph(selected_tab):
    if selected_tab == 'cv':
        df = df_cv
        title = "Cross-Validation Results"
    else:
        df = df_bootstrap
        title = "Bootstrapping Results"

    fig = px.bar(df, x='Model', y='Score', color='Metric',
                 barmode='group', title=title)
    return dcc.Graph(figure=fig)

# Callback to update confusion matrix
@app.callback(
    Output('confusion-matrix-graph', 'figure'),
    [Input('model-dropdown', 'value'),
     Input('tabs', 'value')]
)




def update_confusion_matrix(selected_model, selected_tab):
    if selected_tab == 'cv':
        cm = cv_results[selected_model]['confusion_matrix']
        title = f"{selected_model} - Confusion Matrix (Cross Validation)"
    else:
        cm = bootstrap_results[selected_model]['confusion_matrix']
        title = f"{selected_model} - Confusion Matrix (Bootstrapping)"

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        showscale=True
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)