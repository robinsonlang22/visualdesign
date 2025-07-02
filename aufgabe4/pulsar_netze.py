import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
import plotly.express as px
import base64

df = pd.read_csv('cleaned_pulsar_data.csv')

df.head()

X = df.drop(columns=['target_class'])
y = df['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

simple_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])
simple_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

complex_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
complex_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train two models
history_simple = simple_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
history_complex = complex_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

np.save("history_simple_acc.npy", history_simple.history['accuracy'])
np.save("history_simple_val_acc.npy", history_simple.history['val_accuracy'])
np.save("history_complex_acc.npy", history_complex.history['accuracy'])
np.save("history_complex_val_acc.npy", history_complex.history['val_accuracy'])

# model predication & evaluation
y_pred_simple = (simple_model.predict(X_test) > 0.5).astype("int32")
y_pred_complex = (complex_model.predict(X_test) > 0.5).astype("int32")

report_simple = classification_report(y_test, y_pred_simple, output_dict=True)
report_complex = classification_report(y_test, y_pred_complex, output_dict=True)

print(report_simple.keys())

def report_to_df(report):
    return pd.DataFrame(report).T.loc[["0.0", "1.0", "accuracy", "macro avg", "weighted avg"]]

df_report_simple = report_to_df(report_simple).reset_index().round(3)
df_report_complex = report_to_df(report_complex).reset_index().round(3)

plot_model(model=simple_model, show_shapes=True, show_layer_names=True, to_file='simple_model.png')

plot_model(model=complex_model, show_shapes=True, show_layer_names=True, to_file='complex_model.png')

app = dash.Dash(__name__)

#simple model
weights1, biases1 = simple_model.layers[0].get_weights()
weights2, biases2 = simple_model.layers[1].get_weights()

df_weights1 = pd.DataFrame(weights1, columns=[f'Neuron {i}' for i in range(weights1.shape[1])])
df_weights1['Feature'] = [f'Feature {i}' for i in range(weights1.shape[0])]

df_weights2 = pd.DataFrame(weights2.flatten(), columns=['Weight'])
df_weights2['Neuron'] = [f'Neuron {i}' for i in range(weights2.shape[0])]

#complex model
dense_layers_complex = []
for idx, layer in enumerate(complex_model.layers):
    if len(layer.get_weights()) > 0:
        dense_layers_complex.append((idx, layer.name))

#topology simple
with open("simple_model.png", "rb") as f:
    simple_image = base64.b64encode(f.read()).decode()

#topology complex
with open("complex_model.png", "rb") as f:
    complex_image = base64.b64encode(f.read()).decode()

app.layout = html.Div([
    html.H1("Simple Netz und Complex Netz vergleichen"),

    html.Div([
        html.H3("Performance der Netze vergleichen "),
        html.Div([
            html.Div([
                html.H4("Simple Model"),
                dash_table.DataTable(
                    data=df_report_simple.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df_report_simple.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center'},
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.H4("Complex Model"),
                dash_table.DataTable(
                    data=df_report_complex.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df_report_complex.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center'},
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        ])
    ]),

    html.Div([
        html.H3("Lernkurve"),
        dcc.Graph(
            id="learning-curve",
            figure={
                "data": [
                    go.Scatter(y=np.load("history_simple_val_acc.npy"), name="Simple Val Acc"),
                    go.Scatter(y=np.load("history_complex_val_acc.npy"), name="Complex Val Acc"),
                    go.Scatter(y=np.load("history_simple_acc.npy"), name="Simple Train Acc", line=dict(dash='dot')),
                    go.Scatter(y=np.load("history_complex_acc.npy"), name="Complex Train Acc", line=dict(dash='dot'))
                ],
                "layout": go.Layout(
                    title="Lernkurve",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy"
                )
            }
        )
    ]),

    html.Div([
      html.H3("Gewichte der Layer von Simple Model"),
      dcc.Dropdown(
          id='layer-select',
          options=[
              {'label': 'Layer 1 (Input to Hidden)', 'value': 'layer1'},
              {'label': 'Layer 2 (Hidden to Output)', 'value': 'layer2'},
          ],
          value='layer1'
      ),
      dcc.Graph(id='weight-graph')
    ]),

    html.Div([
      html.H3("Gewichte der Layer von Complex Model"),
      dcc.Dropdown(
        id='complex-layer-select',
        options=[{'label': f"{i} - {name}", 'value': i} for i, name in dense_layers_complex],
        value=dense_layers_complex[0][0]
      ),
      dcc.Graph(id='complex-weight-graph')
    ]),

    html.Div([
      html.H3("Topology von Simple Model"),
      html.Img(src=f"data:image/png;base64,{simple_image}", style={"width": "50%"})
    ]),

    html.Div([
      html.H3("Topology von Complex Model"),
      html.Img(src=f"data:image/png;base64,{complex_image}", style={"width": "50%"})
    ])

])

@app.callback(
    Output('weight-graph', 'figure'),
    Input('layer-select', 'value')
)
def update_simple_graph(layer):
    if layer == 'layer1':
        fig = px.imshow(weights1,
                        labels=dict(x="Neuron", y="Input Feature", color="Weight"),
                        x=[f'Neuron {i}' for i in range(weights1.shape[1])],
                        y=[f'Feature {i}' for i in range(weights1.shape[0])],
                        color_continuous_scale='RdBu',
                        origin='lower')
    else:
        fig = px.imshow(weights2,  # shape (16, 1)
                        labels=dict(x="Output", y="Hidden Neuron", color="Weight"),
                        x=["Output"],
                        y=[f'Neuron {i}' for i in range(weights2.shape[0])],
                        color_continuous_scale='RdBu',
                        origin='lower')
    return fig

@app.callback(
    Output('complex-weight-graph', 'figure'),
    Input('complex-layer-select', 'value')
)
def update_complex_graph(layer_idx):
    weights = complex_model.layers[layer_idx].get_weights()[0]
    fig = px.imshow(weights,
                    labels=dict(x="Output Neuron", y="Input Feature", color="Weight"),
                    x=[f'Neuron {i}' for i in range(weights.shape[1])],
                    y=[f'Feature {i}' for i in range(weights.shape[0])],
                    color_continuous_scale='RdBu',
                    origin='lower')
    fig.update_layout(title=f"Layer {layer_idx} ({complex_model.layers[layer_idx].name}) Gewichte HeatMap")
    return fig

if __name__ == '__main__':
    app.run(debug=True)