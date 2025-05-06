import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, Input, Output

# Preparing data for usage *******************************************

df = pd.read_csv('Aufgabe1.csv', sep=',', encoding='latin-1')
df.columns = (
    df.columns
    .str.strip("'\"")
    .str.strip()
    #.str.replace(' ', '_')
    #.str.replace(r'[^0-9a-zA-Z_]', '', regex=True)
)
#print(df.columns)
df = df[['Known As','Full Name','Overall','Nationality','Age','Club Name','Wage(in Euro)']]
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Wage(in Euro)'] = pd.to_numeric(df['Wage(in Euro)'], errors='coerce')
df = df.dropna(subset=['Age','Wage(in Euro)'])
df = df.reset_index(drop=True)
#print(df.head(5))
#df.to_csv('Aufgabe1_cleaned.csv', index=False)

# read cleaned data
df = pd.read_csv("Aufgabe1_cleaned.csv")

app = Dash(__name__)
app.title = "football_player_data_visualization"

# unique values
nationalities = df['Nationality'].dropna().unique()
clubs = df['Club Name'].dropna().unique()

# App Layout **************************************************************

app.layout = html.Div([
    html.H1("Football Player Data Visualization"),

    html.Div([
        html.Label("Age Range:"),
        dcc.RangeSlider(id='age-slider',
                        min=int(df['Age'].min()), max=int(df['Age'].max()),
                        step=1, value=[int(df['Age'].min()), int(df['Age'].max())],
                        marks={i: str(i) for i in range(int(df['Age'].min()), int(df['Age'].max())+1, 5)}),

        html.Label("Wage Range (€):"),
        dcc.RangeSlider(id='wage-slider',
                        min=int(df['Wage(in Euro)'].min()), max=int(df['Wage(in Euro)'].max()),
                        step=1000, value=[int(df['Wage(in Euro)'].min()), int(df['Wage(in Euro)'].max())],
                        marks={i: f'{i:,}' for i in range(
                        int(df['Wage(in Euro)'].min()), 
                        int(df['Wage(in Euro)'].max()), 
                        50000  )}),

        html.Label("Overall Range:"),
        dcc.RangeSlider(id='overall-slider',
                        min=int(df['Overall'].min()), max=int(df['Overall'].max()),
                        step=1, value=[int(df['Overall'].min()), int(df['Overall'].max())],
                        marks={i: f'{i:,}' for i in range(
                        int(df['Overall'].min()), 
                        int(df['Overall'].max()), 
                        10  )}),

        html.Label("Nationality:"),
        dcc.Dropdown(id='nationality-dropdown',
                     options=[{'label': nat, 'value': nat} for nat in sorted(nationalities)],
                     multi=True),

        html.Label("Clubs:"),
        dcc.Dropdown(id='club-dropdown',
                     options=[{'label': c, 'value': c} for c in sorted(clubs)],
                     multi=True),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

    html.Div([
        dcc.Graph(id='age-dist'),
        dcc.Graph(id='wage-dist'),
        dcc.Graph(id='overall-dist'),
        dcc.Graph(id='nationality-dist'),
        dcc.Graph(id='club-dist'),
        dcc.Graph(id='age-wage-scatter'),
        dcc.Graph(id='age-overall-scatter'),
    ], style={'width': '68%', 'display': 'inline-block'})
])

@app.callback(
    [Output('age-dist', 'figure'),
     Output('wage-dist', 'figure'),
     Output('overall-dist', 'figure'),
     Output('nationality-dist', 'figure'),
     Output('club-dist', 'figure'),
     Output('age-wage-scatter', 'figure'),
     Output('age-overall-scatter', 'figure')],
    [Input('age-slider', 'value'),
     Input('wage-slider', 'value'),
     Input('overall-slider', 'value'),
     Input('nationality-dropdown', 'value'),
     Input('club-dropdown', 'value')]
)
def update_graphs(age_range, wage_range, overall_range, nationalities, clubs):
    dff = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Wage(in Euro)'] >= wage_range[0]) & (df['Wage(in Euro)'] <= wage_range[1]) &
        (df['Overall'] >= overall_range[0]) & (df['Overall'] <= overall_range[1])
    ]

    if nationalities:
        dff = dff[dff['Nationality'].isin(nationalities)]
    if clubs:
        dff = dff[dff['Club Name'].isin(clubs)]

    return (
        px.box(dff, y="Age", points="all", title="age distribution"),
        px.box(dff, y="Wage(in Euro)", points="all", title="wage distribution"),
        px.box(dff, y="Overall", points="all", title="overall distribution"),
        px.histogram(dff, x="Nationality", title="nationality distribution"),
        px.histogram(dff, x="Club Name", title="club distribution"),
        px.scatter(dff, x="Age", y="Wage(in Euro)", title="age versus wage", opacity=0.5),
        px.scatter(dff, x="Age", y="Overall", title="age versus overall", opacity=0.5),
    )

if __name__ == '__main__':
    app.run(debug=True)