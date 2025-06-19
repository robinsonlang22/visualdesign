import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

df = pd.read_csv('pulsar_data.csv')

df.info()

X = df.drop(columns=['target_class'])
y = df['target_class']

numeric_features = X.select_dtypes(include=['float64']).columns


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

x_processed = preprocessor.fit_transform(X)
x_processed = pd.DataFrame(x_processed, columns=numeric_features)

df_cleaned = pd.concat([x_processed, y.reset_index(drop=True)], axis=1)

df_cleaned.to_csv('cleaned_pulsar_data.csv', index=False)
