import pandas as pd

df = pd.read_csv('wein.csv', sep=',', encoding='utf-8')
# print(df.dtypes)
columns_need_clean = ['Alkohol','Apfelsaeure','Asche','Aschen_Alkanitaet','Alle_Phenole',
                      'Flavanoide','Proanthocyanide','Farbintensitaet','Farbwert','Proteinwert']

def date_to_number(column):
    column = column.astype(str).str.rstrip('.')
    return pd.to_numeric(column, errors='coerce').round(2)

for column in columns_need_clean:
    df[column] = date_to_number(df[column])

# print(df.head(5))
# print(df.dtypes)
df.reset_index(drop=True)
df.to_csv('wein_cleaned.csv', index=False)

