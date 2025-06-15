import pandas as pd

df = pd.read_csv('titanic.csv')

median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
df.loc[df['Age'] == 0, 'Age'] = median_age

median_fare = df['Fare'].median()
df['Fare'] = df['Fare'].fillna(median_fare)
df.loc[df['Fare'] == 0, 'Fare'] = median_fare

mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

df.reset_index(drop=True)
df.to_csv('titanic_cleaned.csv', index=False)