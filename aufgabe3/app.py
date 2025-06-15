import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Modelle und Evaluierungswerkzeuge aus scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#####################################
# 1. Datensatz laden und untersuchen
#####################################
df = pd.read_csv("titanic_cleaned.csv")
print("Erste 5 Zeilen des Datensatzes:")
print(df.head())
print("\nInformationen zum Datensatz:")
print(df.info())
print("\nStatistische Zusammenfassung:")
print(df.describe())
print("\nFehlende Werte pro Spalte:")
print(df.isnull().sum())

###########################################
# 2. Datenbereinigung und Dokumentation
###########################################
# Dokumentation des Datenreinigungsprozesses in "datacleaning.md"
with open("datacleaning.md", "w", encoding="utf-8") as f:
    f.write("# Datenbereinigung und Qualitätsanalyse\n\n")
    f.write("## Vorgehensweise:\n")
    f.write("1. Der Titanic‑Datensatz wurde per `pd.read_csv()` geladen.\n")
    f.write("2. Erste Explorationsschritte (head, info, describe, isnull) zeigten:\n")
    f.write("   - Fehlende Werte bei `Age` und `Embarked`\n")
    f.write("   - Spalten wie `PassengerId`, `Ticket`, `Name` und `Cabin` bieten wenig Mehrwert für die Klassifikation.\n")
    f.write("3. Maßnahmen:\n")
    f.write("   - Entfernen der Spalten: `PassengerId`, `Ticket`, `Name`, `Cabin`.\n")
    f.write("   - Imputation: `Age` wird durch den Median ersetzt, `Embarked` durch den Modus.\n")
    f.write("   - One-Hot-Encoding der kategorialen Variablen: `Sex` und `Embarked`.\n")
    f.write("4. Anschließend werden zwei Evaluierungsstrategien verwendet: 10‑fach Cross‑Validation und Bootstrapping (0.632‑Methode), um die Modelle zu bewerten.\n")
    f.write("5. Bewertet werden die Modelle mittels Accuracy, Precision, Recall, F1 Score und der Confusion-Matrix.\n")

# Erstellen einer bereinigten Kopie des Datensatzes
df_clean = df.copy()

# Entferne irrelevante Spalten
cols_to_drop = ['PassengerId', 'Ticket', 'Name', 'Cabin']
df_clean.drop(columns=cols_to_drop, inplace=True)

# Fehlende Werte in 'Age' mit Median ersetzen
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)

# Fehlende Werte in 'Embarked' mit dem häufigsten Wert ersetzen
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)

# Kategoriale Variablen in numerische umwandeln (One-Hot-Encoding)
df_clean = pd.get_dummies(df_clean, columns=['Sex', 'Embarked'], drop_first=True)

print("\nBereinigter Datensatz:")
print(df_clean.head())

#####################################
# 3. Features und Zielvariable
#####################################
X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']

##############################################
# 4. Definition der Evaluierungsfunktionen
##############################################
def evaluate_cv(model, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    conf_matrix_total = np.zeros((2, 2))
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred))
        rec_list.append(recall_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        conf_matrix_total += confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': np.mean(acc_list),
        'precision': np.mean(prec_list),
        'recall': np.mean(rec_list),
        'f1': np.mean(f1_list),
        'confusion_matrix': conf_matrix_total
    }

def evaluate_bootstrap(model, X, y, n_iterations=100):
    n = len(X)
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    conf_matrix_total = np.zeros((2, 2))
    
    for i in range(n_iterations):
        bootstrap_idx = np.random.choice(np.arange(n), size=n, replace=True)
        oob_idx = np.setdiff1d(np.arange(n), bootstrap_idx)
        if len(oob_idx) == 0:
            continue  # Falls keine Out-of-Bag-Beispiele vorhanden
        X_train = X.iloc[bootstrap_idx]
        y_train = y.iloc[bootstrap_idx]
        X_test = X.iloc[oob_idx]
        y_test = y.iloc[oob_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred))
        rec_list.append(recall_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        conf_matrix_total += confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': np.mean(acc_list),
        'precision': np.mean(prec_list),
        'recall': np.mean(rec_list),
        'f1': np.mean(f1_list),
        'confusion_matrix': conf_matrix_total
    }

#####################################
# 5. Modelle definieren
#####################################
log_reg = LogisticRegression(solver='liblinear', random_state=42)
dtree   = DecisionTreeClassifier(random_state=42)
knn     = KNeighborsClassifier(n_neighbors=3)

models = {
    'Logistische Regression': log_reg,
    'Decision Tree': dtree,
    'KNN (k=3)': knn
}

#####################################
# 6. Evaluation (Cross-Validation & Bootstrapping)
#####################################
cv_results = {}
bootstrap_results = {}

for name, model in models.items():
    cv_results[name] = evaluate_cv(model, X, y)
    bootstrap_results[name] = evaluate_bootstrap(model, X, y, n_iterations=100)

# Ergebnisse ausgeben
print("\n--- 10-fache Cross-Validation Ergebnisse ---")
for name, res in cv_results.items():
    print(f"{name}:\n  Accuracy: {res['accuracy']:.3f}\n  Precision: {res['precision']:.3f}\n  Recall: {res['recall']:.3f}\n  F1 Score: {res['f1']:.3f}\n  Confusion Matrix:\n{res['confusion_matrix']}\n")

print("\n--- Bootstrapping (0.632-Methode) Ergebnisse ---")
for name, res in bootstrap_results.items():
    print(f"{name}:\n  Accuracy: {res['accuracy']:.3f}\n  Precision: {res['precision']:.3f}\n  Recall: {res['recall']:.3f}\n  F1 Score: {res['f1']:.3f}\n  Confusion Matrix:\n{res['confusion_matrix']}\n")

#####################################
# 7. Visualisierung der Metriken (Cross-Validation)
#####################################
cv_df = pd.DataFrame({
    'Classifier': list(cv_results.keys()),
    'Accuracy': [cv_results[c]['accuracy'] for c in cv_results],
    'Precision': [cv_results[c]['precision'] for c in cv_results],
    'Recall': [cv_results[c]['recall'] for c in cv_results],
    'F1 Score': [cv_results[c]['f1'] for c in cv_results]
})

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(cv_df))

plt.bar(index, cv_df['Accuracy'], bar_width, label='Accuracy')
plt.bar(index + bar_width, cv_df['Precision'], bar_width, label='Precision')
plt.bar(index + 2 * bar_width, cv_df['Recall'], bar_width, label='Recall')
plt.bar(index + 3 * bar_width, cv_df['F1 Score'], bar_width, label='F1 Score')

plt.xlabel('Klassifizierer')
plt.ylabel('Score')
plt.title('Modellperformance (10-fache Cross-Validation)')
plt.xticks(index + 1.5 * bar_width, cv_df['Classifier'])
plt.legend()
plt.tight_layout()
plt.show()

#####################################
# 8. Darstellung der Confusion-Matrizen (CV)
#####################################
for name, result in cv_results.items():
    plt.figure(figsize=(4,3))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='g', cmap='Blues')
    plt.title(f"Confusion Matrix: {name} (CV)")
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsächlich")
    plt.show()

#####################################
# 9. Visualisierung des Decision Trees
#####################################
# Den gesamten Datensatz verwenden für die Baumvisualisierung
dtree.fit(X, y)
plt.figure(figsize=(20,10))
plot_tree(dtree, feature_names=X.columns, class_names=['Nicht überlebt', 'Überlebt'], filled=True, rounded=True)
plt.title("Visualisierter Decision Tree")
plt.show()