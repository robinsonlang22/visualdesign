# Datenbereinigung und Qualitätsanalyse

## Vorgehensweise:
1. Der Titanic‑Datensatz wurde per `pd.read_csv()` geladen.
2. Erste Explorationsschritte (head, info, describe, isnull) zeigten:
   - Fehlende Werte bei `Age` und `Embarked`
   - Spalten wie `PassengerId`, `Ticket`, `Name` und `Cabin` bieten wenig Mehrwert für die Klassifikation.
3. Maßnahmen:
   - Entfernen der Spalten: `PassengerId`, `Ticket`, `Name`, `Cabin`.
   - Imputation: `Age` wird durch den Median ersetzt, `Embarked` durch den Modus.
   - One-Hot-Encoding der kategorialen Variablen: `Sex` und `Embarked`.
4. Anschließend werden zwei Evaluierungsstrategien verwendet: 10‑fach Cross‑Validation und Bootstrapping (0.632‑Methode), um die Modelle zu bewerten.
5. Bewertet werden die Modelle mittels Accuracy, Precision, Recall, F1 Score und der Confusion-Matrix.
