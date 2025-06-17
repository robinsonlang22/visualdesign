# Datenbereinigung und Qualitätsanalyse

## Vorgehensweise:

1. Erste Explorationsschritte (head, info, describe, isnull) zeigten:

   - Fehlende Werte bei `Age` und `Embarked`
   - Nullen Werte bei `Fare`
   - Spalten wie `PassengerId`, `Ticket`, `Name` und `Cabin` bieten wenig Mehrwert für die Klassifikation
2. Maßnahmen:

   - Entfernen der Spalten: `PassengerId`, `Ticket`, `Name`, `Cabin`
   - Imputation:
     - Fehlende Wrete bei `Age` wird durch den Median ersetzt
     - Fehlende Werte bei `Embarked` wird durch den häufigsten Werte ersetzt
     - Nullen Werte bei `Fare` wird durch den Median ersetzt
   - One-Hot-Encoding der kategorialen Variablen `Sex` und `Embarked`
