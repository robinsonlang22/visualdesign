# Datenbereinigung

## Einlesen der Datei

- Die Spalten CL, CM, CN ( fehlende Attribute ) wurden direkt von der originalen Datei gelöscht
- Die Datei wurde mit dem Dateinamen `Aufgabe1.csv` eingelesen
- Der verwendete Zeichensatz war `latin-1`
- Als Spaltentrenner wurde das Komma `,` verwendet
- Anführungszeichen sowie führende und nachfolgende Leerzeichen wurden aus allen Spaltennamen entfernt

## Auswahl relevanter Spalten

- Es wurden nur folgende Spalten für die Analyse ausgewählt:
  - `'Known As'`
  - `'Full Name'`
  - `'Overall'`
  - `'Nationality'`
  - `'Age'`
  - `'Club Name'`
  - `'Wage(in Euro)'`

## Daten prüfen und reinigen

- Die Spalten `'Age'` und `'Wage(in Euro)'` wurden in numerische Datentypen umgewandelt (`to_numeric()`)
- Nicht konvertierbare Werte wurden dabei zu `NaN`
- Alle Zeilen mit fehlenden Werten `NaN` in den Spalten `'Age'` oder `'Wage(in Euro)'` wurden entfernt

## Index zurücksetzen

- Der Index des DataFrames wurde nach dem Entfernen von Zeilen neu gesetzt, um eine saubere fortlaufende Nummerierung zu gewährleisten
