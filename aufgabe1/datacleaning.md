# Datenbereinigung (datacleaning.md)

## Einlesen der Datei

- Die Datei wurde mit dem Dateinamen `Aufgabe1.csv` eingelesen.
- Der verwendete Zeichensatz war `latin-1`.
- Als Spaltentrenner wurde das Komma `,` verwendet.

## Spaltennamen bereinigen

- Anführungszeichen (`'` und `"`) sowie führende und nachfolgende Leerzeichen wurden aus allen Spaltennamen entfernt.
- Die Umbenennung/Ersetzung von Leerzeichen oder Sonderzeichen in Spaltennamen wurde vorbereitet, aber bewusst auskommentiert, um die Originalnamen beizubehalten.

## Auswahl relevanter Spalten

- Es wurden nur folgende Spalten für die Analyse ausgewählt:
  - `'Known As'`
  - `'Full Name'`
  - `'Overall'`
  - `'Nationality'`
  - `'Age'`
  - `'Club Name'`
  - `'Wage(in Euro)'`

## Datentypkonvertierung

- Die Spalten `'Age'` und `'Wage(in Euro)'` wurden in numerische Datentypen umgewandelt (`to_numeric()`).
- Nicht konvertierbare Werte (z. B. Zeichenketten oder ungültige Einträge) wurden dabei zu `NaN`.

## Umgang mit fehlenden Werten

- Alle Zeilen mit fehlenden Werten (`NaN`) in den Spalten `'Age'` oder `'Wage(in Euro)'` wurden entfernt.

## Index zurücksetzen

- Der Index des DataFrames wurde nach dem Entfernen von Zeilen neu gesetzt (`reset_index(drop=True)`), um eine saubere fortlaufende Nummerierung zu gewährleisten.
