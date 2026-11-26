---
name: fraud-data
description: Lädt und prüft die Daten der Kaggle-Competition playground-series-s6e11 (Predicting Credit Card Fraud) im Ordner Fraud_Detection_Playground_S6E11. Verwenden, wenn data/ leer ist, die Daten neu geladen werden sollen, oder vor dem ersten Experiment in einer frischen Session.
---

# Daten für S6E11 – Credit Card Fraud

Arbeitsordner: `Fraud_Detection_Playground_S6E11/` (alle Befehle von dort aus).

## 1. Prüfen, ob die Daten schon da sind
```bash
ls -lh data/
```
Erwartet: `train.csv` (699 960 Zeilen), `test.csv` (299 984 Zeilen), `sample_submission.csv`.
`*.parquet` und `folds.csv` sind Caches, die `src/` selbst anlegt.

## 2. Herunterladen (falls etwas fehlt)
```bash
python -m src.kaggle_api download
```
- Nutzt die Kaggle-REST-API direkt (nicht die `kaggle`-CLI). In der Claude-Cloud-Umgebung
  injiziert der Proxy die Zugangsdaten – kein Key nötig. Lokal: `~/.kaggle/kaggle.json`
  oder `~/.kaggle/access_token`.
- Bei `403`: Competition-Regeln auf kaggle.com akzeptieren.
- Daten **niemals committen** (`data/*` steht in `.gitignore`).

## 3. Plausibilitätscheck
```bash
python -c "from src.data import load_data; tr, te = load_data(); print(tr.shape, te.shape, tr.is_fraud.mean())"
```
Erwartet: `(699960, 26) (299984, 25)` und Fraud-Rate ≈ `0.0167`.

## Fakten zum Datensatz
- Metrik: **ROC AUC** → nur das Ranking zählt; Vorhersagen müssen Wahrscheinlichkeiten/Scores sein.
- Ziel `is_fraud` stark unbalanciert (1,67 %), daher Stratified K-Fold.
- 13 numerische, 6 boolesche, 5 kategoriale Spalten (siehe `src/config.py`), keine NaNs.
- Synthetischer Playground-Datensatz: Originaldaten (falls auffindbar) als Zusatz-Trainingsdaten testen.
