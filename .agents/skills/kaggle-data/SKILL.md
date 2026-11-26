---
name: kaggle-data
description: Lädt und prüft die Daten einer Kaggle-Competition in deren Projektordner (z. B. Fraud_Detection_Playground_S6E11). Verwenden, wenn data/ leer ist, Daten neu geladen werden sollen, oder vor dem ersten Experiment in einer frischen Session.
---

# Daten laden & prüfen

`COMP` = Competition-Ordner mit `.context/` (`ls -d */.context`). Alle Befehle aus `COMP/`.
Erwartete Zeilenzahlen, Spalten und Zielrate stehen in `COMP/.context/competition.md`.

## 1. Vorhanden?
```bash
ls -lh data/
```
`*.parquet` und `folds.csv` sind Caches, die `src/` selbst erzeugt.

## 2. Herunterladen
```bash
python -m src.kaggle_api download
```
- Kaggle-RPC-API direkt (keine `kaggle`-CLI nötig). In der Claude-Cloud-Umgebung injiziert der
  Proxy die Zugangsdaten. Lokal: `~/.kaggle/access_token` oder `~/.kaggle/kaggle.json`.
- `401/403`: Regeln der Competition auf kaggle.com akzeptieren / Token prüfen.
- Daten **nie committen** (`data/*` ist in `.gitignore`).

## 3. Plausibilitätscheck
```bash
python -c "from src.data import load_data; from src.config import TARGET; tr, te = load_data(); print(tr.shape, te.shape, tr[TARGET].mean())"
```
Mit `competition.md` vergleichen. Abweichungen → in `competition.md` korrigieren und dem Orchestrator melden.
