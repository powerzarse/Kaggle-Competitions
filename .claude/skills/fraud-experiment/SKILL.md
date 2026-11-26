---
name: fraud-experiment
description: Legt ein neues Modell-/Feature-Experiment für die Kaggle-Competition playground-series-s6e11 (Credit Card Fraud) an, führt die 5-Fold-CV aus und protokolliert das Ergebnis in EXPERIMENTS.md. Verwenden bei "neues Experiment", "probier XGBoost/CatBoost/neue Features", "trainiere Modell" o. Ä. im Ordner Fraud_Detection_Playground_S6E11.
---

# Neues Experiment (S6E11)

Arbeitsordner: `Fraud_Detection_Playground_S6E11/`. Voraussetzung: Daten da (Skill `fraud-data`).

## Regeln (nicht verhandelbar)
1. **Immer dieselben Folds**: `src.cv.get_folds()` (aus `data/folds.csv`). Nie eigene Splits bauen –
   sonst sind OOF-Vorhersagen nicht vergleichbar und das Blending leakt.
2. **Ein Experiment = ein Ordner** `experiments/NN_<kurzname>/` mit fortlaufender Nummer.
3. **Eine Änderung pro Experiment** gegenüber dem besten bisherigen Stand, Hypothese in `note`.
4. Ergebnis wird über `save_experiment()` gespeichert → `models/<name>/`, `submissions/<name>.csv`,
   Zeile in `EXPERIMENTS.md`. Nichts davon von Hand anlegen.
5. Bewertet wird die **OOF-AUC** (nicht der Mittelwert der Folds, nicht der Public LB).

## Ablauf
1. Nächste Nummer ermitteln: `ls experiments/`
2. Vorlage kopieren:
   ```bash
   cp -r experiments/_template experiments/NN_<kurzname>
   ```
3. In `run.py` nur `CONFIG` anpassen:
   - `features`: Schlüssel aus `src/features.py::FEATURE_SETS`. Neue Features → **neue Funktion**
     (`v2`, `v3`, …) dort anlegen und registrieren, bestehende Sets nicht verändern
     (alte Experimente müssen reproduzierbar bleiben).
   - `model`: `lgbm` | `xgb` | `catboost` | `logreg` (`src/models.py`). Neues Modell → Factory
     mit Signatur `fn(params=None) -> fit_predict(X_tr, y_tr, X_va, y_va, X_te)` ergänzen und in `MODELS` eintragen.
   - `params`: überschreibt Defaults; `note`: Hypothese in einem Satz.
4. Fehlende Pakete bei Bedarf installieren (`pip install xgboost catboost`).
5. Ausführen (lange Läufe im Hintergrund, Log mitschreiben):
   ```bash
   python experiments/NN_<kurzname>/run.py 2>&1 | tee experiments/NN_<kurzname>/run.log
   ```
   Zum schnellen Debuggen erst mit `learning_rate` hoch (z. B. 0.1) laufen lassen.
6. Optional ein `README.md` im Experimentordner: Hypothese, Ergebnis, Fazit (2–5 Zeilen).
7. Dem Nutzer berichten: OOF-AUC, Fold-Std, Delta zum bisher besten Eintrag in `EXPERIMENTS.md`.

## Ideen-Backlog (Playground-typisch)
- GBDT-Varianten: LGBM/XGB/CatBoost mit getunten Params, mehrere Seeds mitteln.
- Kategoriale als Target-Encoding (fold-intern!) bzw. Count-Encoding; Interaktionen der Risiko-Flags.
- Numerische Spalten zusätzlich als Kategorie (Playground-Daten haben oft „Rundungsartefakte“).
- Originaldatensatz als zusätzliche Trainingszeilen (nur im Train-Teil der Folds).
- Optuna-Tuning auf 1–2 Folds, dann volle CV.
