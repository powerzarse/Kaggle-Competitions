---
name: fraud-blend
description: Kombiniert die gespeicherten OOF-Vorhersagen mehrerer Experimente der Kaggle-Competition playground-series-s6e11 (Credit Card Fraud) per Rank-Blending und Hill Climbing zu einer Ensemble-Submission. Verwenden bei "blend", "ensemble", "kombiniere die Modelle" im Ordner Fraud_Detection_Playground_S6E11.
---

# Blending (S6E11)

Arbeitsordner: `Fraud_Detection_Playground_S6E11/`.

## Ablauf
1. Verfügbare Experimente ansehen:
   ```bash
   python -c "from src.tracking import available_experiments as a; print(a())"
   ```
2. Blend über alle (außer früheren Blends) oder eine Auswahl:
   ```bash
   python -m src.blend
   python -m src.blend 01_lgbm_baseline 02_xgb_v1 03_cat_v1
   ```
   Ausgabe: Einzel-AUCs, Rank-Mean, Hill-Climbing-Gewichte, Blend-AUC.
   Ergebnis landet als `blend_hc` in `models/`, `submissions/blend_hc.csv` und `EXPERIMENTS.md`.

## Hinweise
- AUC ist rang-basiert → Ranks statt Wahrscheinlichkeiten mitteln (macht `src.blend` automatisch).
- Hill Climbing optimiert auf denselben OOF, die es bewertet → leicht optimistisch.
  Gewinn < ~0.0002 gegenüber dem besten Einzelmodell eher als Rauschen behandeln.
- Diversität schlägt Stärke: ein etwas schwächeres, anders gebautes Modell (CatBoost, LogReg,
  andere Features) bringt oft mehr als der fünfte LGBM-Seed.
- Einen Blend nur dann einreichen, wenn er die beste Einzel-OOF klar schlägt (Skill `fraud-submit`).
