---
name: kaggle-blend
description: Kombiniert gespeicherte OOF-Vorhersagen mehrerer Experimente eines Kaggle-Competition-Ordners per Rank-Blending und Hill Climbing zu einer Ensemble-Submission. Verwenden bei "blend", "ensemble", "kombiniere die Modelle".
---

# Blending

`COMP` = Competition-Ordner mit `.context/`. Alle Befehle aus `COMP/`.

## Ablauf
1. Überblick inkl. Rang-Korrelationen:
   ```bash
   python -m src.status --offline
   ```
2. Blend über alle Einzelmodelle (frühere Blends ausgenommen) oder eine Auswahl:
   ```bash
   python -m src.blend
   python -m src.blend 01_lgbm_baseline 02_xgb_v1 03_catboost_v1
   ```
   Ausgabe: Einzel-AUCs, Rank-Mean, Hill-Climbing-Gewichte, Blend-AUC.
   Ergebnis → `models/blend_hc/`, `submissions/blend_hc.csv`, Zeile in `EXPERIMENTS.md`.

## Hinweise
- Rank-Blending ist für rangbasierte Metriken (AUC) richtig; `src.blend` rankt automatisch.
- Hill Climbing bewertet auf denselben OOFs, auf denen es optimiert → leicht optimistisch.
  Gewinn erst ab ≥ 2 × `min_delta_auc` (`rules.md`) als echt werten.
- Diversität schlägt Stärke: niedrige Rang-Korrelation (< 0.98) ist wertvoller als ein weiterer Seed.
