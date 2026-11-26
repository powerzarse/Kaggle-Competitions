# Backlog

Status: `todo` · `running` · `done` · `rejected`. Priorität: P1 (zuerst) … P3.
Erwarteter Gewinn × geringe Kosten = hohe Priorität. Ergebnisse → `learnings.md`.

| ID | P | Status | Hypothese | Umsetzung | Experiment |
|---|---|---|---|---|---|
| H01 | P1 | done | XGBoost ist ähnlich stark und divers genug fürs Blending | `model=xgb`, v1 | 02_xgb_v1 |
| H02 | P1 | running | CatBoost mit nativen Kategorien bringt Diversität | `model=catboost`, v1 | 03_catboost_v1 |
| H03 | P1 | todo | Rank-Blend aus LGBM+XGB+Cat schlägt bestes Einzelmodell um ≥0.0005 | `python -m src.blend` | – |
| H04 | P2 | todo | Feature-Set v1 hilft gegenüber Rohspalten (Ablation) | `features=base`, lgbm | – |
| H05 | P2 | todo | Numerische Spalten zusätzlich als Kategorie (Playground-Rundungsartefakte) | neues Feature-Set v2 | – |
| H06 | P2 | todo | Fold-internes Target-Encoding der Kategorien + Interaktionen (channel×device, auth×channel) | v3, fold-intern | – |
| H07 | P2 | todo | LGBM tunen (Optuna, 1 Fold, 30 Trials) – v. a. num_leaves, min_child_samples, Regularisierung | Tuning-Skript | – |
| H08 | P3 | todo | Originaldatensatz finden und als Zusatz-Trainingsdaten nutzen | Kaggle-Datasets suchen | – |
| H09 | P3 | todo | Seed-Averaging (3 Seeds) für bestes GBDT | params `random_state` | – |
