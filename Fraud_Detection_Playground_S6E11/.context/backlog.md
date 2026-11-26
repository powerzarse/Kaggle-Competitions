# Backlog

Status: `todo` · `running` · `done` · `rejected`. Priorität: P1 (zuerst) … P3.
Erwarteter Gewinn × geringe Kosten = hohe Priorität. Ergebnisse → `learnings.md`.

| ID | P | Status | Hypothese | Umsetzung | Experiment |
|---|---|---|---|---|---|
| H01 | P1 | done | XGBoost ist ähnlich stark und divers genug fürs Blending | `model=xgb`, v1 | 02_xgb_v1 |
| H02 | P1 | done | CatBoost mit nativen Kategorien bringt Diversität | `model=catboost`, v1 | 03_catboost_v1 |
| H03 | P1 | done | Rank-Blend aus LGBM+XGB+Cat schlägt bestes Einzelmodell um ≥0.0005 | `python -m src.blend` | blend_hc |
| H04 | P2 | todo | Feature-Set v1 hilft gegenüber Rohspalten (Ablation) | `features=base`, lgbm | – |
| H05 | P1 | rejected | Numerische Spalten zusätzlich als Kategorie (Playground-Rundungsartefakte) | neues Feature-Set v2 | 04_lgbm_v2 |
| H06 | P1 | rejected | Fold-internes Target-Encoding der Kategorien + Interaktionen (channel×device, auth×channel) | v3, fold-intern | 05_lgbm_v3 |
| H07 | P2 | todo | LGBM tunen (Optuna, 1 Fold, 30 Trials) – v. a. num_leaves, min_child_samples, Regularisierung | Tuning-Skript | – |
| H08 | P3 | todo | Originaldatensatz finden und als Zusatz-Trainingsdaten nutzen | Kaggle-Datasets suchen | – |
| H09 | P3 | todo | Seed-Averaging (3 Seeds) für bestes GBDT | params `random_state` | – |
| H10 | P2 | todo | CatBoost tiefer/länger (depth 8, lr 0.03) – Diversität beibehalten, Stärke steigern | params, teuer (~40 Min.) | – |
| H11 | P1 | todo | Weitere diverse Modellfamilie (z. B. MLP oder LogReg auf v1 + One-Hot) für den Blend; Flag velocity_is_zero + log-Transformationen helfen linearen Modellen | `model=logreg` zuerst (billig) | – |
| H12 | P3 | todo | Early Stopping auf dem Validierungs-Fold macht OOF leicht optimistisch → Runden per innerem Split bestimmen (sauberere CV, evtl. robustere Test-Preds) | Option in src/models.py, Vergleich auf bestem Modell | – |
