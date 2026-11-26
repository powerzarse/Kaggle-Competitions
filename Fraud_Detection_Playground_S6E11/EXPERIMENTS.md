# Experiment-Log

| Datum | Experiment | Features | Modell | CV AUC (OOF) | Fold-Std | Public LB | Notiz |
|---|---|---|---|---|---|---|---|
| 2026-11-26 | 01_lgbm_baseline | v1 | lgbm | 0.94631 | 0.00333 | 0.94630 | LGBM Default-Params, Feature-Set v1 |
| 2026-11-26 | 02_xgb_v1 | v1 | xgb | 0.94653 | 0.00328 |  | xgb Default-Params, Feature-Set v1 |
| 2026-11-26 | 03_catboost_v1 | v1 | catboost | 0.94581 | 0.00315 |  | catboost Default-Params, Feature-Set v1 |
| 2026-11-26 | blend_hc | - | rank-blend | 0.94689 | 0.00325 | 0.94683 | 3 Modelle |
| 2026-11-26 | 04_lgbm_v2 | v2 | lgbm | 0.93902 | 0.00368 |  | H05: numerische Spalten als Kategorie + Count-Encoding |
| 2026-11-26 | 05_lgbm_v3 | v3 | lgbm_te | 0.94605 | 0.00317 |  | H06: fold-internes Target-Encoding + 3 Interaktionen |
