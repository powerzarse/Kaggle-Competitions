# Experiment-Log

| Datum | Experiment | Features | Modell | CV AUC (OOF) | Fold-Std | Public LB | Notiz |
|---|---|---|---|---|---|---|---|
| 2026-11-26 | 01_lgbm_baseline | v1 | lgbm | 0.94631 | 0.00333 | 0.94630 | LGBM Default-Params, Feature-Set v1 |
| 2026-11-26 | 02_xgb_v1 | v1 | xgb | 0.94653 | 0.00328 |  | xgb Default-Params, Feature-Set v1 |
| 2026-11-26 | 03_catboost_v1 | v1 | catboost | 0.94581 | 0.00315 |  | catboost Default-Params, Feature-Set v1 |
| 2026-11-26 | blend_hc | - | rank-blend | 0.94689 | 0.00325 | 0.94683 | 3 Modelle |
| 2026-11-26 | 04_lgbm_v2 | v2 | lgbm | 0.93902 | 0.00368 |  | H05: numerische Spalten als Kategorie + Count-Encoding |
| 2026-11-26 | 05_lgbm_v3 | v3 | lgbm_te | 0.94605 | 0.00317 |  | H06: fold-internes Target-Encoding + 3 Interaktionen |
| 2026-11-26 | 06_logreg_v4 | v4 | logreg | 0.94484 | 0.00308 |  | H11: LogReg auf v4 als diverses Blend-Mitglied |
| 2026-11-26 | blend_02 | - | rank-blend | 0.94698 | 0.00323 |  | 4 Modelle |
| 2026-11-27 | 07_lgbm_v6_l4 | v6 | lgbm_vte | 0.95019 | 0.00350 |  | H14: Wert-/Ziffern-TE + flache Bäume (L4) |
| 2026-11-27 | 08_lgbm_v6_l2 | v6 | lgbm_vte | 0.95004 | 0.00343 |  | H14: Wert-/Ziffern-TE + flache Bäume (L2) |
| 2026-11-27 | 09_xgb_v6_d2 | v6 | xgb_vte | 0.95016 | 0.00333 |  | H15a: XGB depth 2 auf v6-TE (Blend-Partner zu 07) |
| 2026-11-27 | 10_glm_v6 | v6 | glm_vte | 0.95010 | 0.00335 |  | H15b: GLM auf logit(v6-TE) + Flags + One-Hot, C=1 (Blend-Partner zu 07) |
| 2026-11-27 | blend_03 | - | rank-blend | 0.95061 | 0.00342 |  | 2 Modelle |
| 2026-11-27 | 11_glmboost_v6 | v6 | glmboost_vte | 0.95059 | 0.00328 |  | H16: GLM-Boost – LGBM L4 mit init_score = GLM-Logit (innere OOF, Variante A) auf v6-TE |
