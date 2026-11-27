# 09_xgb_v6_d2 – Wert-/Ziffern-Target-Encoding + XGBoost Tiefe 2

**Hypothese (H15a):** Ein zweites flaches Baummodell (XGB statt LGBM) auf denselben v6-TE-Features ist ähnlich stark wie 07,
macht aber leicht andere Fehler und ergänzt 07 deshalb im Blend.

**Änderung:**
- `src/models.py`: Den TE-Schritt aus `lgbm_vte` habe ich als Helfer `vte_frames` herausgezogen (alle `__key`-Spalten per
  `target_encode_keys` encodieren, Schlüssel entfernen, `<c>_te` anhängen). `lgbm_vte` ruft ihn auf und liefert identische
  Ergebnisse: Auf Fold 0 sind die TE-Matrizen von Train, Val und Test per `assert_frame_equal` identisch, und der Smoke
  mit lr 0.1 ergibt exakt 0.94867 wie in 07.
- Neue Factory `xgb_vte`: `vte_frames` (m=20, 5 innere Folds, Seed 42, also gleich wie 07), danach XGBoost mit hist,
  max_depth 2, lr 0.04, subsample 0.8, colsample_bytree 0.5, min_child_weight 5, reg_lambda 1, n_estimators 20000 und
  Early Stopping 300. Die 5 Kategorien gehen wie in `xgb` nativ ein (enable_categorical). Insgesamt sind es 84 Spalten
  (30 roh + 54 TE). Feature-Set `v6` ist unverändert.

**Leakage-Schutz:** wie 07. Das TE läuft komplett fold-intern in `fit_predict`. Val dient nur dem Early Stopping.

**Smoke-Test (Fold 0, lr 0.1):** 0.94908 nach 551 Runden (LGBM 07 mit lr 0.1: 0.94867).

**Ergebnis:** OOF-AUC **0.95016** (Fold-Mittel 0.95023 ± 0.00333).
Folds [0.94920, 0.95122, 0.95493, 0.95109, 0.94470] gegenüber 07 [0.94926, 0.95115, 0.95508, 0.95142, 0.94435]
→ **Δ −0.00003 zu 07** (2/5 Folds besser). Damit ist 09 gleich stark wie 07.
Best-Iteration je Fold: 1344 / 2117 / 1749 / 1132 / 1776.
Rang-Korrelation 0.995 zu 07, 0.993 zu 08 und 0.982 zu 10.
Blend-Probe (Rank, In-Sample): 07 + 09 bringt höchstens +0.00011 (w 0.5) bzw. +0.00009 bei w 0.3 (5/5 Folds), also wenig,
weil 09 zu ähnlich zu 07 ist. Im Dreier-Blend mit 10 siehe `experiments/10_glm_v6/README.md`.

**Laufzeit:** 356 s für den vollen 5-Fold-Lauf (4 CPU). Keine Warnungen im Log.
