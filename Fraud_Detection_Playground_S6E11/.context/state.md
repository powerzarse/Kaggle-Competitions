# State  (aktualisiert: 2026-11-26 14:00 UTC, Zyklus Z1 abgeschlossen)
- Deadline: 2026-11-30 23:59 UTC (noch 4 Tage) · Submissions heute übrig: 8
- Bestes Einzelmodell: 02_xgb_v1 CV 0.94653 (LB –)
- Bester Blend: blend_02 CV 0.94698 (LGBM .286 / XGB .429 / Cat .143 / LogReg .143, LB –, nicht eingereicht:
  +0.00008 zu blend_hc < min_delta)
- Beste eingereichte CV: 0.94689 (blend_hc, LB 0.94683)
- Laufende Jobs: keine
- Nächster Schritt (Z2): H07 LGBM/XGB-Tuning (Optuna auf 1 Fold), H13 LogReg v5 mit Bin×Bin-Interaktionen,
  danach H10 (CatBoost größer) bzw. H09 (Seeds) → neuer Blend
- Offene Fragen an den Menschen: keine
