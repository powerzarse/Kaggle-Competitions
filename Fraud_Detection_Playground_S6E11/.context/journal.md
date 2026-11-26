# Journal

Ein Eintrag pro Zyklus (append-only, neueste unten): Datum · Zyklus · was gelaufen ist · Ergebnis · Entscheidung.

- 2026-11-26 · Z0 · Setup, 01_lgbm_baseline (OOF 0.94631) eingereicht → LB 0.94630. 02_xgb_v1 und
  03_catboost_v1 gestartet. Agentensystem (`.agents/`) und `.context/` angelegt.
- 2026-11-26 · Z0 (Forts.) · 02_xgb_v1 0.94653 (neutral), 03_catboost_v1 0.94581 (schlechter, aber divers),
  blend_hc 0.94689 (5/5 Folds besser). Backlog um H10/H11 ergänzt. Submission blend_hc vorgeschlagen.
- 2026-11-26 · Z1 · blend_hc eingereicht (LB 0.94683). H05 04_lgbm_v2 0.93902 (rejected), H06 05_lgbm_v3 0.94605
  (rejected), H11 06_logreg_v4 0.94484 (neutral, divers 0.967) → blend_02 0.94698 (+0.00008, nicht eingereicht).
  Neu: H12, H13. Priorität auf Tuning (H07) und diverse Modelle (H13) verschoben.
