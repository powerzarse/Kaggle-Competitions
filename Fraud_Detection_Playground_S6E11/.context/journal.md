# Journal

Ein Eintrag pro Zyklus (append-only, neueste unten): Datum · Zyklus · was gelaufen ist · Ergebnis · Entscheidung.

- 2026-11-26 · Z0 · Setup, 01_lgbm_baseline (OOF 0.94631) eingereicht → LB 0.94630. 02_xgb_v1 und
  03_catboost_v1 gestartet. Agentensystem (`.agents/`) und `.context/` angelegt.
- 2026-11-26 · Z0 (Forts.) · 02_xgb_v1 0.94653 (neutral), 03_catboost_v1 0.94581 (schlechter, aber divers),
  blend_hc 0.94689 (5/5 Folds besser). Backlog um H10/H11 ergänzt. Submission blend_hc vorgeschlagen.
- 2026-11-26 · Z1 · blend_hc eingereicht (LB 0.94683). H05 04_lgbm_v2 0.93902 (rejected), H06 05_lgbm_v3 0.94605
  (rejected), H11 06_logreg_v4 0.94484 (neutral, divers 0.967) → blend_02 0.94698 (+0.00008, nicht eingereicht).
  Neu: H12, H13. Priorität auf Tuning (H07) und diverse Modelle (H13) verschoben.
- 2026-11-27 · Z2 · Lücken-Check: LB-Median 0.9492, #50 0.9577, #1 0.9705 – wir 0.94683 (Rang 2346/3338).
  System-Fix: src.status zeigt LB-Position, Orchestrator macht Lücken-Check, Researcher sichtet zuerst
  öffentliche Notebooks/Datasets. Researcher mit Fokus „Lücke“ gestartet.
- 2026-11-27 · Z2 · Researcher: Lücke = Wert-/Ziffern-TE + flache Bäume (Generator „Label zuerst“), Originaldaten nutzlos.
  H14 07_lgbm_v6_l4 0.95019 (neue Basis, Leakage-Gegenproben ok), 08 L2 0.95004; H15 09_xgb_v6_d2 0.95016 (nicht divers),
  10_glm_v6 0.95010 (divers); H16 11_glmboost_v6 0.95059 (bestes Einzelmodell). blend_03 0.95061, blend_04 0.95072.
  Submission blend_04 vorgeschlagen (Freigabe offen). System-Fix: LB-Lücken-Check.
