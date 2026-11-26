# Learnings

Kurze, überprüfte Erkenntnisse. Jede Zeile: Was wurde geändert → Δ OOF-AUC → Urteil → Warum (Vermutung).
Neueste oben. Der Researcher leitet daraus neue Hypothesen ab; nichts doppelt testen, was hier steht.

## Gesichert
- [2026-11-26] 05_lgbm_v3 (H06): fold-internes Target-Encoding (m-Estimate m=20, innere 5-fold-OOF auf X_tr)
  der 5 CAT_COLS + 3 Interaktionen channel×device_type / auth_method×channel / merchant_category×channel
  (40/25/60 Level), Kategorien bleiben `category` → Δ −0.00026 (1/5 Folds) → schlechter (knapp über min_delta).
  Code geprüft, leakage-frei. Bei 5–12 Leveln je Kategorie und ≤ 60 Interaktions-Leveln findet LGBM diese
  Kombinationen über native Kategorie-Splits schon selbst. Das TE ist redundant, und das rauschigere OOF-TE im
  Train passt nicht ganz zum vollen Mapping in Val/Test. Rang-Korr. 0.989 zu 01. Blend-Zuschlag zu blend_hc
  höchstens +0.00002 (w 0.1, In-Sample), 05 statt 01 im blend_hc 0.94685 (−0.00004) → kein Blend-Kandidat.
  TE niedrigkardinaler Kategorien nicht erneut für GBDTs testen. Für lineare Modelle/NN (H11) kann es weiter nützen.
- [2026-11-26] 04_lgbm_v2 (H05): v1 + 6 numerische Spalten als `category` (_cat) + 6 Count-Encodings
  (_cnt, über train+test, label-frei) → Δ −0.00729 (0/5 Folds) → schlechter. Ursache sind die _cat-Spalten:
  Reviewer-Probe Fold 0, lr 0.1 (v1 0.94279): +_cnt 0.94251 (−0.0003, Rauschen), +alle _cat 0.93334
  (−0.0095; merchant_risk_score_cat/velocity_score_cat übernehmen Top-Gain = Overfit auf 674/959 Level),
  +nur niedrigkardinale _cat (4 Spalten, 12–64 Level) 0.94179 (−0.0010) → „Zahl als Kategorie“ schadet
  LGBM generell, Count-Encoding bringt nichts; nicht erneut testen. Rang-Korr. 0.92, aber Blend-Zuschlag
  zu blend_hc nur +0.0001 (w 0.1, In-Sample) < min_delta → kein Blend-Kandidat.
- [2026-11-26] blend_hc eingereicht: LB 0.94683 (CV 0.94689) → Blend-Gewinn überträgt sich auf den LB (+0.00053 ggü. Baseline).
- [2026-11-26] blend_hc (H03): Rank-Blend LGBM 0.25 / XGB 0.5 / Cat 0.25 → OOF 0.94689, Δ +0.00036 zu
  02_xgb_v1 (5/5 Folds besser). Ungewichteter Rank-Mean (nicht optimiert) 0.94688 → Gewinn ist
  kein Hill-Climbing-Overfit. Knapp unter 2×min_delta, aber konsistent → Blending lohnt sich.
- [2026-11-26] 03_catboost_v1 (H02): CatBoost Defaults (lr 0.05, depth 6), v1 → OOF 0.94581,
  Δ −0.00072 zu XGB (0/5 Folds) → schlechter als Einzelmodell, aber diversestes Modell
  (Rang-Korr. 0.978) → im Blend behalten. Laufzeit ~25 Min. (CPU) – teuerstes Modell.
- [2026-11-26] 02_xgb_v1 (H01): XGBoost statt LGBM, v1 → Δ +0.00022 (3/5 Folds besser, 1 gleich) → neutral.
  Rang-Korrelation zu LGBM 0.988 (< 0.99) → als Blend-Partner behalten.
- [2026-11-26] **CV ≈ LB**: 01_lgbm_baseline OOF 0.94631 vs. Public LB 0.94630 → der CV ist
  vertrauenswürdig, auf CV optimieren.
- [2026-11-26] LGBM (lr 0.03, 63 Leaves) auf v1: OOF 0.94631, Fold-Std 0.0033. Referenzpunkt.

## Offen / Hinweise
- Fold-Std (0.0033) ist groß gegenüber typischen Gewinnen → Vergleiche immer gepaart auf
  denselben Folds (OOF-AUC), nicht über Fold-Mittelwerte.
- velocity_score = 0.0 kommt 69 426× vor (train+test; 48 499× in train ≈ 6.9 %) mit Fraud-Rate 0.04 %
  (Basis 1.67 %) → Häufungs-/Imputationswert, fast sicher „kein Fraud“. Numerische Splits fangen ihn
  bereits (velocity_score ist Top-5-Gain in v1); ggf. als Flag `velocity_is_zero` für lineare/NN-Modelle (H11).
- Methodik (alle Experimente gleich betroffen): `src/models.py` nutzt den Validierungs-Fold auch fürs
  Early Stopping → OOF leicht optimistisch. CV ≈ LB zeigt, dass der Effekt klein ist; Vergleiche bleiben fair.
