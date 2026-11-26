# Learnings

Kurze, überprüfte Erkenntnisse. Jede Zeile: Was wurde geändert → Δ OOF-AUC → Urteil → Warum (Vermutung).
Neueste oben. Der Researcher leitet daraus neue Hypothesen ab; nichts doppelt testen, was hier steht.

## Gesichert
- [2026-11-26] blend_02 (Z1-Blend): Rank-Blend + Hill Climbing über 01/02/03/06 (keine rejected; max. Rang-Korr. 0.989 < 0.995)
  → Gewichte LGBM 0.286 / XGB 0.429 / Cat 0.143 / LogReg 0.143 → OOF 0.94698. Δ +0.00008 zu blend_hc (5/5 Folds,
  +0.00005…+0.00014), Δ +0.00044 zu 02_xgb_v1 (5/5 Folds) = erstmals ≥ 2×min_delta ggü. bestem Einzelmodell.
  Genestet (HC auf 4 Folds, 5. geprüft) +0.00005 zu blend_hc (5/5 Folds, Gewichte stabil: LogReg 0.14–0.20);
  ungewichteter Rank-Mean der 4 nur 0.94691 (+0.00002, 2/5) → LogReg braucht kleines Gewicht. Gewinn echt, aber
  < min_delta ggü. eingereichtem blend_hc (Schwelle 0.94709) → laut Submit-Policy nicht einreichen.
  Varianten (nur Scratch, nicht gespeichert): +05 → ±0.00000 (05 bringt nichts); +04 → 0.94707 (+0.00009 in-sample,
  4/5, < min_delta, 04 ist rejected) → nicht übernehmen. Nächster Blend-Hebel: weiteres diverses, stärkeres Modell
  (Rang-Korr. < 0.975, OOF ≥ ~0.945), nicht weitere GBDT-Varianten.
- [2026-11-26] 06_logreg_v4 (H11): LogReg (One-Hot + StandardScaler fold-intern in der Pipeline, C=1.0) auf neuem
  Feature-Set v4 (v1 + velocity_is_zero, log1p-Spalten, feste Dezil-Bins, geclippte Zähler, 7 Flag-Interaktionen)
  → OOF 0.94484, Δ −0.00147 zu 01_lgbm_baseline (0/5 Folds; nur Kontext) → neutral (divers, Blend-Kandidat).
  Code geprüft, leakage-frei: Bin-Grenzen = exakt die Dezile über train+test (label-frei, nachgerechnet, als Konstanten),
  v1–v3 unverändert, get_folds(). Rang-Korr. 0.967 (XGB) / 0.969 (LGBM) / 0.978 (Cat) / 0.976 (blend_hc) = diversestes
  brauchbares Modell (04 ist diverser, aber schwach). Blend-Probe nachgerechnet: blend_hc-Ränge ×0.85 + 06 ×0.15 → 0.94697 (+0.00008, 5/5 Folds);
  w 0.05–0.20 alle 5/5 Folds besser, w per 4-Fold gewählt und am 5. geprüft (genestet) → +0.00007 → echter, aber kleiner
  Zuschlag (< min_delta). Lineares Modell auf Bins erfasst offenbar Monotonie-/Additivanteile anders als GBDTs; der
  Abstand zu GBDTs (−0.0015) spricht für fehlende Interaktionen.
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
- [Z1-Fazit] Feature Engineering für GBDTs auf v1 stagniert (H05 −0.0073, H06 −0.0003). Größte offene Hebel:
  GBDT-Tuning (H07) und diverse, starke Nicht-GBDT-Modelle (H13, MLP) für den Blend.
- Fold-Std (0.0033) ist groß gegenüber typischen Gewinnen → Vergleiche immer gepaart auf
  denselben Folds (OOF-AUC), nicht über Fold-Mittelwerte.
- velocity_score = 0.0 kommt 69 426× vor (train+test; 48 499× in train ≈ 6.9 %) mit Fraud-Rate 0.04 %
  (Basis 1.67 %) → Häufungs-/Imputationswert, fast sicher „kein Fraud“. Numerische Splits fangen ihn
  bereits (velocity_score ist Top-5-Gain in v1); ggf. als Flag `velocity_is_zero` für lineare/NN-Modelle (H11).
- Methodik (alle Experimente gleich betroffen): `src/models.py` nutzt den Validierungs-Fold auch fürs
  Early Stopping → OOF leicht optimistisch. CV ≈ LB zeigt, dass der Effekt klein ist; Vergleiche bleiben fair.
