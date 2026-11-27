# Learnings

Kurze, überprüfte Erkenntnisse. Jede Zeile: Was wurde geändert → Δ OOF-AUC → Urteil → Warum (Vermutung).
Neueste oben. Der Researcher leitet daraus neue Hypothesen ab; nichts doppelt testen, was hier steht.

## Gesichert
- [2026-11-27] 11_glmboost_v6 (H16): GLM-Boost = LGBM 4 Leaves (Defaults wie 07) mit `init_score` = GLM-Logit (GLM wie 10,
  C=1; X_tr innere 5-Fold-OOF-Logits, X_va/X_te vom GLM auf ganz X_tr) → OOF 0.95059, **Δ +0.00040 zu 07 (4/5 Folds,
  Fold 1 −0.00002)** → besser, **neues bestes Einzelmodell**; ggü. blend_03 (07+10) Δ −0.00002 (2/5) → gleichwertig.
  Das GLM trägt den additiven Teil, LGBM lernt nur das Residuum (best_iter 391–1114 statt 1722–2754) → 11 ist praktisch
  „07+10 in einem Modell“ (Rang-Korr. 0.996 zu 10, 0.988 zu 07; Test 0.997/0.991). Code geprüft, leakage-frei: neue Factory
  rein additiv (lgbm_vte/glm_vte unverändert), get_folds(); innere GLM-Fits nur auf X_tr-Teilen und auf den schon
  OOF-encodeten X_tr-TE-Spalten (OOF-GLM-AUC tr 0.948–0.951 ≈ Val → kein Train/Val-Bruch); `eval_init_score` beim Early
  Stopping, Vorhersage sigmoid(raw_score + GLM-Logit) für Val und Test; Test-Mittel 0.01669 = OOF-Mittel 0.01674. Rest-Effekt
  zweiter Ordnung: TE-Split (Seed 42) ≠ GLM-Split (Seed 43) → TE der Fit-Zeilen enthält Labels der Enc-Zeilen (unvermeidbar
  ohne genestetes TE, im Log nicht sichtbar). Smoke A (OOF) vs. B (In-Sample) 0.94974 vs. 0.94979 (Fold 0, lr 0.1) → kein
  Unterschied; B ist ~4× schneller. **Genestete Blend-Prüfung** (Grid 0.05 auf 4 Folds, am 5. gemessen): 07+10+11 → gepoolt
  0.95070, **Δ +0.00010 zu blend_03 (4/5)**, +0.00011 zu 11 (4/5, Fold 4 −0.00016); Gewichte stabil 07 0.30–0.40 / 10 0.05–0.10 /
  11 0.50–0.65; 07+11 allein (w(07) 0.30–0.40) liefert dasselbe (+0.00011) → 10 ist in 11 aufgegangen, 07 legt noch etwas
  drauf, aber alles < min_delta. In-Sample: 07/10/11 = 0.35/0.05/0.60 → 0.95072, 0.3/0/0.7 → 0.95071, je 1/3 → 0.95068.
  Folge: GLM-Verbesserungen (H17) zahlen jetzt direkt auf das beste Einzelmodell ein.
- [2026-11-27] 10_glm_v6 (H15b): GLM = LogReg (C=1) auf logit(54 v6-TE, geclippt 1e-4) + 6 Flags + One-Hot(5 Kategorien),
  TE via `vte_frames` wie 07 → OOF 0.95010, Δ −0.00009 zu 07 (2/5 Folds) → neutral, **Blend-Partner Nr. 1**. Ein lineares
  Modell ist auf den TE-Logits so stark wie 4-Leaf-Bäume → Signal ist praktisch additiv in logit(TE). Code geprüft,
  leakage-frei: X_tr-TE innere OOF, Val/Test-Mapping aus ganz X_tr, logit nur auf `_te`-Spalten aus `vte_frames`,
  OneHot/StandardScaler in der Pipeline nur auf X_tr gefittet, kein Val-Label genutzt, get_folds(). Rang-Korr. 0.982 zu 07
  (knapp über 0.98, Test 0.986). **Genestete Blend-Prüfung** (w(10) per Grid 0.05 auf 4 Folds gewählt, am 5. gemessen):
  07+10 → **Δ +0.00039 zu 07, 5/5 Folds (+0.00026…+0.00061)**, gewählte w(10) stabil 0.40–0.50, gepoolt 0.95060
  (In-Sample 0.5/0.5: 0.95061) → Blend-Gewinn ist echt (≈ 2×min_delta), kein Gewichts-Overfit; Ränge je Fold statt global
  geben dasselbe (+0.00040). Damit erstmals ein Blend-Kandidat über 07; Verbesserungen am GLM (H16/H17) zahlen jetzt
  direkt auf den Blend ein.
- [2026-11-27] 09_xgb_v6_d2 (H15a): XGB max_depth 2 (lr 0.04 statt geplant 0.03, colsample 0.5) auf v6-TE via `vte_frames`
  → OOF 0.95016, Δ −0.00003 zu 07 (2/5 Folds) → neutral, nicht divers (Rang-Korr. 0.995, Test 0.998). Zweites flaches
  GBDT auf denselben TE-Spalten lernt dieselbe additive Funktion. Genestet: 07+09 +0.00009 (4/5, < min_delta);
  07+09+10 (Grid auf 4 Folds) +0.00041 (5/5) vs. 07+10 +0.00039 → 09 legt nur +0.00002 drauf → kein Blend-Partner,
  weitere GBDT-Varianten auf v6 nicht als Diversität testen. Refactoring geprüft: `vte_frames` ist eine 1:1-Auslagerung
  der bisherigen TE-Zeilen aus `lgbm_vte` (gleiche Schlüsselauswahl, gleiche Argumente) → 07/08 bleiben reproduzierbar.
- [2026-11-27] 07_lgbm_v6_l4 (H14): Feature-Set v6 (30 Rohspalten inkl. Cents + 54 Wert-/Ziffern-Schlüssel, fold-internes
  OOF-TE m=20 via `lgbm_vte`) + LGBM 4 Leaves (lr 0.03, colsample 0.5) → OOF 0.95019, **Δ +0.00388 zu 01_lgbm_baseline
  (5/5 Folds, +0.0026…+0.0053), Δ +0.00322 zu blend_02 (5/5 Folds)** → besser. **Neue Basis** (bestes Einzelmodell und besser
  als jeder bisherige Blend). Ursache: Label-first-generierte Daten → jeder exakte Wert/jede Ziffer trägt fast additives Signal,
  das flache Bäume auf geglättetem TE abgreifen. Code geprüft, leakage-frei: TE nur aus y_tr, X_tr per innerer 5-Fold-OOF
  encodet (nicht In-Sample), X_va/X_te mit Mapping aus ganz X_tr, gemeinsames `factorize` nur label-frei, Schlüssel danach
  entfernt, v1–v4 und alte Factories unverändert, get_folds(). Gegenproben Fold 0 (lr 0.1, Basis 0.94867): y_tr permutiert →
  AUC 0.494; + Zufallsschlüssel (30k Level) + id als Schlüssel → 0.94850 (−0.0002, kein Gewinn); Einzel-TE amount_usd__key:
  AUC 0.540 OOF-Train / 0.555 Val vs. 0.739 bei In-Sample-TE → kein Train/Val-Bruch. Test-Mittel = OOF-Mittel (0.0168).
  Rang-Korr. 0.954 zu 01/02, 0.959 zu blend_02. Blend-Probe (In-Sample): 07 ×0.8 + blend_02-Ränge ×0.2 → 0.95033 (+0.00014)
  < min_delta → v1-Modelle bringen im Blend praktisch nichts mehr. **Relativiert**: (1) alle Befunde auf v1 (01–06, blend_hc,
  blend_02, „63 Leaves Referenz“) sind nur noch Kontext, Vergleichsmaßstab ab jetzt 07; (2) H06-Learning „TE bringt GBDTs
  nichts“ gilt nur für niedrigkardinale Kategorien – geglättetes TE *hochkardinaler* Wert-/Ziffern-Schlüssel ist der Hebel;
  (3) H05 bleibt: dieselben Werte als *native* Kategorie overfitten; (4) Blend-Hebel „diverses Modell OOF ≥ ~0.945“ ist
  überholt – Partner müssen auf v6-TE aufsetzen (H15/H16) und ≥ ~0.949 erreichen. Early Stopping auf Val wie bisher (H12).
- [2026-11-27] 08_lgbm_v6_l2 (H14, Ablation): wie 07, aber 2 Leaves (Stümpfe) → OOF 0.95004, Δ −0.00015 zu 07 (07 besser in
  4/5 Folds) → neutral ggü. 07 (ggü. alter Basis besser, Δ +0.00373, 5/5). Rang-Korr. 0.993 zu 07 (nicht divers);
  07+08 Rank-Mittel 0.95031 (+0.00012 < min_delta). 2 vs. 4 Leaves ist innerhalb des Seed-Rauschens des inneren TE
  (bis 0.0008 auf Fold 0, lr 0.1) nicht unterscheidbar → Signal ist fast rein additiv; 4 Leaves als Default behalten,
  Leaves-Ablation erst mit TE-Seed-Mittelung (H20) wiederholen.
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
- [2026-11-27 · Researcher Z2] **Warum wir hinter dem Feld liegen – Hauptursache gefunden** (Quellen: Notebooks
  megayak „S6E11 | Per-Value + Digit TE | Shallow-Tree Stack“ (v1–v18), evgendvorkin „S6E11 | Deep EDA | Wide LM Stack
  OOF 0.952642“, goodpjw2008 „Board Model + RealMLP“, asteriospanoras „Will the Top Survive Private?“):
  - **Datenerzeugung**: Zeilen stammen aus einem LLM-Generator (GReaT-artig), der **zuerst das Label zieht** und dann jedes
    Feld *gegeben dem Label* bei Temperatur ≈ 0.7 schreibt (megayak, Abschnitt 0: Label-first-Modell mit T 0.65–0.75 erklärt
    die Fraud-Raten je Wert 2–5× besser als die Original-Raten). Folge: (1) Das Signal ist **fast additiv** über die Felder,
    (2) **jeder exakte Wert und jede Ziffer/jedes Token einer Zahl** (Cents, Ganzzahlteil, Zehner, 1. Nachkommastelle,
    führende Ziffern, GPT-2-Tokens) trägt eigene Label-Information.
  - **Probe (eigene Daten, Fold 0, LGBM lr 0.1, Scratch)**: Rohspalten (+Cents als Zahl) 63 Leaves 0.94322 · **4 Leaves 0.94643**
    (+0.0032) · 16 Leaves 0.94449 · 4 Leaves ohne Cents 0.94587 · + fold-internes Per-Value-TE der 24 Spalten 0.94721 ·
    **+ Per-Value- und Ziffern-TE (54 Schlüssel, m=20, innere 5-Fold-OOF) 4 Leaves 0.94947** · dasselbe 2 Leaves (Stümpfe)
    **0.94961** · 8 Leaves 0.94922 · 63 Leaves 0.94571 · + Value-Counts aller Schlüssel 0.94871 (−0.0008 bei lr 0.1).
    Zum Vergleich unsere Modelle auf Fold 0: 01_lgbm 0.94395, 02_xgb 0.94463, blend_02 0.94506 → **+0.0044…+0.0057 auf
    Fold 0** allein durch flache Bäume + Ziffern-/Wert-TE. Das deckt die Lücke zum Median (~0.950) vollständig.
  - Öffentliche CV-gestützte Referenzen: 4-Leaf-LGBM/Tiefe-2-XGB auf TE ≈ 0.9506–0.9507 OOF, GLM (LogReg auf TE-Logits +
    Joint-Keys + Counts) 0.9508, LGBM mit `init_score` = GLM-Logit („GLM-Boost“) **0.9514 (bestes Einzelmodell)**, Blend 0.9515;
    evgendvorkin XGB: roh 0.94727 → Ganzzahl+Nachkomma-TE 0.94947 → + exaktes Wert-TE 0.95018 → + Gruppenprofile 0.95028.
    Mit GPT-2-Zeilen-LM (GPU, 18 GPU-h/Fold für medium) und RealMLP/TabM: breite Stacks ≈ **0.9526 OOF / 0.9531 LB** = CV-Decke.
  - Kein Widerspruch zu H05/H06: Zahl als *native* Kategorie overfittet (auch megayak: Ziffern als native LGBM-Kategorien
    0.94567, 5/5 Folds schlechter); TE niedrigkardinaler Kategorien ist redundant. Der Hebel ist **geglättetes OOF-TE
    hochkardinaler Wert-/Ziffern-Schlüssel + sehr flache Bäume**.
  - **Public LB > ~0.953 ist Public-Board-Fitting** (Mischen öffentlicher Submission-Dateien entlang LB-Richtungen, „board
    model“, „zero-fraud zones“, Rank-Extrapolation; ohne OOF). asteriospanoras: der 0.95504-Stand liegt per OOF-Replay ~0.0004
    *unter* seiner 0.95442-Basis. Für die Spitze (0.960–0.971) gibt es keinen CV-gestützten öffentlichen Beleg → nicht jagen,
    Private-LB (80 % des Tests) wird umsortieren. Realistisches Ziel: OOF 0.951–0.952.
- [2026-11-27] **Originaldatensatz** `uditjain13/credit-card-fraud-detection-2026` (CC0, 20 000 Zeilen, 339 Fraud = 1.695 %,
  gleiche 24 Spalten + `transaction_id`): **0 exakte Zeilen-Matches** train↔orig, test↔orig, train↔test; 0 Duplikate in train/test.
  Werte werden wiederverwendet (97 % der train-Zeilen haben einen amount_usd-Wert aus orig, 85 % bei account_balance_usd).
  Probe Fold 0, LGBM 63 Leaves lr 0.1: + Original als Zusatz-Trainingszeilen 0.94315 vs. 0.94322 (−0.00007) → **bringt nichts**
  (megayak: LogReg auf Original als Stack-Input −0.000016). Verteilungen weichen ab (u. a. Distanz-Tail, Stunde, Alter).
- [2026-11-27] **Kein Train/Test-Drift**: Adversarial Validation (LGBM, 25 % Holdout) AUC 0.5008. id train 0–699 959, test
  699 960–999 943. Keine Duplikate → keine Leak-/Duplikat-Hebel.
- [Z1-Fazit, **überholt durch Z2-Befund oben**: der Hebel ist Wert-/Ziffern-TE + flache Bäume] Feature Engineering für GBDTs auf v1 stagniert (H05 −0.0073, H06 −0.0003). Größte offene Hebel:
  GBDT-Tuning (H07) und diverse, starke Nicht-GBDT-Modelle (H13, MLP) für den Blend.
- Fold-Std (0.0033) ist groß gegenüber typischen Gewinnen → Vergleiche immer gepaart auf
  denselben Folds (OOF-AUC), nicht über Fold-Mittelwerte.
- velocity_score = 0.0 kommt 69 426× vor (train+test; 48 499× in train ≈ 6.9 %) mit Fraud-Rate 0.04 %
  (Basis 1.67 %) → Häufungs-/Imputationswert, fast sicher „kein Fraud“. Numerische Splits fangen ihn
  bereits (velocity_score ist Top-5-Gain in v1); ggf. als Flag `velocity_is_zero` für lineare/NN-Modelle (H11).
- Methodik (alle Experimente gleich betroffen): `src/models.py` nutzt den Validierungs-Fold auch fürs
  Early Stopping → OOF leicht optimistisch. CV ≈ LB zeigt, dass der Effekt klein ist; Vergleiche bleiben fair.
