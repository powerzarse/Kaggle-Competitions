# 11_glmboost_v6 – GLM-Boost: LGBM (4 Leaves) auf dem Residuum des GLM

**Hypothese (H16):** LightGBM startet mit `init_score` = GLM-Logit (GLM wie 10_glm_v6) und lernt deshalb nur das, was das
additive GLM nicht erklärt. Öffentlich ist das das beste Einzelmodell (0.9514 vs. 0.9507 für reines 4-Leaf-LGBM). Erwartet
waren **+0.0005 zu 07**.

**Änderung:** In `src/models.py` gibt es eine neue Factory `glmboost_vte`, rein additiv; `lgbm_vte` und `glm_vte` sind unverändert.
Der Ablauf in `fit_predict`:
1. TE über `vte_frames`, identisch zu 07 und 10 (m=20, 5 innere Folds, Seed 42).
2. GLM wie in `glm_vte`: logit(geclipptes TE) + 6 Flags + One-Hot(5 Kategorien), StandardScaler/OneHot in der Pipeline,
   LogReg mit C=1.
3. LGBM mit den Defaults von `lgbm_vte` (4 Leaves, lr 0.03, colsample 0.5, min_child_samples 100, subsample 0.8,
   Early Stopping 300) auf denselben 84 Spalten wie 07. Dabei gilt `init_score` = GLM-Logit, und Early Stopping läuft auf X_va
   mit `eval_init_score`.
4. Die Vorhersage ist `sigmoid(raw_score + GLM-Logit)` für Val und Test.

**Variante A (gewählt):** Die GLM-Logits für X_tr sind innere 5-Fold-OOF auf X_tr (`StratifiedKFold`, Seed 43). Das GLM wird
dafür je innerem Fold auf den schon vorhandenen TE-Features von X_tr gefittet. Für X_va und X_te nimmt die Factory das GLM auf ganz X_tr.
Damit verhält sich init_score im Train so wie in Val/Test: Im Log liegt die OOF-GLM-AUC auf X_tr bei 0.948–0.951 und ist damit
etwa so hoch wie die Val-AUC. In-Sample läge sie höher (Fold 0: 0.95033 gegenüber 0.94974 OOF).
**Variante B:** In-Sample-Logits für X_tr (`glm_oof=False`).
**Smoke-Test (Fold 0, lr 0.1):** A 0.94974 (274 Bäume, 228 s), B 0.94979 (227 Bäume, 60 s). B ist nicht klar besser
(+0.00005, innerhalb des TE-Seed-Rauschens von bis zu 0.0008), deshalb A wie vorgesehen.
Zum Vergleich auf Fold 0 mit lr 0.1: 07 0.94867, 10 0.94906.

**Ergebnis:** OOF-AUC **0.95059** (Fold-Mittel 0.95064 ± 0.00328).
Folds [0.94961, 0.95113, 0.95554, 0.95152, 0.94539].
- Gegenüber 07 [0.94926, 0.95115, 0.95508, 0.95142, 0.94435]: **Δ +0.00040, 4/5 Folds besser** (Fold 1 −0.00002).
- Gegenüber 10: Δ +0.00049, 5/5 Folds besser.
- Best-Iteration je Fold: 391 / 864 / 530 / 862 / 1114, bei 07 waren es 1722–2754. LGBM braucht also nur noch etwa ein Drittel
  der Bäume, weil das GLM den additiven Teil vorab trägt.
- Test-Mittel 0.01669 und OOF-Mittel 0.01674 stimmen überein.
- Rang-Korrelation (OOF): 0.988 zu 07, **0.996 zu 10**.

**Blend-Probe (Rank-Mittel der OOF-Ränge, In-Sample, nichts gespeichert):**
- 11 allein: 0.95059. Rank-Blend 07+10 (0.5/0.5) zum Vergleich: 0.95061.
- 11+10: w(10) 0.2/0.3/0.4/0.5 → 0.95057/0.95055/0.95051/0.95047. Das ist immer schlechter als 11 allein, weil 10 in 11 schon enthalten ist.
- 07+10+11 gleichgewichtet: **0.95068** (Δ +0.00009 zu 11, 4/5 Folds; Δ +0.00049 zu 07, 5/5). 0.25/0.25/0.5 bzw. 0.3/0.3/0.4
  ergeben 0.95069. Der Zuschlag gegenüber 11 allein bzw. 07+10 liegt unter min_delta.
- Fazit: 11 ist als Einzelmodell etwa so gut wie der Blend 07+10 und ersetzt ihn praktisch, bringt aber kaum Zusatzdiversität.

**Laufzeit:** 415 s für den vollen 5-Fold-Lauf mit 4 CPU. Davon gehen ca. 6 GLM-Fits je Fold auf Variante A.
LightGBM 4.7 gibt je Fold eine harmlose `LGBMDeprecationWarning` aus (`eval_set` → `eval_X`/`eval_y`).
