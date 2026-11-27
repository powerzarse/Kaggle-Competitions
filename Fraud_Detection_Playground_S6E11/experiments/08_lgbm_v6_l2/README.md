# 08_lgbm_v6_l2 – Wert-/Ziffern-Target-Encoding + LGBM-Stümpfe (2 Leaves)

**Hypothese (H14, Ablation der Baumtiefe):** wie 07_lgbm_v6_l4, aber mit num_leaves=2. Bei rein additivem Signal reichen
Stümpfe aus. Auf Fold 0 mit lr 0.1 lag die Probe bei 0.94961 gegenüber 0.94947 mit 4 Leaves.

**Änderung:** Gleicher Code wie 07 (Feature-Set `v6`, Factory `lgbm_vte`). Nur `params={"num_leaves": 2}` ist anders.
Details zu Schlüsseln und TE stehen in `experiments/07_lgbm_v6_l4/README.md`.

**Leakage-Schutz:** wie 07. Das TE läuft fold-intern, X_tr bekommt OOF-Werte über 5 innere Splits, Val und Test das Mapping
aus ganz X_tr, ungesehene Werte den Prior.

**Ergebnis:** OOF-AUC **0.95004** (Fold-Mittel 0.95014 ± 0.00343).
Folds [0.94925, 0.95094, 0.95505, 0.95101, 0.94445]
→ Δ +0.00373 zu 01_lgbm_baseline (5/5 Folds) und **Δ −0.00015 zu 07** (1/5 Folds besser).
Das liegt unter min_delta: Stümpfe sind praktisch gleich gut wie 4 Leaves, aber nicht besser.
Best-Iteration je Fold: 4533 / 4691 / 3675 / 3703 / 5152. Rang-Korrelation zu 07: 0.993.
Der ungewichtete Rank-Mean aus 07 und 08 kommt auf 0.95031, also +0.00012 zu 07 (In-Sample, < min_delta).

**Laufzeit:** 697 s für den vollen 5-Fold-Lauf (4 CPU), wegen der rund doppelt so vielen Runden.
