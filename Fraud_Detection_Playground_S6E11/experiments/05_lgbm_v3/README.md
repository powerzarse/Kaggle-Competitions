# 05_lgbm_v3 – LGBM mit fold-internem Target-Encoding auf Feature-Set v3

**Hypothese (H06):** Target-Encoding der Kategorien plus der Interaktionen channel×device_type,
auth_method×channel und merchant_category×channel liefert LightGBM zusätzliche Information. Die
Interaktionen gehen bewusst numerisch (TE) ein und nicht als weitere `category`-Spalten (Lehre aus 04_lgbm_v2).

**Änderung:**
- `src/features.py`: neues Set `v3` = `v1` + 3 String-Schlüssel `<a>__<b>` (40/25/60 Level). Sie dienen nur als Rohmaterial
  fürs Encoding. `base`, `v1` und `v2` sind unverändert, `v3` wurde zusätzlich in `FEATURE_SETS` eingetragen.
- `src/models.py`: neue Factory `lgbm_te` und Helfer `target_encode`. Die 5 CAT_COLS und die 3 Schlüssel werden encodiert
  (neue Spalten `<name>_te`, m-Estimate mit m=20 gegen den Prior). Danach fallen die Schlüssel weg, die CAT_COLS
  bleiben als `category` erhalten. Anschließend läuft unverändert `lgbm(params)`, mit denselben Defaults und demselben Early Stopping (200).

**Leakage-Schutz:** Alles passiert in `fit_predict`, also fold-intern. Die TE-Werte für X_tr sind out-of-fold über 5 innere
`StratifiedKFold(shuffle, SEED)`-Splits auf X_tr; der Prior ist dabei der Mittelwert des inneren Fit-Teils. X_va und X_test
bekommen das Mapping aus ganz X_tr. Unbekannte Werte erhalten den Prior. Das Validierungs-Label wird nicht verwendet
(nur fürs Early Stopping, wie in allen LGBM-Läufen).

**Ergebnis:** OOF-AUC **0.94605** (Fold-Mittel 0.94613 ± 0.00317).
Folds [0.94400, 0.94766, 0.95097, 0.94636, 0.94166] gegenüber 01_lgbm_baseline
[0.94395, 0.94848, 0.9512, 0.94663, 0.94171] → **Δ −0.00026, 1/5 Folds besser**. Neutral bis leicht schlechter.
Die Rang-Korrelation zu 01 liegt bei 0.989, das Modell bringt also kaum neue Diversität.
Smoke-Test (Fold 0, lr 0.2): v3/lgbm_te 0.93984 gegenüber v1/lgbm 0.93961.

**Laufzeit:** 156 s für den vollen 5-Fold-Lauf (4 CPU), gegenüber 01 etwas langsamer wegen des TE und der 8 zusätzlichen Spalten.
