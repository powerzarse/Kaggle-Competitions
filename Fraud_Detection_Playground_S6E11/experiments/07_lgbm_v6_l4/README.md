# 07_lgbm_v6_l4 – Wert-/Ziffern-Target-Encoding + flache LGBM-Bäume (4 Leaves)

**Hypothese (H14):** Die Daten sind „Label zuerst, Felder gegeben Label“ erzeugt. Jeder exakte Wert und jede Ziffer einer Zahl
trägt deshalb eigenes, fast additives Signal. Ein geglättetes OOF-Target-Encoding dieser Schlüssel zusammen mit sehr flachen
Bäumen sollte die Lücke zum Feld schließen. Erwartet war OOF ≈ 0.950–0.951.

**Änderung:**
- `src/features.py`: neues Set `v6` = `base` (24 Rohspalten, Kategorien als `category`) + `cents_<c>` = round(x·100) % 100 für die
  6 Floats + **54 String-Schlüssel** `…__key`. Davon sind 24 exakte Werte aller Feature-Spalten und 30 Ziffern-Schlüssel
  (`cents`, `int` = floor(x), `ten` = floor(x/10), `dec` = floor(x·10), `l2` = Länge·100 + erste 2 Ziffern des Ganzzahlteils)
  für amount_usd, hours_since_last_txn, distance_from_home_km, account_balance_usd, velocity_score und merchant_risk_score.
  Keine Labels; bestehende Sets sind unverändert.
- `src/models.py`: neuer Helfer `target_encode_keys` (gleiche Semantik wie `target_encode`, aber vektorisiert: `pd.factorize` +
  `np.bincount`, ~4 s je äußerem Fold) und neue Factory `lgbm_vte`. Sie encodiert alle `__key`-Spalten (m-Estimate, m=20),
  entfernt die Schlüssel und trainiert dann LGBM auf 84 Spalten (30 roh + 54 TE). Defaults: num_leaves 4, lr 0.03,
  colsample 0.5, min_child_samples 100, subsample 0.8, n_estimators 20000, Early Stopping 300.

**Leakage-Schutz:** Das TE läuft komplett in `fit_predict`. X_tr bekommt OOF-Werte über 5 innere `StratifiedKFold(shuffle, SEED=42)`-Splits
von X_tr, mit dem Prior des jeweiligen inneren Fit-Teils. X_va und X_test bekommen das Mapping aus ganz X_tr. Ungesehene Werte
erhalten den Prior (count 0). Die Codes kommen aus `factorize` über X_tr, X_va und X_te zusammen; das ist rein label-frei.
Das Val-Label dient wie in allen LGBM-Läufen nur dem Early Stopping.

**Smoke-Test (Fold 0, lr 0.1, 4 Leaves):** 0.94867 mit innerem Seed 42. Mit dem inneren Seed 0 der Probe kommt
**exakt 0.94947 / 763 Bäume** heraus, das Probe-Ergebnis ist also reproduziert. Die Streuung über den Seed ist groß
(0/1/2/42: 0.94947/0.94894/0.94920/0.94867). Der Lauf bleibt beim Repo-Seed 42.

**Ergebnis:** OOF-AUC **0.95019** (Fold-Mittel 0.95025 ± 0.00350).
Folds [0.94926, 0.95115, 0.95508, 0.95142, 0.94435] gegenüber 01_lgbm_baseline [0.94395, 0.94848, 0.9512, 0.94663, 0.94171]
→ **Δ +0.00388, 5/5 Folds besser**. Gegenüber blend_02 (0.94698) sind es +0.00321.
Best-Iteration je Fold: 1906 / 2754 / 1722 / 1843 / 2060.
Rang-Korrelation 0.954 zu 01 und 02, 0.950 zu 03, 0.993 zu 08.

**Laufzeit:** 448 s für den vollen 5-Fold-Lauf (4 CPU). Davon entfallen ~18 s auf Features und ~4 s je Fold auf das TE.
