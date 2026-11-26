# 04_lgbm_v2 – LGBM (Defaults) auf Feature-Set v2

**Hypothese (H05):** Bei synthetischen Playground-Daten tragen die exakten Werte numerischer Spalten
bzw. deren Häufigkeit Information (Rundungsartefakte des Generators).

**Änderung:** neues Feature-Set `v2` = `v1` + 12 Spalten (Modell `lgbm`, `params={}`, identisch zu 01):
- `_cat`: txn_count_last_24h, time_of_day_hour, card_age_months, customer_age, velocity_score,
  merchant_risk_score als pandas-`category`; Kategorienliste = sortierte Vereinigung train+test
  (identisch in beiden Frames).
- `_cnt`: Count-Encoding für velocity_score, merchant_risk_score, hours_since_last_txn,
  distance_from_home_km, amount_usd, account_balance_usd, gezählt über train+test zusammen.
- **Leakage:** keins. Das Count-Encoding nutzt nur Werte-Häufigkeiten, keine Labels. train+test gemeinsam
  zu zählen ist transduktiv, aber label-frei. Deshalb steht es in `features.py` (über das neue
  `JOINT_FEATURE_SETS` in `build`) und nicht fold-intern. `base`/`v1` sind unverändert.
- Kategorie-Params (`max_cat_to_onehot=4`, `cat_smooth=10`) nicht gesetzt: Das sind die LightGBM-Defaults,
  der Smoke-Test ergab damit dieselbe AUC.

**Ergebnis:** OOF-AUC **0.93902** (Fold-Mittel 0.93909 ± 0.00368).
Folds [0.93708, 0.94032, 0.94411, 0.94070, 0.93325] gegenüber 01_lgbm_baseline
[0.94395, 0.94848, 0.9512, 0.94663, 0.94171] → **Δ −0.00729, 0/5 Folds besser**. Die Hypothese ist in dieser Form widerlegt.
Vermutung: Die hochkardinalen `_cat`-Spalten (bis 959 Level) erlauben LightGBM Overfitting über
kategoriale Splits. Dazu passt, dass Early Stopping früher greift (Laufzeit kürzer als bei 01).

**Laufzeit:** 103 s für den vollen 5-Fold-Lauf (4 CPU). Smoke-Test: 1 Fold mit lr 0.2, v2 0.9254 gegenüber v1 0.9396.
