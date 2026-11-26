# 06_logreg_v4 (H11)

**Hypothese (H11):** Eine lineare Modellfamilie (LogReg) rankt anders als die GBDTs und ergänzt den
Blend; `velocity_is_zero` + log-Transformationen + Bins machen die Daten linear-modell-freundlich.

**Features `v4`** (`src/features.py`, 55 Spalten vor One-Hot): v1 + `velocity_is_zero`; log1p von
account_balance, distance_from_home, hours_since_last_txn, amount_to_balance, txn_per_hour; feste
Dezil-Bins als `category` (10–11 Level, Grenzen als Konstanten, label-frei über train+test bestimmt)
für amount, hours_since_last_txn, account_balance, velocity_score (eigener Bin für genau 0),
merchant_risk_score, customer_age; geclippte Zähler als `category` (txn_count ≤ 7, cvv_retry ≤ 2,
prior_disputes ≤ 3); 7 Risiko-Flag-Interaktionen + `risk_flags_ge2`.

**Modell:** bestehende Factory `logreg` (One-Hot + StandardScaler, lbfgs, max_iter 2000), `C=1.0`.
C-Wahl auf Fold 0: C 0.1 → 0.94282, 0.3 → 0.94277, 1.0 → 0.94308 (Referenz LogReg auf v1, C 1: 0.94118).
Unterschiede sind klein, wir nehmen C 1.0. Keine Konvergenz-Warnungen.

**Ergebnis:** OOF-AUC **0.94484** (Folds 0.94308 / 0.94630 / 0.94975 / 0.94470 / 0.94059, Std 0.0031),
also −0.00205 zu blend_hc und −0.00169 zu 02_xgb_v1. Laufzeit 61 s für 5 Folds.
Rang-Korrelation zu den GBDTs: 0.967 (XGB), 0.969 (LGBM, LGBM-TE), 0.978 (CatBoost).

**Blend-Probe** (nur auf OOF, nichts gespeichert): Rank-Blend 0.25 LGBM / 0.5 XGB / 0.25 Cat (= blend_hc,
0.94689) mit Gewicht w für 06 und Rest (1−w) proportional:
w 0.05 → 0.94693 (+0.00004), 0.10 → 0.94696 (+0.00007), **0.15 → 0.94697 (+0.00008)**,
0.20 → 0.94696 (+0.00007); bis w 0.15 sind alle 5 Folds besser, ab w 0.25 kippen einzelne Folds.
Der Gewinn ist konsistent, liegt aber unter min_delta (0.0002).
