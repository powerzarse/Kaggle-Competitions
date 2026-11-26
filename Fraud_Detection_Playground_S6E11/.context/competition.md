# Competition: Playground Series S6E11 – Predicting Credit Card Fraud

- Slug: `playground-series-s6e11` · https://www.kaggle.com/competitions/playground-series-s6e11
- Metrik: **ROC AUC** (nur Ranking zählt → Kalibrierung egal, Rank-Blending ok)
- Deadline: **2026-11-30 23:59 UTC** (Team-Merge ebenso) · max. **10 Submissions/Tag**
- Am Ende 2 Submissions für den Private LB wählen; Public LB ≈ 20 % des Tests (Playground-üblich)

## Daten
| Datei | Zeilen | Spalten |
|---|---|---|
| train.csv | 699 960 | id + 24 Features + `is_fraud` |
| test.csv | 299 984 | id + 24 Features |
| sample_submission.csv | 299 984 | `id,is_fraud` |

- Ziel `is_fraud`: 1,67 % positiv → Stratified 5-Fold (`data/folds.csv`, Seed 42)
- Numerisch (13): amount_usd, hours_since_last_txn, txn_count_last_24h, distance_from_home_km,
  card_age_months, customer_age, account_balance_usd, cvv_retry_count, velocity_score,
  time_of_day_hour, day_of_week, merchant_risk_score, prior_disputes
- Boolesch (6): is_foreign_transaction, is_new_merchant, used_vpn, ip_country_mismatch,
  billing_shipping_mismatch, is_ai_generated_scam_attempt
- Kategorial (5): merchant_category (12), card_type (5), auth_method (5), channel (5), device_type (8)
- Keine fehlenden Werte. Synthetisch generiert (Playground) → Originaldatensatz evtl. auf Kaggle zu finden.

## Rechenumgebung
- Cloud-Container: 4 CPU-Kerne, 15 GB RAM, keine GPU → **ein Trainingsjob gleichzeitig**.
- LGBM 5-Fold auf v1: ~3 Min.
