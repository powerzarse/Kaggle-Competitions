# Predicting Credit Card Fraud – Playground Series S6E11

Competition: https://www.kaggle.com/competitions/playground-series-s6e11
Metrik: **ROC AUC** · Deadline: **30.11.2026 23:59 UTC** · max. 10 Submissions/Tag

## Daten
| Datei | Zeilen | Inhalt |
|---|---|---|
| `train.csv` | 699 960 | 24 Features + `is_fraud` (Fraud-Rate 1,67 %) |
| `test.csv` | 299 984 | 24 Features |
| `sample_submission.csv` | 299 984 | `id,is_fraud` |

Features: 13 numerisch (Betrag, Velocity, Merchant-Risk, …), 6 boolesche Risiko-Flags
(VPN, IP-Mismatch, Auslandstransaktion, …), 5 kategorial (Merchant-Kategorie, Kartentyp,
Auth-Methode, Kanal, Gerät). Keine fehlenden Werte.

## Struktur
```
Fraud_Detection_Playground_S6E11/
├── CLAUDE.md          # Arbeitsregeln für Claude in diesem Ordner
├── EXPERIMENTS.md     # Ergebnis-Log (wird von src/tracking.py fortgeschrieben)
├── src/
│   ├── config.py      # Pfade, Spaltenlisten, Seed, Folds
│   ├── data.py        # CSV -> Parquet-Cache, Dtypes
│   ├── features.py    # benannte Feature-Sets (base, v1, …)
│   ├── cv.py          # feste Stratified-5-Folds + run_cv()
│   ├── models.py      # lgbm / xgb / catboost / logreg im fit_predict-Format
│   ├── tracking.py    # OOF/Test speichern, Submission schreiben, Log-Eintrag
│   ├── blend.py       # Rank-Blend + Hill Climbing über OOFs
│   └── kaggle_api.py  # download / submit / submissions (REST, ohne CLI)
├── experiments/
│   ├── _template/run.py      # Vorlage für neue Experimente
│   └── 01_lgbm_baseline/     # erstes Experiment
├── notebooks/         # EDA
├── data/              # Rohdaten + folds.csv (nicht im Git)
├── models/            # OOF-/Test-Vorhersagen je Experiment (nicht im Git)
├── submissions/       # Submission-CSVs (nicht im Git)
├── download_data.sh
└── requirements.txt
```

## Workflow
```bash
pip install -r requirements.txt
./download_data.sh                                    # Skill: fraud-data
cp -r experiments/_template experiments/02_xgb_v1     # Skill: fraud-experiment
python experiments/02_xgb_v1/run.py
python -m src.blend                                   # Skill: fraud-blend
python -m src.kaggle_api submit submissions/blend_hc.csv "blend"   # Skill: fraud-submit
```

## Claude-Skills (`/.claude/skills/`)
| Skill | Zweck |
|---|---|
| `fraud-data` | Daten laden & prüfen |
| `fraud-experiment` | Experiment anlegen, CV laufen lassen, loggen |
| `fraud-blend` | OOF-basiertes Rank-Blending |
| `fraud-submit` | Validieren, (nach Bestätigung) einreichen, LB-Score loggen |
