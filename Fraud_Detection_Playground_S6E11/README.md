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
├── .context/          # Gedächtnis des Agentensystems: rules, state, backlog, learnings, journal
├── EXPERIMENTS.md     # Ergebnis-Log (wird von src/tracking.py fortgeschrieben)
├── src/
│   ├── config.py      # Pfade, Spaltenlisten, Seed, Folds
│   ├── data.py        # CSV -> Parquet-Cache, Dtypes
│   ├── features.py    # benannte Feature-Sets (base, v1, …)
│   ├── cv.py          # feste Stratified-5-Folds + run_cv()
│   ├── models.py      # lgbm / xgb / catboost / logreg im fit_predict-Format
│   ├── tracking.py    # OOF/Test speichern, Submission schreiben, Log-Eintrag
│   ├── blend.py       # Rank-Blend + Hill Climbing über OOFs
│   ├── status.py      # Lagebericht: Rangliste, Rang-Korrelationen, Submission-Budget
│   └── kaggle_api.py  # download / limits / submit / submissions (ohne CLI)
├── experiments/
│   ├── _template/run.py      # Vorlage für neue Experimente
│   ├── 01_lgbm_baseline/
│   ├── 02_xgb_v1/
│   └── 03_catboost_v1/
├── notebooks/         # EDA
├── data/              # Rohdaten + folds.csv (nicht im Git)
├── models/            # OOF-/Test-Vorhersagen je Experiment (nicht im Git)
├── submissions/       # Submission-CSVs (nicht im Git)
├── download_data.sh
└── requirements.txt
```

## Workflow
Automatisiert über das Agentensystem im Repo-Root (`/.agents/`, siehe dortiges README):
```bash
../.agents/setup_claude.sh      # einmal pro Clone, macht die Skills für Claude Code sichtbar
```
Dann in Claude Code `/kaggle-autopilot` (ein Zyklus) oder `/loop /kaggle-autopilot`.

Manuell:
```bash
pip install -r requirements.txt
./download_data.sh                                     # Skill: kaggle-data
cp -r experiments/_template experiments/04_<name>      # Skill: kaggle-experiment
python experiments/04_<name>/run.py
python -m src.status                                   # Rangliste + Budget
python -m src.blend                                    # Skill: kaggle-blend
python -m src.kaggle_api submit submissions/blend_hc.csv "blend"   # Skill: kaggle-submit
```
