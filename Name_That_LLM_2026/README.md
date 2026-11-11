# Name That LLM (Kaggle)

Competition: https://www.kaggle.com/competitions/name-that-llm

## Struktur
```
Name_That_LLM_2026/
├── PLAN.md          # Vorgehen, Validierung, Modellauswahl
├── data/            # Rohdaten von Kaggle (nicht im Git)
├── notebooks/       # 01_EDA.ipynb – Datenanalyse
├── src/             # data, features, models, cv, submission
├── experiments/     # je Modellansatz ein Ordner (01 TF-IDF … 05 Ensemble)
├── models/          # OOF-Vorhersagen & Modelle je Experiment (nicht im Git)
├── submissions/     # Submission-Dateien (Parquet)
├── download_data.sh
└── requirements.txt
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Kaggle-Token: kaggle.com -> Settings -> API -> "Create New Token"
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Regeln der Competition auf der Website akzeptieren, dann:
./download_data.sh
```
