# Name That LLM (Kaggle)

Competition: https://www.kaggle.com/competitions/name-that-llm

Architektur, Datenfluss und Ergebnisse: [ARCHITECTURE.md](ARCHITECTURE.md)

## Struktur
```
Name_That_LLM_2026/
├── ARCHITECTURE.md  # Architektur- & Flow-Diagramme (Mermaid), Ergebnisse
├── src/             # gemeinsamer Code: Daten, Features, CV, Modelle
├── experiments/     # je Ansatz ein Ordner (01 TF-IDF … 06 externe Daten)
├── kaggle/          # Generatoren für Kaggle-Notebooks (Abgabe, GPU-Training, OOF)
├── notebooks/       # 01_EDA.ipynb – Datenanalyse
├── data/            # Rohdaten & Open Pool (nicht im Git)
├── models/          # OOF-Vorhersagen & Modelle je Experiment (nicht im Git)
├── download_data.sh
└── requirements.txt
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Kaggle-Token: kaggle.com -> Settings -> API -> "Create New Token"
mkdir -p ~/.kaggle && echo <TOKEN> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

# Regeln der Competition auf der Website akzeptieren, dann:
./download_data.sh
```
