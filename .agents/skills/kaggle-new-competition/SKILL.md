---
name: kaggle-new-competition
description: Legt für eine neue Kaggle-Competition einen Projektordner nach der Repo-Konvention an (src/, experiments/_template, .context/, EXPERIMENTS.md), angepasst an Metrik, Ziel und Spalten. Verwenden bei "neue Competition", "neues Kaggle-Projekt aufsetzen" oder einem Kaggle-Competition-Link.
---

# Neue Competition aufsetzen

Referenz-Implementierung: `Fraud_Detection_Playground_S6E11/` (binäre Klassifikation, ROC AUC).

## Ablauf
1. Metadaten holen (Slug aus der URL):
   ```bash
   curl -s "https://www.kaggle.com/api/v1/competitions/list?search=<slug>"
   curl -s "https://www.kaggle.com/api/v1/competitions/data/list/<slug>"
   ```
   → Titel, Metrik, Deadline, Submissions/Tag, Dateien.
2. Ordner `<Name>/` anlegen und aus der Referenz kopieren: `src/`, `experiments/_template/`,
   `.gitignore`, `requirements.txt`, `download_data.sh`, leere `data/ models/ submissions/ notebooks/` (mit `.gitkeep`).
   **Nicht** kopieren: `experiments/0*`, `EXPERIMENTS.md`, `.context/`-Inhalte.
3. `src/config.py`: `COMPETITION`, `ID`, `TARGET`, `METRIC`, Spaltenlisten anpassen.
   Andere Metrik/Aufgabe → `src/cv.py` (Scorer, Stratifizierung) und `src/models.py` (Objective) anpassen;
   Regression: `KFold` statt `StratifiedKFold`, `predict` statt `predict_proba`.
4. Daten laden (Skill `kaggle-data`), kurze EDA.
5. `.context/` anlegen (Dateien wie in der Referenz): `competition.md` mit echten Fakten,
   `rules.md` (Defaults übernehmen, `auto_submit: false`), leeres `learnings.md`, `journal.md`,
   `state.md`, und ein `backlog.md` mit 5–8 Start-Hypothesen (Baseline zuerst).
6. Baseline als `experiments/01_<modell>_baseline` (Skill `kaggle-experiment`), Ordner im Root-`README.md` eintragen.
7. Ab hier übernimmt der Skill `kaggle-autopilot`.
