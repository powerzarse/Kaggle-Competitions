---
name: kaggle-experiment
description: Legt ein neues Modell- oder Feature-Experiment in einem Kaggle-Competition-Ordner an, führt die feste K-Fold-CV aus und protokolliert das Ergebnis (models/, submissions/, EXPERIMENTS.md). Verwenden bei "neues Experiment", "probier XGBoost/CatBoost/neue Features", "trainiere Modell".
---

# Neues Experiment

`COMP` = Competition-Ordner mit `.context/`. Alle Befehle aus `COMP/`.
Regeln in `COMP/.context/rules.md` gelten vorrangig (Folds, Leakage, Reproduzierbarkeit).

## Ablauf
1. Nächste Nummer: `ls experiments/` → `NN` = höchste + 1.
2. Vorlage kopieren:
   ```bash
   cp -r experiments/_template experiments/NN_<kurzname>
   ```
3. In `run.py` nur Docstring und `CONFIG` anpassen:
   - `features`: Schlüssel aus `src/features.py::FEATURE_SETS`. Neue Features → **neue Funktion**
     (`v2`, `v3`, …) anlegen und registrieren; bestehende Sets nie ändern.
   - `model`: Schlüssel aus `src/models.py::MODELS`. Neues Modell → Factory
     `fn(params=None) -> fit_predict(X_tr, y_tr, X_va, y_va, X_te) -> (pred_va, pred_te)` ergänzen.
     Alles, was Labels nutzt (Target-Encoding …), gehört **in** `fit_predict` (fold-intern).
   - `params`: überschreibt Defaults · `note`: Hypothese/Backlog-ID in einem Satz.
4. Pakete bei Bedarf: `pip install -r requirements.txt`.
5. Smoke-Test (billig), dann voller Lauf im Hintergrund:
   ```bash
   python experiments/NN_<kurzname>/run.py 2>&1 | tee experiments/NN_<kurzname>/run.log
   ```
6. `experiments/NN_<kurzname>/README.md`: Hypothese, Änderung, Ergebnis, Laufzeit (2–5 Zeilen).
7. Melden: OOF-AUC, Fold-Std, Δ zur Bestmarke (`python -m src.status --offline`).
   Die Bewertung (Learning, Backlog-Status) macht der Reviewer bzw. der Skill-Nutzer danach.
