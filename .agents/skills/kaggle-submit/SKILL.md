---
name: kaggle-submit
description: Validiert eine Submission-CSV eines Kaggle-Competition-Ordners, reicht sie gemäß Submit-Policy (.context/rules.md) ein und trägt den Public-LB-Score in EXPERIMENTS.md und .context nach. Verwenden bei "submit", "einreichen", "auf Kaggle hochladen", "wie ist der LB-Score".
---

# Submission

`COMP` = Competition-Ordner mit `.context/`. Alle Befehle aus `COMP/`.
Limits und Deadline: `COMP/.context/competition.md`. Policy: `COMP/.context/rules.md`.

## 1. Validieren (immer)
```bash
python - <<'EOF'
import pandas as pd
from src.config import ID, TARGET
f = "submissions/<name>.csv"
s, ss = pd.read_csv(f), pd.read_csv("data/sample_submission.csv")
assert list(s.columns) == [ID, TARGET], s.columns
assert len(s) == len(ss) and (s[ID].values == ss[ID].values).all(), "IDs/Länge falsch"
assert s[TARGET].notna().all() and s[TARGET].nunique() > 1000, "NaNs oder konstant"
print("ok", s[TARGET].describe().round(4).to_dict())
EOF
```

## 2. Freigabe
- `auto_submit: false` → Datei, OOF-AUC, Δ zur besten eingereichten CV nennen und **Zustimmung abwarten**.
- `auto_submit: true` → nur, wenn die Bedingungen der Submit-Policy erfüllt sind.
- Budget prüfen: `python -m src.kaggle_api limits`.

## 3. Einreichen
```bash
python -m src.kaggle_api submit submissions/<name>.csv "<name> | CV <oof_auc>"
```
Wartet bis zu 10 Min. auf den Score; später: `python -m src.kaggle_api submissions`.

## 4. Protokollieren
- Public LB in Spalte `Public LB` der passenden Zeile in `EXPERIMENTS.md`.
- `.context/state.md` (beste eingereichte CV, Submissions übrig) aktualisieren.
- Weicht LB um > 0.002 von der CV ab → Learning eintragen; dem **CV** weiter vertrauen.

## Fehlerbilder
- `401/403` → Token fehlt/abgelaufen oder Regeln nicht akzeptiert.
- `HTTP 404` bei `StartSubmissionUpload` → API geändert; Referenz ist `kaggle_api_extended.py::competition_submit`
  im pip-Paket `kaggle`.
