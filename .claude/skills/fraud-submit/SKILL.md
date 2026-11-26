---
name: fraud-submit
description: Prüft eine Submission-CSV der Kaggle-Competition playground-series-s6e11 (Credit Card Fraud), reicht sie nach Bestätigung des Nutzers bei Kaggle ein und trägt den Public-LB-Score in EXPERIMENTS.md nach. Verwenden bei "submit", "einreichen", "auf Kaggle hochladen", "wie ist der LB-Score".
---

# Submission (S6E11)

Arbeitsordner: `Fraud_Detection_Playground_S6E11/`. Limit: **10 Submissions pro Tag**,
Deadline **30.11.2026 23:59 UTC**. Am Ende werden 2 Submissions für den Private LB gewählt.

## 1. Datei validieren (immer, vor jedem Upload)
```bash
python - <<'EOF'
import pandas as pd, sys
f = "submissions/<name>.csv"
s, ss = pd.read_csv(f), pd.read_csv("data/sample_submission.csv")
assert list(s.columns) == ["id", "is_fraud"], s.columns
assert len(s) == len(ss) == 299984 and (s.id.values == ss.id.values).all(), "IDs/Länge falsch"
assert s.is_fraud.notna().all() and s.is_fraud.nunique() > 1000, "NaNs oder konstant"
print("ok", s.is_fraud.describe().to_dict())
EOF
```

## 2. Nutzer fragen
Einreichen ist nach außen sichtbar und verbraucht ein Tageskontingent → **vor dem Upload**
Datei, OOF-AUC (aus `EXPERIMENTS.md`) und Beschreibung nennen und Bestätigung einholen.

## 3. Einreichen
```bash
python -m src.kaggle_api submit submissions/<name>.csv "<name> | CV <oof_auc>"
```
Wartet bis zu 5 Min. auf den Score. Danach / später:
```bash
python -m src.kaggle_api submissions
```

## 4. Protokollieren
Public-LB-Wert in der Spalte `Public LB` der passenden Zeile in `EXPERIMENTS.md` eintragen.
Wenn CV und LB auseinanderlaufen: der **CV vertrauen** (Public LB = nur ~20 % des Tests);
Abweichung im Notizfeld vermerken.

## Fehlerbilder
- `401/403`: Regeln nicht akzeptiert oder keine Zugangsdaten (lokal: `~/.kaggle/kaggle.json`).
- Upload-Endpunkt geändert: als Fallback `pip install kaggle` und
  `kaggle competitions submit -c playground-series-s6e11 -f <csv> -m "<msg>"` (braucht lokale Credentials).
