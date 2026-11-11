"""Erzeugt kaggle/submission_tfidf.ipynb – ein eigenständiges Kaggle-Abgabe-Notebook.

Der Feature-Code wird aus src/features.py eingebettet, damit lokales Training und
Kaggle-Abgabe exakt dieselben Features nutzen.

Aufruf (aus Name_That_LLM_2026/):  .venv/bin/python kaggle/build_notebook.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "submission_tfidf.ipynb"

features_src = (ROOT / "src" / "features.py").read_text()
# Modul-Docstring entfernen, Rest unverändert übernehmen
features_src = re.sub(r'^""".*?"""\n', "", features_src, count=1, flags=re.S).strip()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md("# Name That LLM – Abgabe: TF-IDF + Stilometrie + Logistische Regression\n\n"
       "CV (GroupKFold nach `prompt_id`): **Balanced Accuracy ≈ 0.352**. "
       "Das Notebook trainiert auf `train.csv` und sagt `test.csv` vorher. Interaktiv enthält `test.csv` nur eine "
       "Beispielzeile; beim Einreichen läuft das Notebook erneut auf dem versteckten Test (~4000 Texte). "
       "Ausgabe: `submission.parquet` im Arbeitsverzeichnis. Internet wird nicht benötigt."),
    code("""import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Kaggle-Pfad laut "Submission File"-Seite; Fallback: Suche unter /kaggle/input, lokal ../data
KAGGLE_DIR = Path("/kaggle/input/competitions/name-that-llm")
if KAGGLE_DIR.exists():
    DATA_DIR = KAGGLE_DIR
else:
    candidates = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    DATA_DIR = Path(candidates[0]).parent if candidates else Path(os.environ.get("NTL_DATA_DIR", "../data"))
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
print("DATA_DIR:", DATA_DIR, "| OUT_DIR:", OUT_DIR)
print(sorted(p.name for p in DATA_DIR.iterdir()))"""),
    md("## Features (aus `src/features.py`)"),
    code(features_src),
    md("## Training & Vorhersage"),
    code("""from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print("train", train.shape, "| test", test.shape)

model = make_pipeline(text_features(use_stylo=True), LogisticRegression(C=10, max_iter=5000))
model.fit(train["text"], train["model"])
pred = model.predict(test["text"])"""),
    md("## Submission\n`submission.parquet` mit den Spalten `id`, `model`. Die Prüfungen unten entsprechen den "
       "Ablehnungsgründen von Kaggle: fehlende Spalten, Nullwerte, doppelte IDs, unbekannte Labels."),
    code("""sub = pd.DataFrame({"id": test["id"], "model": pred})

# Prüfungen = Ablehnungsgründe von Kaggle
assert list(sub.columns) == ["id", "model"]
assert sub.notna().all().all(), "Nullwerte in der Submission"
assert sub["id"].is_unique, "doppelte IDs"
assert len(sub) == len(test) and set(sub["id"]) == set(test["id"]), "nicht jede Test-ID genau einmal"
assert train["model"].nunique() == 18 and set(sub["model"]) <= set(train["model"]), "unbekanntes Label"

sub.to_parquet(OUT_DIR / "submission.parquet", index=False)
print(sub["model"].value_counts().head(10))
sub.head()"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"geschrieben: {OUT}")
