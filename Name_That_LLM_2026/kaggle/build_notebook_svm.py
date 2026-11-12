"""Erzeugt kaggle/svm_pool/submission_svm_pool.ipynb – Abgabe: LinearSVC auf Open Pool + train.csv.

Feature- und Modellcode werden aus src/features.py und src/linear_large.py eingebettet,
damit lokale Experimente und Kaggle-Abgabe identisch rechnen.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_svm.py --per-class 6000 --real-weight 5 --C 0.1
Lokaler Test (kleiner Pool):
    cd kaggle/svm_pool && NTL_DATA_DIR=../../data NTL_POOL=../../data/external/open_pool/ntl_open_pool.parquet \
        ../../.venv/bin/jupyter nbconvert --to notebook --execute submission_svm_pool.ipynb --output test_run.ipynb
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle" / "svm_pool"


def strip_module(src: str) -> str:
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)       # Modul-Docstring
    src = re.sub(r"^from src\.[^\n]*\n", "", src, flags=re.M)            # interne Imports
    return src.strip()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def oof_cells(real_weight):
    import sys
    sys.path.insert(0, str(ROOT))
    from src.cv import get_folds
    from src.data import load_train
    df = load_train()
    fmap = dict(zip(df["prompt_id"], get_folds(df).tolist()))
    return [
        md("## 5-Fold-OOF auf train.csv (Pool + train-Folds)\n\n"
           "Nacheinander statt parallel: Ein Fit braucht ~14 GB RAM (liblinear kopiert die Matrix)."),
        code(f"""import time
import scipy.sparse as sp

FOLD_OF_PROMPT = {json.dumps(fmap)}
folds = train["prompt_id"].map(FOLD_OF_PROMPT).values

t0 = time.time()
vec = LargeVectorizer()
X_pool = vec.fit_transform(pool["text"])
X_train = vec.transform(train["text"])
y_pool, y_train = pool["model"].values, train["model"].values
print(f"Vektorisierung {{(time.time() - t0) / 60:.1f}} min, {{X_pool.shape[1]}} Merkmale", flush=True)

oof = None
for k in range(5):
    tr, va = folds != k, folds == k
    X = sp.vstack([X_pool, X_train[tr]], format="csr")
    y = np.concatenate([y_pool, y_train[tr]])
    clf = make_svm(C).fit(X, y, sample_weight=sample_weights(len(y_pool), tr.sum(), REAL_WEIGHT))
    del X
    if oof is None:
        oof = np.zeros((len(train), len(clf.classes_)), dtype=np.float32)
    oof[va] = clf.decision_function(X_train[va])
    acc = (clf.classes_[oof[va].argmax(1)] == y_train[va]).mean()
    print(f"Fold {{k}} fertig: acc={{acc:.4f}} | {{(time.time() - t0) / 60:.1f}} min", flush=True)
    out = pd.DataFrame(oof, columns=clf.classes_)
    out.insert(0, "id", train["id"].values)
    out.to_parquet(OUT_DIR / "svm_oof_decision.parquet", index=False)   # nach jedem Fold sichern"""),
        code("""from sklearn.metrics import balanced_accuracy_score
pred = clf.classes_[oof.argmax(1)]
print("OOF Balanced Accuracy:", round(balanced_accuracy_score(y_train, pred), 4))"""),
    ]


def build(per_class: int, real_weight: float, C: float, kimi_high: bool, oof: bool = False):
    features_src = strip_module((ROOT / "src" / "features.py").read_text())
    linear_src = strip_module((ROOT / "src" / "linear_large.py").read_text())

    cells = [
        md("# Name That LLM – LinearSVC auf Open Pool + train.csv\n\n"
           "Trainiert auf dem externen Datensatz *Name That LLM Open Pool* (crtproduction, CC BY-SA 4.0) plus `train.csv`, "
           "Features: Wort- + Char-TF-IDF + Stilometrie. Ausgabe: `submission.parquet`. Kein Internet nötig.\n\n"
           "**Benötigte Inputs:** Competition `name-that-llm` und Dataset `crtproduction/name-that-llm-open-pool`."),
        code(f"""# Parameter (aus experiments/06_external_data/run_fast.py)
PER_CLASS = {per_class or None}      # Pool-Texte je Modell (None = kompletter Pool)
REAL_WEIGHT = {real_weight}  # Gewicht der echten train.csv-Texte
C = {C}
KIMI_HIGH_ONLY = {kimi_high}
SEED = 42"""),
        code("""import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

def find(pattern, env, fallback):
    if os.environ.get(env):
        return Path(os.environ[env])
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return Path(hits[0]) if hits else Path(fallback)

KAGGLE_DIR = Path("/kaggle/input/competitions/name-that-llm")
if KAGGLE_DIR.exists():
    DATA_DIR = KAGGLE_DIR
elif os.environ.get("NTL_DATA_DIR"):
    DATA_DIR = Path(os.environ["NTL_DATA_DIR"])
else:
    DATA_DIR = find("train.csv", "_UNUSED", "../../data/train.csv").parent
POOL_PATH = find("ntl_open_pool.parquet", "NTL_POOL", "../../data/external/open_pool/ntl_open_pool.parquet")
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
print("DATA_DIR:", DATA_DIR, "| POOL:", POOL_PATH, "| OUT_DIR:", OUT_DIR)
assert (DATA_DIR / "train.csv").exists() and POOL_PATH.exists(), "Input fehlt – Open-Pool-Dataset eingebunden?\""""),
        md("## Features & Modell (aus `src/features.py` und `src/linear_large.py`)"),
        code(features_src),
        code(linear_src),
        md("## Daten"),
        code("""train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

pool = pd.read_parquet(POOL_PATH, columns=["text", "model", "gen_reasoning"])
if KIMI_HIGH_ONLY:
    pool = pool[(pool["model"] != "kimi-k3") | (pool["gen_reasoning"] == "high")]
if PER_CLASS:
    pool = pool.sample(frac=1, random_state=SEED).groupby("model").head(PER_CLASS)
assert set(pool["model"]) == set(train["model"])
print("train", train.shape, "| pool", pool.shape, "| test", test.shape)"""),
        md("## Training & Vorhersage"),
        code("""import scipy.sparse as sp

vec = LargeVectorizer()
X_pool = vec.fit_transform(pool["text"])
X_train = vec.transform(train["text"])
X = sp.vstack([X_pool, X_train], format="csr")
y = np.concatenate([pool["model"].values, train["model"].values])
w = sample_weights(len(pool), len(train), REAL_WEIGHT)

clf = make_svm(C).fit(X, y, sample_weight=w)
pred = clf.predict(vec.transform(test["text"]))"""),
        md("## Submission"),
        code("""sub = pd.DataFrame({"id": test["id"], "model": pred})

# Prüfungen = Ablehnungsgründe von Kaggle
assert list(sub.columns) == ["id", "model"]
assert sub.notna().all().all(), "Nullwerte"
assert sub["id"].is_unique, "doppelte IDs"
assert len(sub) == len(test) and set(sub["id"]) == set(test["id"]), "nicht jede Test-ID genau einmal"
assert train["model"].nunique() == 18 and set(sub["model"]) <= set(train["model"]), "unbekanntes Label"

sub.to_parquet(OUT_DIR / "submission.parquet", index=False)
print(sub["model"].value_counts().head(10))
sub.head()"""),
    ]
    out_dir, nb_name = OUT_DIR, "submission_svm_pool.ipynb"
    kernel_id, title = "linuszarse/name-that-llm-svm-open-pool", "Name That LLM SVM Open Pool"
    if oof:
        # OOF-Modus: statt Abgabe 5-Fold-OOF auf train.csv (Pool + train-Folds) für Ensemble-Gewichte
        train_idx = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "Training" in c["source"])
        cells = cells[:train_idx] + oof_cells(real_weight)
        out_dir, nb_name = ROOT / "kaggle" / "svm_oof", "svm_oof.ipynb"
        kernel_id, title = "linuszarse/name-that-llm-svm-oof", "Name That LLM SVM OOF"
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / nb_name).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": nb_name,
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["crtproduction/name-that-llm-open-pool"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [], "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"geschrieben: {out_dir}/{nb_name} (+ kernel-metadata.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6000, help="0 = kompletter Pool")
    ap.add_argument("--real-weight", type=float, default=5)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--kimi-high", action="store_true")
    ap.add_argument("--oof", action="store_true", help="statt Abgabe: 5-Fold-OOF auf train.csv (kaggle/svm_oof/)")
    a = ap.parse_args()
    build(a.per_class, a.real_weight, a.C, a.kimi_high, a.oof)
