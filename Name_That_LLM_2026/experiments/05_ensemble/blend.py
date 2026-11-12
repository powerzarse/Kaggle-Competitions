"""Experiment 05: Mischgewichte für das Ensemble aus Vorhersagen auf train.csv bestimmen.

Jedes Mitglied ist eine Parquet-Datei mit Spalte `id` und 18 Modellspalten:
- Wahrscheinlichkeiten (BERT, LoRA):   name=pfad
- SVM-Scores (decision_function):     name=pfad:svm   -> softmax(Score * T), T wird mitgesucht

Mischung: gewichteter Mittelwert der Wahrscheinlichkeiten (wie im Abgabe-Notebook).
Die Gewichte werden auf 4 Folds gesucht und auf dem 5. bewertet ("ehrlich"); daneben steht der
optimistische Wert, bei dem Gewichte und Bewertung dieselben Texte nutzen.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python experiments/05_ensemble/blend.py \\
        bert=models/07_kaggle/bert_stage2_oof.parquet svm=models/07_kaggle/svm_oof_decision.parquet:svm
"""
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

from src.cv import get_folds
from src.data import load_train

TEMPS = (3.0, 5.0, 10.0)
STEP = 0.1


def simplex(n, step=STEP):
    ticks = np.round(np.arange(0, 1 + 1e-9, step), 6)
    return [w for w in product(ticks, repeat=n) if abs(sum(w) - 1) < 1e-6]


def main(args):
    real = load_train()
    classes = np.sort(real["model"].unique())
    y = real["model"].values
    folds = get_folds(real)
    members = {}
    for a in args:
        name, spec = a.split("=", 1)
        path, kind = (spec[:-4], "svm") if spec.endswith(":svm") else (spec, "proba")
        m = pd.read_parquet(path).set_index("id").loc[real["id"], classes].values
        members[name] = (m, kind)
    names = list(members)
    has_svm = any(k == "svm" for _, k in members.values())

    def probs(T):
        return [softmax(m * T, 1) if k == "svm" else m for m, k in members.values()]

    def acc(P, idx):
        return balanced_accuracy_score(y[idx], classes[P[idx].argmax(1)])

    print("Einzelmodelle:")
    for n, (m, _) in members.items():
        print(f"  {n:<8} {acc(m, np.arange(len(y))):.4f}")

    grid = [(T, w) for T in (TEMPS if has_svm else (1.0,)) for w in simplex(len(names))]
    stacks = {T: np.stack(probs(T)) for T in {g[0] for g in grid}}
    blend = lambda T, w: np.tensordot(np.array(w), stacks[T], axes=1)

    all_idx = np.arange(len(y))
    best = max(grid, key=lambda g: acc(blend(*g), all_idx))
    honest = np.zeros((len(y), len(classes)))
    for k in range(folds.max() + 1):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        g = max(grid, key=lambda g: acc(blend(*g), tr))
        honest[va] = blend(*g)[va]
    print(f"\nBeste Mischung: T={best[0]:g} | " + " ".join(f"{n}={w:.1f}" for n, w in zip(names, best[1])))
    print(f"  optimistisch (Gewichte auf denselben Texten): {acc(blend(*best), all_idx):.4f}")
    print(f"  ehrlich (Gewichte je Fold auf den anderen 4):  {acc(honest, all_idx):.4f}")
    top = sorted(grid, key=lambda g: -acc(blend(*g), all_idx))[:8]
    print("Top-Kombinationen:")
    for T, w in top:
        print(f"  T={T:<4g} " + " ".join(f"{n}={x:.1f}" for n, x in zip(names, w)) + f"  -> {acc(blend(T, w), all_idx):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
