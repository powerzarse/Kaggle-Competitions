"""Experiment 03: hierarchisches Modell vs. flache Baseline.

Aufruf (aus Name_That_LLM_2026/):  .venv/bin/python experiments/03_hierarchical/run.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from src.config import MODEL_DIR
from src.cv import evaluate, get_folds, run_cv, save_oof
from src.data import load_train
from src.hierarchical import HierarchicalClassifier

OUT = MODEL_DIR / "03_hierarchical"


def load_oof(path):
    o = pd.read_parquet(path)
    cl = o.columns[1:].to_numpy()
    return o[cl].to_numpy(), cl


def recall_table(y, preds: dict, classes):
    return pd.DataFrame(
        {name: recall_score(y, p, labels=classes, average=None) for name, p in preds.items()},
        index=classes,
    ).round(3)


if __name__ == "__main__":
    df = load_train()
    folds = get_folds(df)
    y = df["model"].values

    results, preds = {}, {}
    flat, classes = load_oof(MODEL_DIR / "01_tfidf_linear" / "oof_tfidf_logreg.parquet")
    results["flat"] = evaluate(y, flat, classes)
    preds["flat"] = classes[flat.argmax(1)]

    for name, kw in [("hier_soft", {}), ("hier_hard", {"hard_routing": True})]:
        t = time.time()
        oof, cl, m = run_cv(HierarchicalClassifier(**kw), df, folds, verbose=True)
        assert (cl == classes).all()
        save_oof(name, df, oof, cl, OUT)
        results[name], preds[name] = m, cl[oof.argmax(1)]
        print(f"{name}: {time.time() - t:.0f}s")

    # Mischung flach + hierarchisch (geometrisches Mittel der Wahrscheinlichkeiten)
    hier, _ = load_oof(OUT / "oof_hier_soft.parquet")
    for w in (0.3, 0.5, 0.7):
        blend = np.exp(w * np.log(hier + 1e-9) + (1 - w) * np.log(flat + 1e-9))
        results[f"blend_w{w}"] = evaluate(y, blend, classes)
        preds[f"blend_w{w}"] = classes[blend.argmax(1)]

    print("\n=== Ergebnisse ===")
    print(pd.DataFrame(results).T.round(4).sort_values("balanced_accuracy", ascending=False))
    print("\n=== Recall je Modell ===")
    print(recall_table(y, preds, classes))
