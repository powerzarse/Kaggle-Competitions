"""Experiment 06b: LinearSVC auf Open Pool (+ echte train.csv).

Teil A – Skalierung: nur Pool trainieren, auf der kompletten train.csv testen.
Teil B – Kombination (nur bei der größten Poolgröße): Pool + train.csv-Folds trainieren,
         GroupKFold-OOF auf train.csv (dieselben Folds wie alle anderen Experimente).

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python experiments/06_external_data/run_fast.py
    .venv/bin/python experiments/06_external_data/run_fast.py --per-class 1000 3000 --real-weights 1 5
    .venv/bin/python experiments/06_external_data/run_fast.py --only-b --per-class 3000 --C 0.1

Laufzeit grob (Mac): je Fit ~36 s pro 1.000 Texte/Modell; Standard-Lauf ~45-60 min.
Ergebnisse: models/06_external_data/results_fast.csv + oof_*.parquet
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import MODEL_DIR
from src.cv import evaluate, get_folds, save_oof
from src.data import load_open_pool, load_train
from src.linear_large import LargeVectorizer, make_svm, sample_weights, svm_proba

OUT = MODEL_DIR / "06_external_data"


def fmt(m):
    return f"balanced_acc={m['balanced_accuracy']:.4f} | family_acc={m['family_accuracy']:.4f}"


def save_results(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results_fast.csv"
    new = pd.DataFrame(rows)
    if path.exists():  # bisherige Ergebnisse behalten, gleiche Konfiguration überschreiben
        keys = ["per_class", "setup", "C", "real_weight"]
        old = pd.read_csv(path)
        new = pd.concat([old, new]).drop_duplicates(keys, keep="last")
    new.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, nargs="+", default=[1000, 3000, 6000])
    ap.add_argument("--C", type=float, nargs="+", default=[0.05, 0.1])
    ap.add_argument("--real-weights", type=float, nargs="+", default=[1, 5])
    ap.add_argument("--kimi-high", action="store_true", help="kimi-k3 nur mit Reasoning 'high'")
    ap.add_argument("--only-b", action="store_true",
                    help="Teil A überspringen; Teil B mit dem ersten --C-Wert und der größten --per-class")
    args = ap.parse_args()

    real = load_train()
    y_real = real["model"].values
    folds = get_folds(real)
    rows = []
    t_all = time.time()

    per_classes = [max(args.per_class)] if args.only_b else sorted(args.per_class)
    for pc in per_classes:
        t = time.time()
        pool = load_open_pool(per_class=pc, kimi_reasoning_high_only=args.kimi_high)
        vec = LargeVectorizer()
        X_pool, X_real = vec.fit_transform(pool["text"]), vec.transform(real["text"])
        y_pool = pool["model"].values
        print(f"\n### {pc} Texte/Modell: {len(pool)} Pool-Texte, {X_pool.shape[1]} Features "
              f"(Vektorisierung {time.time() - t:.0f}s)", flush=True)

        # Teil A: nur Pool -> Test auf kompletter train.csv
        best_C, best = args.C[0], -1
        for C in ([] if args.only_b else args.C):
            t = time.time()
            clf = make_svm(C).fit(X_pool, y_pool)
            m = evaluate(y_real, svm_proba(clf, X_real), clf.classes_)
            rows.append({"per_class": pc, "setup": "nur Pool", "C": C, "real_weight": 0,
                         "seconds": time.time() - t, **m})
            save_results(rows)
            print(f"  A nur Pool      C={C:<5g}: {fmt(m)} ({time.time() - t:.0f}s)", flush=True)
            if m["balanced_accuracy"] > best:
                best_C, best = C, m["balanced_accuracy"]

        if pc != max(args.per_class):
            continue

        # Teil B: Pool + train.csv-Folds (GroupKFold auf train.csv), bestes C aus A
        for rw in args.real_weights:
            t = time.time()
            oof = np.zeros((len(real), 18))
            for k in range(folds.max() + 1):
                tr, va = folds != k, folds == k
                X = sp.vstack([X_pool, X_real[tr]], format="csr")
                y = np.concatenate([y_pool, y_real[tr]])
                clf = make_svm(best_C).fit(X, y, sample_weight=sample_weights(len(y_pool), tr.sum(), rw))
                oof[va] = svm_proba(clf, X_real[va])
                print(f"    Fold {k} fertig ({time.time() - t:.0f}s)", flush=True)
            m = evaluate(y_real, oof, clf.classes_)
            rows.append({"per_class": pc, "setup": "Pool + train", "C": best_C, "real_weight": rw,
                         "seconds": time.time() - t, **m})
            save_results(rows)
            print(f"  B Pool + train  C={best_C:<5g} real_w={rw:g}: {fmt(m)} ({time.time() - t:.0f}s)", flush=True)
            save_oof(f"svm_pool{pc}_rw{rw:g}", real, oof, clf.classes_, OUT)

    res = pd.DataFrame(rows)
    print(f"\n=== Übersicht (Balanced Accuracy auf echter train.csv), {(time.time() - t_all) / 60:.0f} min ===")
    print("Referenz: nur train.csv (CV) 0.352 | Public LB 0.320")
    print(res[["per_class", "setup", "C", "real_weight", "balanced_accuracy", "family_accuracy", "seconds"]]
          .sort_values("balanced_accuracy", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
