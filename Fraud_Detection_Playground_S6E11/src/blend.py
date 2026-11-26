"""Blending über gespeicherte OOF-Vorhersagen.

    python -m src.blend                     # alle Experimente, Hill Climbing
    python -m src.blend exp_a exp_b exp_c   # nur ausgewählte
"""
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .cv import get_folds
from .data import load_data
from .tracking import available_experiments, load_oof, save_experiment


def _rank(m: np.ndarray) -> np.ndarray:
    return np.column_stack([rankdata(m[:, j]) / len(m) for j in range(m.shape[1])])


def hill_climb(oof: np.ndarray, y: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Greedy Forward Selection mit Zurücklegen (Caruana); gibt Gewichte zurück."""
    n = oof.shape[1]
    counts = np.zeros(n)
    best_j = int(np.argmax([roc_auc_score(y, oof[:, j]) for j in range(n)]))
    counts[best_j] = 1
    best = roc_auc_score(y, oof[:, best_j])
    for _ in range(max_iter):
        cand = [roc_auc_score(y, oof @ (counts + np.eye(n)[j])) for j in range(n)]
        j = int(np.argmax(cand))
        if cand[j] <= best + 1e-6:
            break
        counts[j] += 1
        best = cand[j]
    return counts / counts.sum()


def main(names: list[str]) -> None:
    train, test = load_data()
    y = train["is_fraud"].values
    names = names or [n for n in available_experiments() if not n.startswith("blend")]
    oof, te = load_oof(names)
    oof_r, te_r = _rank(oof), _rank(te)
    for j, n in enumerate(names):
        print(f"{roc_auc_score(y, oof[:, j]):.5f}  {n}")
    print(f"{roc_auc_score(y, oof_r.mean(1)):.5f}  <rank mean>")
    w = hill_climb(oof_r, y)
    print("Gewichte:", {n: round(float(v), 3) for n, v in zip(names, w) if v > 0})
    folds = get_folds(train)
    res = {"oof": oof_r @ w, "test": te_r @ w, "cv": roc_auc_score(y, oof_r @ w),
           "fold_scores": [roc_auc_score(y[folds == k], (oof_r @ w)[folds == k])
                           for k in range(folds.max() + 1)]}
    print(f"{res['cv']:.5f}  <hill climb>")
    save_experiment("blend_hc", res, test["id"], "-", "rank-blend",
                    params={"weights": dict(zip(names, map(float, w)))},
                    note=f"{len(names)} Modelle")


if __name__ == "__main__":
    main(sys.argv[1:])
