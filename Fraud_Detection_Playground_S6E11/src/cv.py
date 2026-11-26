"""Feste Stratified-K-Fold-Splits für ALLE Experimente (Voraussetzung für sauberes Blending)."""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import DATA_DIR, ID, N_FOLDS, SEED, TARGET

FOLD_FILE = DATA_DIR / "folds.csv"


def get_folds(train: pd.DataFrame) -> np.ndarray:
    """Fold-Index je Zeile; wird einmal erzeugt und dann aus data/folds.csv gelesen."""
    if FOLD_FILE.exists():
        f = pd.read_csv(FOLD_FILE)
        assert len(f) == len(train) and (f[ID].values == train[ID].values).all(), "folds.csv passt nicht"
        return f["fold"].values
    fold = np.zeros(len(train), dtype=int)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(skf.split(train, train[TARGET])):
        fold[va] = k
    pd.DataFrame({ID: train[ID], "fold": fold}).to_csv(FOLD_FILE, index=False)
    return fold


# fit_predict(X_tr, y_tr, X_va, y_va, X_te) -> (pred_va, pred_te)
FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
                      tuple[np.ndarray, np.ndarray]]


def run_cv(fit_predict: FitPredict, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
           folds: np.ndarray, verbose: bool = True) -> dict:
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    scores = []
    for k in range(folds.max() + 1):
        tr, va = folds != k, folds == k
        p_va, p_te = fit_predict(X[tr], y[tr], X[va], y[va], X_test)
        oof[va] = p_va
        test_pred += p_te / (folds.max() + 1)
        scores.append(roc_auc_score(y[va], p_va))
        if verbose:
            print(f"fold {k}: AUC {scores[-1]:.5f}", flush=True)
    cv = roc_auc_score(y, oof)
    if verbose:
        print(f"OOF AUC {cv:.5f} | mean {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return {"oof": oof, "test": test_pred, "fold_scores": scores, "cv": cv}
