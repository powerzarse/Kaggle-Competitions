"""Kreuzvalidierung: GroupKFold nach prompt_id (kein Prompt in Train UND Valid).

Speichert Out-of-Fold-Wahrscheinlichkeiten, damit Modelle später gestackt /
ensembled werden können.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from src.data import GROUP_COL, TARGET, TEXT_COL, model_to_family

N_SPLITS = 5
# Wettbewerbsmetrik: Balanced Accuracy (Mittel der Recalls je Modell, K=18).
# Für GridSearchCV & Co.: scoring=SCORING
PRIMARY_METRIC = "balanced_accuracy"
SCORING = "balanced_accuracy"


def get_folds(df: pd.DataFrame, n_splits: int = N_SPLITS) -> np.ndarray:
    """Fold-Index je Zeile (deterministisch, für alle Modelle identisch)."""
    folds = np.full(len(df), -1)
    for k, (_, va) in enumerate(GroupKFold(n_splits).split(df, groups=df[GROUP_COL])):
        folds[va] = k
    return folds


def evaluate(y_true, proba: np.ndarray, classes) -> dict:
    classes = np.asarray(classes)
    y_pred = classes[proba.argmax(1)]
    fam_true = pd.Series(y_true).map(model_to_family)
    fam_pred = pd.Series(y_pred).map(model_to_family)
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, proba, labels=classes),
        "family_accuracy": accuracy_score(fam_true, fam_pred),
    }


def run_cv(model, df: pd.DataFrame, folds: np.ndarray | None = None, verbose: bool = True):
    """Trainiert `model` je Fold, gibt (oof_proba, classes, metrics) zurück."""
    folds = get_folds(df) if folds is None else folds
    classes = np.sort(df[TARGET].unique())
    oof = np.zeros((len(df), len(classes)))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        m = clone(model).fit(df.loc[tr, TEXT_COL], df.loc[tr, TARGET])
        # Spalten auf globale Klassenreihenfolge ausrichten
        idx = np.searchsorted(classes, m.classes_)
        oof[np.ix_(va, idx)] = m.predict_proba(df.loc[va, TEXT_COL])
        if verbose:
            bacc = balanced_accuracy_score(df.loc[va, TARGET], classes[oof[va].argmax(1)])
            print(f"Fold {k}: balanced_acc={bacc:.4f}")
    metrics = evaluate(df[TARGET].values, oof, classes)
    if verbose:
        print({k: round(v, 4) for k, v in metrics.items()})
    return oof, classes, metrics


def save_oof(name: str, df: pd.DataFrame, oof: np.ndarray, classes, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(oof, columns=classes)
    out.insert(0, "id", df["id"].values)
    out.to_parquet(out_dir / f"oof_{name}.parquet", index=False)
