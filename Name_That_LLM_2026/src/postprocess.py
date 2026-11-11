"""Post-Processing für Balanced Accuracy.

Balanced Accuracy = Mittel der Recalls je Klasse. Ein Modell, das manche Klassen
selten vorhersagt (z.B. claude-fable-5), verliert dort viel Recall. Ein additiver
Bias je Klasse auf log(proba) verschiebt die Entscheidungsgrenzen so, dass der
mittlere Recall maximal wird. Unabhängig von der Klassenverteilung im Test, da
Balanced Accuracy jede Klasse gleich gewichtet.

Bias immer auf OOF-Vorhersagen fitten und mit `cv_bias_gain` prüfen, dass er
auch auf ungesehenen Folds hilft (sonst Overfitting).
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def _score(y_idx, logp, bias):
    return balanced_accuracy_score(y_idx, (logp + bias).argmax(1))


def fit_class_bias(proba: np.ndarray, y_true, classes, n_rounds: int = 3,
                   grid=np.linspace(-1.5, 1.5, 31)) -> np.ndarray:
    """Koordinatenweise Suche eines Bias-Vektors (eine Zahl je Klasse)."""
    classes = np.asarray(classes)
    y_idx = np.searchsorted(classes, np.asarray(y_true))
    logp = np.log(np.clip(proba, 1e-9, 1))
    bias = np.zeros(len(classes))
    best = _score(y_idx, logp, bias)
    for _ in range(n_rounds):
        improved = False
        for c in range(len(classes)):
            for b in grid:
                trial = bias.copy()
                trial[c] = b
                s = _score(y_idx, logp, trial)
                if s > best + 1e-9:
                    best, bias, improved = s, trial, True
        if not improved:
            break
    return bias


def apply_class_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Gibt angepasste (unnormierte) Scores zurück; argmax = Vorhersage."""
    return np.log(np.clip(proba, 1e-9, 1)) + bias


def cv_bias_gain(proba: np.ndarray, y_true, classes, folds: np.ndarray) -> dict:
    """Ehrliche Schätzung: Bias auf k-1 Folds fitten, auf Fold k anwenden."""
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)
    adjusted = np.zeros_like(proba)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        bias = fit_class_bias(proba[tr], y_true[tr], classes)
        adjusted[va] = apply_class_bias(proba[va], bias)
    before = balanced_accuracy_score(y_true, classes[proba.argmax(1)])
    after = balanced_accuracy_score(y_true, classes[adjusted.argmax(1)])
    return {"before": before, "after": after, "gain": after - before}
