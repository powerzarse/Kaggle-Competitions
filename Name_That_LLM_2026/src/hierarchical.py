"""Hierarchischer Klassifikator: erst Gruppe (Familie), dann Modell innerhalb der Gruppe.

P(modell | x) = P(gruppe | x) * P(modell | gruppe, x)

Die Gruppen orientieren sich an den Verwechslungen aus der EDA (nicht nur an
Familiennamen): Claude und GLM teilen Typografie (—, gerade '), ebenso GPT, Kimi und Qwen.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from src.data import model_to_family
from src.models import tfidf_logreg

# Familie -> Gruppe
DEFAULT_GROUPS = {
    "claude": "claude_glm",
    "glm": "claude_glm",
    "gpt": "gpt_kimi_qwen",
    "kimi": "gpt_kimi_qwen",
    "qwen": "gpt_kimi_qwen",
    "gemini": "gemini",
}


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """
    group_model:  Estimator für die Gruppe (Stufe 1)
    specialist:   Estimator-Vorlage für jede Gruppe (Stufe 2), wird je Gruppe geklont
    groups:       dict Familie -> Gruppe
    hard_routing: True = nur die wahrscheinlichste Gruppe nutzen (sonst weich multiplizieren)
    """

    def __init__(self, group_model=None, specialist=None, groups=None, hard_routing=False):
        self.group_model = group_model
        self.specialist = specialist
        self.groups = groups
        self.hard_routing = hard_routing

    def _group_of(self, y):
        groups = self.groups or DEFAULT_GROUPS
        return np.array([groups[model_to_family(m)] for m in y])

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=object), np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        g = self._group_of(y)

        self.group_model_ = clone(self.group_model or tfidf_logreg()).fit(X, g)
        self.specialists_ = {}
        for grp in self.group_model_.classes_:
            mask = g == grp
            if len(np.unique(y[mask])) == 1:  # Gruppe mit nur einem Modell
                self.specialists_[grp] = y[mask][0]
            else:
                self.specialists_[grp] = clone(self.specialist or tfidf_logreg()).fit(X[mask], y[mask])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=object)
        p_group = self.group_model_.predict_proba(X)
        if self.hard_routing:
            p_group = np.eye(p_group.shape[1])[p_group.argmax(1)]
        proba = np.zeros((len(X), len(self.classes_)))
        for j, grp in enumerate(self.group_model_.classes_):
            spec = self.specialists_[grp]
            if isinstance(spec, str):
                proba[:, np.searchsorted(self.classes_, spec)] += p_group[:, j]
                continue
            idx = np.searchsorted(self.classes_, spec.classes_)
            proba[:, idx] += p_group[:, [j]] * spec.predict_proba(X)
        return proba

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
