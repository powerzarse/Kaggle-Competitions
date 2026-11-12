"""Lineare Modelle für große Trainingsmengen (Open Pool, 100k+ Texte).

Unterschiede zu src/models.py:
- Der Vektorisierer wird einmal gefittet (unüberwacht, nur auf Pool-Texten) und dann
  für alle Folds wiederverwendet -> kein teures Neu-Fitten je Fold.
- LinearSVC statt LogisticRegression/lbfgs.
- Stilo-Features mit Gewicht 0.1 angehängt: gleiche Genauigkeit, ~20x schnellere Konvergenz.

Vergleich auf 1.000 Pool-Texten je Modell, getestet auf der echten train.csv:
    LinearSVC, Stilo x0.1   0.400  36 s    <- Standard
    LinearSVC, Stilo x1.0   0.400  746 s
    LinearSVC, ohne Stilo   0.377  16 s
    LogReg (lbfgs) + Stilo  0.388  13 min
    SGD (log_loss, avg)     0.361  20 s
"""
import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.features import char_tfidf, stylometric_features, word_tfidf

STYLO_WEIGHT = 0.1
SVM_TEMPERATURE = 5.0  # Skalierung der SVM-Scores für Pseudo-Wahrscheinlichkeiten (softmax)


class LargeVectorizer:
    """Wort- + Char-TF-IDF (begrenztes Vokabular, float32) + gewichtete Stilo-Features."""

    def __init__(self, max_word: int = 300_000, max_char: int = 500_000,
                 stylo_weight: float = STYLO_WEIGHT):
        self.word = word_tfidf(min_df=3, max_features=max_word, dtype=np.float32)
        self.char = char_tfidf(min_df=5, max_features=max_char, dtype=np.float32)
        self.scaler = StandardScaler()
        self.stylo_weight = stylo_weight

    def _stack(self, texts, fit: bool):
        if fit:
            w, c = self.word.fit_transform(texts), self.char.fit_transform(texts)
            s = self.scaler.fit_transform(stylometric_features(texts))
        else:
            w, c = self.word.transform(texts), self.char.transform(texts)
            s = self.scaler.transform(stylometric_features(texts))
        parts = [w, c]
        if self.stylo_weight:
            parts.append(sp.csr_matrix(s * self.stylo_weight))
        return sp.hstack(parts, format="csr", dtype=np.float32)

    def fit_transform(self, texts):
        return self._stack(texts, fit=True)

    def transform(self, texts):
        return self._stack(texts, fit=False)


def make_svm(C: float = 0.1):
    return LinearSVC(C=C, class_weight="balanced")


def svm_proba(clf, X, temperature: float = SVM_TEMPERATURE) -> np.ndarray:
    """Softmax über die SVM-Scores (für Ensembles; Temperatur später auf OOF kalibrieren)."""
    return softmax(clf.decision_function(X) * temperature, axis=1)


def sample_weights(n_pool: int, n_real: int, real_weight: float) -> np.ndarray:
    """Gewicht je Zeile: Pool = 1, echte train.csv-Texte = real_weight."""
    return np.concatenate([np.ones(n_pool), np.full(n_real, real_weight)]).astype(np.float32)
