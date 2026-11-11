"""Modell-Factory. Jede Funktion liefert ein unfit sklearn-Estimator mit predict_proba."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import SEED
from src.features import StyloTransformer, text_features


def tfidf_logreg(C: float = 10, use_stylo: bool = True):
    return make_pipeline(text_features(use_stylo), LogisticRegression(C=C, max_iter=5000))


def tfidf_svm(C: float = 0.5, use_stylo: bool = False):
    # Kalibriert, damit predict_proba für Ensembles verfügbar ist.
    # use_stylo=False als Default: mit den dichten Stilo-Spalten konvergierte
    # LinearSVC nicht und der CV-Lauf dauerte ~11 h.
    svm = CalibratedClassifierCV(LinearSVC(C=C, max_iter=5000), cv=3, method="sigmoid")
    return make_pipeline(text_features(use_stylo), svm)


def stylo_gbm():
    """Nur stilometrische Features + Gradient Boosting (schnell, interpretierbar)."""
    return make_pipeline(
        StyloTransformer(),
        StandardScaler(),
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED),
    )


MODEL_REGISTRY = {
    "tfidf_logreg": tfidf_logreg,
    "tfidf_svm": tfidf_svm,
    "stylo_gbm": stylo_gbm,
}


def get_model(name: str, **kw):
    return MODEL_REGISTRY[name](**kw)
