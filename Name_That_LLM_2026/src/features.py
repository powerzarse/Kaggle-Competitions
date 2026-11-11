"""Stilometrische Features + TF-IDF-Bausteine.

Die EDA zeigt: Typografie (Gedankenstriche, typografische vs. gerade Anführungszeichen,
Semikolons) trennt die Modellfamilien sehr gut. Daher werden diese Zeichen explizit
gezählt und Tokenizer arbeiten mit lowercase=False.
"""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

# Zeichen, deren Häufigkeit (pro 100 Wörter) als Feature dient
CHAR_COUNTS = {
    "em_dash": "—",
    "en_dash": "–",
    "hyphen": "-",
    "curly_apos": "’",
    "curly_dquote": "“",
    "straight_apos": "'",
    "straight_dquote": '"',
    "colon": ":",
    "semicolon": ";",
    "comma": ",",
    "paren": "(",
    "question": "?",
    "exclam": "!",
    "ellipsis": "…",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b\w+\b")


def stylometric_features(texts: pd.Series) -> pd.DataFrame:
    """Längen-, Satz-, Wort- und Interpunktionsstatistiken je Text."""
    rows = []
    for text in texts:
        words = _WORD.findall(text)
        n_words = max(len(words), 1)
        sents = [s for s in _SENT_SPLIT.split(text.strip()) if s]
        sent_lens = [len(_WORD.findall(s)) for s in sents] or [0]
        row = {
            "n_chars": len(text),
            "n_words": len(words),
            "n_sents": len(sents),
            "n_paras": text.count("\n\n") + 1,
            "words_per_sent": np.mean(sent_lens),
            "sent_len_std": np.std(sent_lens),
            "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
            "type_token_ratio": len({w.lower() for w in words}) / n_words,
            "upper_ratio": sum(c.isupper() for c in text) / max(len(text), 1),
            "non_ascii_ratio": sum(ord(c) > 127 for c in text) / max(len(text), 1),
            "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
            "ends_with_period": float(text.rstrip().endswith(".")),
        }
        for name, ch in CHAR_COUNTS.items():
            row[f"per100w_{name}"] = 100 * text.count(ch) / n_words
        rows.append(row)
    return pd.DataFrame(rows, index=texts.index)


class StyloTransformer(BaseEstimator, TransformerMixin):
    """sklearn-Wrapper um stylometric_features (für Pipelines)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return stylometric_features(pd.Series(X)).to_numpy(dtype=float)


def word_tfidf(**kw) -> TfidfVectorizer:
    params = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False)
    return TfidfVectorizer(**{**params, **kw})


def char_tfidf(**kw) -> TfidfVectorizer:
    params = dict(analyzer="char", ngram_range=(1, 5), min_df=3, sublinear_tf=True, lowercase=False)
    return TfidfVectorizer(**{**params, **kw})


def text_features(use_stylo: bool = True):
    """FeatureUnion aus Wort-TF-IDF, Char-TF-IDF und (optional) skalierten Stilo-Features."""
    parts = [word_tfidf(), char_tfidf()]
    if use_stylo:
        parts.append(make_pipeline(StyloTransformer(), StandardScaler()))
    return make_union(*parts)
