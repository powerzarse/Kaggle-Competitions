# 01 – TF-IDF + lineare Modelle (Baseline)

- Features: Wort-TF-IDF (1–2-Gramme) + Char-TF-IDF (1–5-Gramme), `lowercase=False` (Groß-/Kleinschreibung und Typografie bleiben erhalten)
- Modelle: `tfidf_logreg`, `tfidf_svm` (kalibriert) aus `src/models.py`
- Tuning: C, ngram_range, min_df, max_features
- Output: OOF-Wahrscheinlichkeiten nach `models/01_tfidf_linear/`
