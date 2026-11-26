# Arbeitsregeln – S6E11 Credit Card Fraud

- Sprache: Deutsch. Arbeitsordner für alle Befehle: `Fraud_Detection_Playground_S6E11/`.
- Für Daten, Experimente, Blending und Submissions die Skills `fraud-data`, `fraud-experiment`,
  `fraud-blend`, `fraud-submit` verwenden.
- Validierung: ausschließlich die festen Folds aus `src.cv.get_folds()`; Zielgröße ist die OOF-AUC.
- Jedes Experiment in eigenem Ordner `experiments/NN_<name>/`, Ergebnisse nur über
  `src.tracking.save_experiment()` speichern. Bestehende Feature-Sets nicht verändern, neue anlegen.
- Nie Daten, Modelle oder Submissions committen (siehe `.gitignore`); `EXPERIMENTS.md` schon.
- Kaggle-Submissions nur nach ausdrücklicher Bestätigung des Nutzers (Tageslimit 10).
