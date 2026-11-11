# 05 – Ensemble & Post-Processing

Ziel: **Balanced Accuracy** maximieren (Mittel der Recalls je Modell).

- Gewichtetes Mittel bzw. Stacking (LogReg) auf den OOF-Wahrscheinlichkeiten aller Experimente
- Gewichte nur auf OOF optimieren (gleiche Folds) und mit `balanced_accuracy` bewerten, nicht auf dem Public LB
- Klassen-Bias (`src/postprocess.py`): additiver Bias je Modell auf log(proba), der den mittleren Recall maximiert. Nur übernehmen, wenn `cv_bias_gain` auf ungesehenen Folds einen stabilen Gewinn zeigt.
- ~~Hungarian-Matching je Prompt~~: verworfen. Laut Evaluation-Seite ist die Anzahl Texte je Modell im Test **nicht** gleich, also nicht „ein Text je Modell und Prompt“.
