# 05 – Ensemble & Post-Processing

Ziel: **Balanced Accuracy** maximieren (Mittel der Recalls je Modell).

- Gewichtetes Mittel bzw. Stacking (LogReg) auf den OOF-Wahrscheinlichkeiten aller Experimente
- Gewichte nur auf OOF optimieren (gleiche Folds) und mit `balanced_accuracy` bewerten, nicht auf dem Public LB
- Klassen-Bias (`src/postprocess.py`): additiver Bias je Modell auf log(proba), der den mittleren Recall maximiert. Nur übernehmen, wenn `cv_bias_gain` auf ungesehenen Folds einen stabilen Gewinn zeigt.
- ~~Hungarian-Matching je Prompt~~: verworfen. Laut Evaluation-Seite ist die Anzahl Texte je Modell im Test **nicht** gleich, also nicht „ein Text je Modell und Prompt“.

## Skript
`blend.py` bestimmt die Mischgewichte aus Vorhersagen auf train.csv (Parquet je Mitglied, in `models/07_kaggle/`).
Es gibt einen optimistischen und einen ehrlichen Wert aus (Gewichte je Fold auf den anderen 4 Folds gesucht).

| Mitglieder | Beste Mischung | ehrlich |
|---|---|---|
| ModernBERT Stufe 2 (0.583) + SVM kompletter Pool (0.502) | T=5, BERT 0.6 / SVM 0.4 | 0.596 (LB 0.565) |
| + Qwen3-4B-LoRA (0.525) | T=5, BERT 0.4 / SVM 0.4 / LoRA 0.2 | 0.603 |

Das Abgabe-Notebook (`kaggle/build_notebook_ensemble.py`) nimmt die Gewichte als Parameter; `--w-lora` schaltet die Qwen-LoRA als drittes Mitglied zu.
