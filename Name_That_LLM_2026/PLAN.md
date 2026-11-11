# Vorgehen: Name That LLM

## Ausgangslage (aus `notebooks/01_EDA.ipynb`)
- 2699 Texte, 18 Modelle aus 6 Familien (Claude 5, GPT 5, Gemini 4, GLM 2, Qwen 1, Kimi 1), balanciert mit ca. 150 Texten je Modell
- 150 Prompts, jeder von (fast) allen 18 Modellen beantwortet
- Kurze Texte (36–274 Wörter, ein Absatz, kein Markdown)
- Starke Signale für die **Familie** über Typografie (— vs. kein Gedankenstrich, ’ vs. ')
- **Familie wird zu ~68 % richtig erkannt, das konkrete Modell nur zu ~33 %.** Die Schwierigkeit liegt innerhalb der Familien.

## Validierung
- `GroupKFold(5)` nach `prompt_id` (`src/cv.py`). Gleiche Folds für alle Modelle, damit OOF-Vorhersagen ensemblebar sind.
- **Wettbewerbsmetrik: Balanced Accuracy** (Mittel der Recalls über alle 18 Modelle). Im Test kommen die Modelle unterschiedlich oft vor, und die Verteilung unterscheidet sich zwischen Public und Private LB.
  - Folge: Jedes Modell zählt gleich viel. Ein schwaches Modell (z. B. `claude-fable-5`, Recall 12 %) kostet genauso viel wie ein starkes. Ziel ist also, die **schwächsten Klassen** zu heben.
  - Tuning immer mit `scoring="balanced_accuracy"` (`src.cv.SCORING`)
  - Keine Annahmen über die Klassenhäufigkeit im Test (keine Priors aus dem Test schätzen)
- Zusätzlich beobachtet: Accuracy, Macro-F1, Log-Loss, Familien-Accuracy
- Für jedes Experiment werden die OOF-Wahrscheinlichkeiten nach `models/<experiment>/` gespeichert.
- Entscheidungen fallen anhand der CV und nicht anhand des Public Leaderboards (der Testdatensatz ist klein bzw. versteckt).

## Erste CV-Ergebnisse (GroupKFold 5, OOF in `models/`)
| Modell | **Balanced Acc.** | + Klassen-Bias (CV) | Macro-F1 | Log-Loss | Familien-Acc. | Laufzeit |
|---|---|---|---|---|---|---|
| stylo_gbm (nur Stilometrie) | 0.222 | 0.225 | 0.220 | 3.89 | 0.578 | 2 min |
| tfidf_logreg ohne Stilo | 0.329 | 0.321 | 0.324 | 2.20 | 0.684 | 1 min |
| **tfidf_logreg + Stilo** | **0.352** | 0.358 | **0.337** | **2.00** | 0.672 | 3 min |
| tfidf_svm + Stilo (kalibriert) | 0.347 | 0.342 | 0.309 | 2.09 | 0.670 | ~11 h ⚠️ |

Der Train ist balanciert, daher ist Balanced Accuracy dort ≈ Accuracy. Der Klassen-Bias (`src/postprocess.py`) bringt bisher nichts Verlässliches (±0,6 Pp.) und wird erst beim Ensemble erneut geprüft.

Zufallsniveau: 1/18 ≈ 0.056. Die SVM konvergierte mit den Stilo-Features nicht und läuft jetzt standardmäßig ohne sie.

## Modelle / Experimente (`experiments/`)
| # | Ansatz | Warum | Status |
|---|---|---|---|
| 01 | TF-IDF (Wort + Char) + LogReg / SVM | starke, schnelle Baseline für Stil und Wortwahl | ✅ Funktionen fertig |
| 02 | Stilometrie + Gradient Boosting | Typografie und Satzbau, interpretierbar | ✅ Funktionen fertig |
| 03 | Hierarchisch: Familie → Modell | Spezialisten für die schwierigen Familien Claude und GPT | ❌ getestet: 0.351 (flach 0.352), kein Gewinn mit TF-IDF-Spezialisten |
| 04 | Transformer-Finetuning (ModernBERT, DeBERTa-v3) | lernt subtile Stilunterschiede | 🔄 Fold 0: 0.357 (TF-IDF auf Fold 0: 0.374), Blend 0.383–0.391 |
| 05 | Ensemble / Stacking + Klassen-Bias | kombiniert die Stärken der Ansätze | geplant |

## Reihenfolge
1. Baseline tunen (01): n-Gramm-Bereiche, C, `max_features`, SVM vs. LogReg
2. Stilo-Features erweitern (02): Satzanfänge, Funktionswörter, Wortlängenverteilung, häufige LLM-Phrasen
3. ~~Hierarchisches Modell (03)~~: getestet, kein Gewinn. Der Engpass ist das Signal *innerhalb* der Familie, nicht die Struktur. Wiederaufnahme mit Transformer-Spezialisten.
4. Transformer (04) auf denselben Folds; zuerst `deberta-v3-base`
5. Ensemble (05): Gewichte auf den OOF-Vorhersagen optimieren. Klassen-Bias für Balanced Accuracy mit `cv_bias_gain` prüfen.

## Code (`src/`)
- `data.py`: Laden der Daten, `model_to_family`
- `features.py`: `stylometric_features`, `StyloTransformer`, TF-IDF-Bausteine
- `models.py`: Modell-Factory (`get_model("tfidf_logreg")`)
- `cv.py`: `get_folds`, `run_cv` (OOF), `evaluate`, `save_oof`
- `postprocess.py`: Bias je Klasse zur Optimierung der Balanced Accuracy (`fit_class_bias`, `cv_bias_gain`)
- `hierarchical.py`: `HierarchicalClassifier` (Gruppe → Modell, beliebige Estimator)
- `submission.py`: `make_submission` (Parquet, Spalten id/model)
