# 04 – Transformer-Finetuning

Skript: `train.py` (aus dem Ordner `Name_That_LLM_2026/` starten)

```bash
# 1) Funktionstest (~1 min)
.venv/bin/python experiments/04_transformer/train.py --smoke

# 2) Ein Fold (~30 min auf dem Mac) -> erster Vergleich mit TF-IDF (0.352)
.venv/bin/python experiments/04_transformer/train.py --folds 0

# 3) Restliche Folds (einzeln oder zusammen); OOF wird geschrieben, sobald alle 5 da sind
.venv/bin/python experiments/04_transformer/train.py --folds 1,2,3,4
```

- Standard: `answerdotai/ModernBERT-base`, LR 5e-5, Batch 8, 4 Epochen, Gradient Checkpointing, GroupKFold-Folds aus `src/cv.py`
- Ausgabe je Epoche: Train-Loss, Balanced Accuracy, Familien-Accuracy, Log-Loss, die 4 schwächsten Recalls
- Ergebnisse: `models/04_transformer/<run>/` (`log_fold*.csv`, `fold*_val.npy`, `fold*_test.npy`, `oof_*.parquet`)
- OOF stammt aus der letzten Epoche (keine Epochenauswahl auf dem Valid-Fold, damit die CV ehrlich bleibt)

## Modellwahl (Benchmark Mac 16 GB, MPS)
| Modell | Batch | Grad.-Ckpt | s/Schritt | GPU-Speicher |
|---|---|---|---|---|
| DeBERTa-v3-base | 8 | nein | 2.6 | 7.1 GB |
| DeBERTa-v3-base | 8 | ja | 3.0 | 5.5 GB |
| **ModernBERT-base** | **8** | **ja** | **1.65** | **4.4 GB** |

- ModernBERT ist ~2x schneller und erhält die Typografie (— ’ “ …) verlustfrei. DeBERTa macht „…“ zu `[UNK]`.
- Batch 16 ohne Checkpointing hat den Mac in den Swap getrieben. Ein Speicherlimit (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`) bricht jetzt lieber mit Out-of-Memory ab.
- Transformers v5 lädt Modelle im Checkpoint-dtype (fp16) -> im Skript wird `dtype=torch.float32` erzwungen (sonst NaN-Loss).
- Für DeBERTa-large oder schnelleres Training: Kaggle-Notebook mit GPU (T4/P100); das Skript nutzt CUDA + fp16 automatisch.

## Ergebnis Lauf 1: ModernBERT-base, LR 5e-5, 4 Epochen, nur Fold 0
| Epoche | Train-Loss | Balanced Acc. | Familien-Acc. | Val-Log-Loss | Dauer |
|---|---|---|---|---|---|
| 1 | 2.72 | 0.178 | 0.581 | 2.34 | 8 min |
| 2 | 2.03 | 0.270 | 0.667 | 1.98 | 8 min |
| 3 | 1.49 | 0.320 | 0.696 | 1.91 | 8 min |
| 4 | 0.87 | 0.357 | 0.706 | 2.10 | 7 min |

Vergleich auf **demselben Fold 0** (540 Texte, 30 Prompts):

| Modell | Balanced Acc. |
|---|---|
| TF-IDF + LogReg | 0.374 |
| ModernBERT | 0.357 |
| Blend geom. 0.3·BERT + 0.7·TF-IDF | **0.391** |
| Blend arithm. 0.5/0.5 | 0.383 |

- Fold 0 ist ein eher leichter Fold (TF-IDF gesamt: 0.352).
- Die beiden Modelle machen **unterschiedliche Fehler**: Nur 41 % der Vorhersagen stimmen überein. 15,7 % trifft nur TF-IDF, 14,1 % nur BERT. Das ist eine gute Grundlage für ein Ensemble.
- BERT hilft genau bei schwachen Klassen: glm-5.3 0.07 → 0.27, claude-opus-4-8 0.20 → 0.37, claude-fable-5 0.03 → 0.10.
- Die Balanced Accuracy steigt in Epoche 4 noch, der Val-Log-Loss aber schon (Überkonfidenz). 5 Epochen wären einen Test wert.
- Blend-Gewichte sind auf nur 540 Texten bestimmt (Rauschen ca. ±2 Pp.). Für belastbare Zahlen braucht es alle 5 Folds.
