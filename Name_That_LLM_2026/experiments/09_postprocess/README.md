# 09 – GLM-Stufe und Apex-Nachbearbeitung (0.702-Notebook)

Frage: Geht im 0.702-Notebook ein Teil dessen verloren, was die GLM-Stufe (mit unserem Klassifikator) korrigiert?

Ablauf im Notebook: Stack-Logits → GLM-Stufe (Top 15 % nach GLM-Masse, `zl + W/T'·ll`) → Apex
(Geschwister-Residual + weiche Typografie-Regeln). Apex rechnet mit den **Stack-Logits vor der GLM-Stufe** und
ersetzt jede Vorhersage, die nicht GLM ist. Was die GLM-Stufe von GLM weg oder zwischen Nicht-GLM-Klassen
verschiebt, wird damit überschrieben.

`simulate.py` baut die Kette auf train.csv nach. Den Code (Residual, Apex, Temperatur) lädt es direkt aus
`kaggle/public_702/public_702.ipynb`. Als Stack-Ersatz dient unser Ensemble (OOF 0.609). Der HGB-Klassifikator
und das Residual sind OOF über die GroupKFold-Folds gerechnet. Merkmale kommen aus `ll_train.npz`.

## Ergebnis (2026-11-17, balanced accuracy auf train.csv)
| Variante | bal. acc | Δ vs. 0.702 |
|---|---|---|
| Stack allein | 0.6094 | −2,40 |
| Stack + GLM-Stufe (ohne Apex) | 0.6397 | +0,63 |
| Stack + Apex (ohne GLM-Stufe) | 0.6068 | −2,66 |
| **0.702 wie öffentlich** (Apex auf Stack-Logits) | 0.6334 | 0 |
| dito, Typografie hart (penalty 0) | 0.6334 | 0 |
| **Apex auf Logits nach der GLM-Stufe** (GLM-Labels behalten) → V8 | 0.6375 | **+0,41** |
| dito, `corrected=True` | 0.6375 | +0,41 |
| dito, strength 0.2 / 0.3 | 0.6379 / 0.6372 | +0,45 / +0,37 |

- Die GLM-Stufe ändert 185 von 2.699 Vorhersagen (137 nach GLM, 43 von GLM weg). Der öffentliche Apex
  überschreibt davon 48 wieder. In 18 dieser Fälle lag die GLM-Stufe richtig, nur in 7 das Apex-Ergebnis.
- Bootstrap über Prompts (V8 vs. 0.702): +0,41 Pp., 90 %-Intervall [+0,15; +0,68], P(>0) = 1,00.
- **Vorsicht:** Auf unserem Ersatz-Stack schadet Apex insgesamt (−0,26). Das ist nicht übertragbar: Unser
  Ensemble enthält schon eine Zeichen-TF-IDF-SVM, die die Typografie abdeckt, der echte Stack vermutlich weniger.
  Belastbar ist nur der Mechanismus (Apex sollte die Korrekturen der GLM-Stufe sehen), nicht die Größe des Effekts.

Umsetzung: `kaggle/public_702/build_v7_hgb.py --apex-post-glm` → `v8_postapex/`. Einzige Änderung gegenüber V7:
`apply_postprocessing` bekommt `post-glm.npz` statt `FINAL_LOGITS` als Basis.
