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

## Abdeckung der GLM-Stufe (Nachtrag 17.11.)
V7 (0.702-Basis, fest 15 %) kam auf 0.704, V6 (0.701-Basis, Abdeckung nach Zeit, höchstens 30 %) auf 0.707.
Simulation mit anderer Abdeckung (bal. acc auf train.csv, sonst wie oben):

| Abdeckung | nur GLM-Stufe | Apex vorher (öffentlich) | Apex nachher (V8/V9) |
|---|---|---|---|
| 10 % | 0.6331 | 0.6271 | 0.6309 |
| 15 % (0.702) | 0.6397 | 0.6334 | 0.6375 |
| 20 % (≈ V6) | 0.6442 | 0.6379 | 0.6434 |
| 25 % | 0.6472 | 0.6390 | 0.6472 |
| 30 % | 0.6464 | 0.6397 | 0.6460 |
| 40 % | 0.6464 | 0.6401 | 0.6468 |

Weiche und harte Typografie (penalty 0.05 / 0) geben bei jeder Abdeckung dieselbe Zahl. Mehr Abdeckung hilft bis
etwa 25 %, danach nicht mehr. Daher V9: die Apex-Korrektur auf der V6-Basis (`kaggle/public_701/build_v9_postapex.py`).

## Mischgewicht W in der V9-Kette (Nachtrag 17.11., abends)
Apex auf den Logits nach der GLM-Stufe, sonst wie oben:

| Abdeckung | W 0.75 | W 1.0 | W 1.25 | W 1.5 | W 2.0 |
|---|---|---|---|---|---|
| 15 % | 0.6371 | 0.6375 | 0.6371 | 0.6386 | 0.6368 |
| 20 % | 0.6397 | 0.6434 | 0.6431 | 0.6453 | 0.6431 |
| 25 % | 0.6423 | 0.6472 | 0.6475 | **0.6494** | 0.6468 |
| 30 % | 0.6409 | 0.6460 | 0.6449 | 0.6472 | 0.6423 |

W = 1.5 ist in jeder Zeile vorn (+0,2 Pp.), 25 % vor 30 %. → V10 (`kaggle/public_701/build_v10_w15.py`).
Recall je Klasse nach der V9-Kette (25 %, W 1): GLM-5.3 0.23 → 0.60, GLM-Flash 0.44 → 0.74; die schwächsten
Klassen bleiben unberührt (Gemini-3.7-Flash 0.17, Claude-Fable-5 0.32, GPT-5.6-Sol 0.49, Kimi-K3 0.56).
`stack_proxy` findet die OOF-Dateien jetzt in beiden Ablagen (flach oder in `bert/`, `svm/`, `lora/`).

## Public LB widerlegt die Simulation (18.11.)
| Abgabe | Apex | W / Abdeckung | Public LB |
|---|---|---|---|
| V6 | vorher (öffentlich) | 1.0 / Deadline, max. 30 % | **0.707** |
| V9 | nachher | 1.0 / Deadline, max. 30 % | 0.701 |
| V11 | vorher | 1.5 / max. 25 % | 0.702 |
| V10 | nachher | 1.5 / max. 25 % | 0.696 |

Zwei gepaarte Vergleiche, beide −0,6 Pp. für „Apex nachher“. Die Korrektur schadet auf dem Test, anders als in der
Simulation (+0,4 bis +0,55). Vermutung: Auf dem Test sind die Verschiebungen der GLM-Stufe *zwischen Nicht-GLM-Klassen*
überwiegend falsch. Der Klassifikator erkennt GLM gut, die übrigen 16 Klassen aber schlecht. Auf train.csv (OOF,
gleiche Verteilung wie die Pool-Merkmale) sieht das besser aus als auf dem Test. Der öffentliche Apex setzt
Nicht-GLM-Vorhersagen auf den Stack zurück und filtert genau das heraus. Auch W 1.5 / 25 % schadet (V11 vs. V6: −0,5).
Lehre: Simulationen mit unserem Ersatz-Stack nur für die GLM-Entscheidung heranziehen, nicht für Nicht-GLM-Klassen.
