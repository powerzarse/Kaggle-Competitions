# Fork: 0.727 | Qwen3.8-27B + PB115 Hybrid + Sibling Odds (lightningv08)

Quelle: https://www.kaggle.com/code/lightningv08/0-727-qwen3-8-27b-pb115-hybrid-sibling-odds (am 26.11. per
`kaggle kernels pull -m` geholt, `public_727.ipynb` + `original-kernel-metadata.json`; Credits im Notebook).

Gegenüber V21 (lumibee 0.724): Basis mathleteutd V12 (0.725, 40 % GLM-Abdeckung + Sibling Odds 0.55), Qwen3.8-27B-
GSQ3-Reader mit 3,5 % vor der Embedding-Stufe, PB-115-3-Wege-Gewicht 0.35 → 0.40. Gleiche Datenquellen wie V21,
anderes Docker-Image.

Aufbau, soweit für uns relevant (Experiment 18):
- Vorab trainierter Kern `ntl666-prefit` (in `lightningv08/ntl-olmo2-native24-final-inference-assets`, Rezept
  areebbb/ntl-we-back-0-666 V14): svc4 (LinearSVC C 0.5 balanced, 4 Ansichten char/word/shape/function; train w1 +
  `dhruvtibarewal/nlm-matched-clean` w0.5 + Open Pool v13 w0.35; Blendgewicht 12), tfidf (LR C 20, train +
  matched-clean; 0.25), distort (0.10).
- Zur Laufzeit auf train.csv gefittet: Sibling-Residual (`pngy_residual.py`, Stil-LR je Familie, Stärke 0.10) und
  der PB-115-GLM-Kopf.

## V23 (`v23_fork/`, Kaggle: `linuszarse/ntl-727-fork-v23`)
Unverändert (bytegleich mit `public_727.ipynb`), nur ID/Titel/privat in `kernel-metadata.json`. Push 26.11. ~10:30,
Commit-Lauf OK (11:08). Hinweis: Im Commit-Lauf (FAST_COMMIT) schreibt dieses Notebook nur einen 3-Zeilen-Platzhalter
(`variant-proof.json`: variant „queue“) und überspringt alle Stufen samt Smoke-Tests; geprüft wird nur der
Abschluss-Audit. Abgabe 26.11. 11:09, Ref 58940981.

**Public LB:** ausstehend.

Push: `.venv/bin/kaggle kernels push -p kaggle/public_727/v23_fork`
