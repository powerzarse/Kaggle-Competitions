# 08 – GLM-Likelihood-Klassifikator mit Pool-Daten

Hebel im öffentlichen 0.694-Notebook (`kaggle/public_694/`): Die GLM-Stufe bewertet ausgewählte Testtexte unter
den echten Modellen GLM-5.3 und GLM-5.3-Flash und verrechnet einen kleinen Klassifikator (`ll_model.json`,
multinomiale LogReg auf 54 Likelihood-Merkmalen, 160 Tokens) mit dem Stack. Dieser Klassifikator ist nur auf den
2.699 train.csv-Texten trainiert. Idee: mehr Trainingstexte aus dem Open Pool → besserer Klassifikator,
**ohne zusätzliche Laufzeit in der Abgabe** (die Laufzeit ist dort mit ~8,7 von 9 h ausgereizt).

## Bausteine
- `v1src/`: Code der Stufe unverändert aus dem öffentlichen Notebook (Apache 2.0, estopadilla) + `ll_model.json`.
- `models/08_glm_likelihood/train_v3/`: Token-Scores von train.csv unter beiden GLMs
  (Output von `estopadilla/ntl-glm-likelihood-train`, Version 3). Daraus nachgebaute 160-Token-Merkmale
  stimmen **exakt** mit Mittelwert/Skala in `ll_model.json` überein (geprüft).
- `kaggle/build_notebook_glm_pool.py` → `linuszarse/ntl-glm-likelihood-pool`: bewertet 6.600 Pool-Texte
  (GLM-lastig, ohne kb-Proxy, stilnähere Hälfte, ein Text je Prompt) in 7 Häppchen, stoppt vor 11 h.
- `evaluate.py`: baut die Stufe lokal nach. Stack-Ersatz = unser Ensemble (OOF 0.609); vergleicht
  Klassifikator-Varianten per GroupKFold-OOF. Pool-Zeilen werden je Klasse auf gleiches Gewicht gebracht.

## Ergebnisse (Stack-Ersatz, balanced accuracy auf train.csv)
| Klassifikator | LL allein | Stufe mass 20 % | Stufe uncertain 30 % (+fable) |
|---|---|---|---|
| nur train, C=0.05 (wie öffentlich) | 0.190 | 0.609 → 0.637 | 0.638 |
| LogReg + Pool (w=1) | 0.196 | 0.638 | 0.640 |
| HGB nur train | 0.182 | 0.638 | 0.642 |
| **HGB + Pool (w=0.5, W=1)** → V5 | 0.189 | **0.645** | – |

Pool-Lauf 1 (2026-11-15): 6 von 7 Häppchen = 5.670 Texte in 9,5 h (Flash ~2,1 s/Text, GLM-5.3 ~3,7 s/Text).
Bootstrap über Prompts (HGB+Pool w=1 vs. LogReg, mass 20 %): +0,48 Pp., 90 %-Intervall [−0,07; +1,00], P(>0) = 0,91.
Abdeckung der Stufe (LogReg): 5 % 0.621 · 10 % 0.631 · 15 % 0.635 · 20 % 0.637 · 30 % 0.638 → bei 20 % gesättigt.
V5 ändert im Nachbau 8,4 % der Vorhersagen (Abbruchgrenze der Stufe: 25 %).
