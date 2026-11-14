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
| + Pool | ausstehend | | |
