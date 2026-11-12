# 06 – Externe Daten: Name That LLM Open Pool

Datensatz: https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool (CC BY-SA 4.0)
- 223.840 Texte, alle 18 Labels, **keine Überschneidung mit train.csv** -> train.csv bleibt saubere Testmenge
- Lokal: `data/external/open_pool/ntl_open_pool.parquet`, laden mit `src.data.load_open_pool(per_class=...)`
- Auf Kaggle: als Input einbinden (Notebook: „Add Input“, CLI: `dataset_sources` in kernel-metadata.json)
- Ungleich verteilt (3.5k–23k je Modell) -> balanciert ziehen (`per_class`) bzw. `class_weight="balanced"`

## Skripte
- `run_fast.py`: LinearSVC (Wort- + Char-TF-IDF + Stilo ×0.1) – Skalierung (nur Pool) + Pool + train.csv (GroupKFold-OOF)

## Ergebnisse (Balanced Accuracy auf echter train.csv)
| Setup | Balanced Acc. |
|---|---|
| nur train.csv (CV, Referenz) | 0.352 (Public LB 0.320) |
| LogReg, Pool 1.000/Modell | 0.388 |
| LinearSVC, Pool 1.000/Modell | 0.400 |
| LinearSVC, Pool 300/Modell + train.csv (Gewicht 5) | 0.402 |
