# Fork: 0.701 | All Public + What Failed (alexchilton, Linie lumibee „NTL Kiro v44 Apex Grandmaster“)

Quelle: https://www.kaggle.com/code/alexchilton/name-that-llm-0-702-all-public-what-failed
(zum Zeitpunkt des Forks „0.701“, inzwischen umbenannt in „0.702“; Credits im Notebook).

## V6 (`v6_hgb/`, Kaggle: `linuszarse/ntl-701-fork-v6-hgb`) → **Public LB 0.707**
- Basis unverändert, einzige Änderung: Die GLM-Likelihood-Stufe (José Padilla) nutzt statt der LogReg auf train.csv
  ein HistGradientBoosting auf train.csv + 5.670 Open-Pool-Texten (8.369 Zeilen, Pool-Gewicht 0.5, Mischgewicht W = 1.0),
  also derselbe Klassifikator wie V5 (Experiment 08). Zelle 29 trainiert ihn im Notebook (17 s) und patcht
  `member_logp` in `v1_stage.py`; Auswahl, Deadline (8,8 h), Temperatur und Abbruchgrenze bleiben unverändert.
- Merkmale aus dem privaten Datensatz `linuszarse/ntl-glm-ll-train` (`ll_train.npz`, siehe `kaggle/public_694/v5_hgb_data/`).
- Vergleich: Basis auf dem Konto des Autors 0.701 → V6 0.707 (+0,6 Pp.; identischer Code schwankt laut Autor um bis zu 0,008, der Gewinn allein ist also kein Beleg –
  zusammen mit V4 → V5 (+1,0) aber konsistent).

## Herkunft der Dateien
Der Generator für V6 lag nur auf dem Laptop und ist nicht gepusht. `v6_hgb/` wurde am 2026-11-17 per
`kaggle kernels pull linuszarse/ntl-701-fork-v6-hgb -m` von Kaggle wiederhergestellt (genau die abgegebene Version,
inkl. Commit-Ausgaben); in `kernel-metadata.json` ist nur `code_file` auf `v6_hgb.ipynb` angepasst.

Push: `.venv/bin/kaggle kernels push -p kaggle/public_701/v6_hgb` (T4, GPU-Kontingent!)
