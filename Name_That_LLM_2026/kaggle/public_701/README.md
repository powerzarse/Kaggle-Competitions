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

## V9 (`v9_postapex/`, Kaggle: `linuszarse/ntl-701-fork-v9-postapex`)
V6 + Apex-Nachbearbeitung auf den Logits **nach** der GLM-Stufe (wie V8 auf der 0.702-Basis, Experiment 09).
Erzeugt mit `build_v9_postapex.py` aus `v6_hgb/v6_hgb.ipynb`. Zwei Änderungen:
- Neue Zelle 30 patcht `v1_stage.py`: Die Stufe speichert `V1_OUT/post-glm.npz` (ids, labels, logits). Zuerst
  schreibt sie die Stack-Logits (gelten, falls die Stufe aus Zeitgründen aussteigt), nach dem Mischen `zl + w·ll`.
  Die Vorhersagen der Stufe bleiben bitgleich.
- Die Apex-Zelle (33) liest `post-glm.npz` statt `FINAL_LOGITS`, falls vorhanden, und prüft die Zeilenreihenfolge.

Warum auf V6 statt V8: Die Stufe von V6 bewertet so viele Texte, wie bis 8,8 h passen (höchstens 30 %, nach
Zeitschätzung ~20 %). Das 0.702-Notebook (V7/V8) bewertet fest 15 %. In der Simulation bringt mehr Abdeckung
+0,5 Pp. (Experiment 09), passend zu V6 0.707 vs. V7 0.704.

Lokaler Test (Stufe mit Ersatz-Pipeline, die die echten Merkmale aus `ll_train.npz` liefert, 900 train-Texte):
`post-glm.npz` hat dieselbe argmax wie die Abgabe der Stufe. Die V6-Apex überschreibt 67 Vorhersagen, die V9-Apex 3.
Steigt die Stufe aus, sind V6 und V9 identisch. Commit-Lauf auf Kaggle (17.11.) OK: Patch angewendet, keine Fehler.

**Public LB 0.701** (V6: 0.707). Die Korrektur schadet auf dem Test, siehe `experiments/09_postprocess/README.md`.

Push: `.venv/bin/kaggle kernels push -p kaggle/public_701/v9_postapex`

## Herkunft der Dateien
Der Generator für V6 lag nur auf dem Laptop und ist nicht gepusht. `v6_hgb/` wurde am 2026-11-17 per
`kaggle kernels pull linuszarse/ntl-701-fork-v6-hgb -m` von Kaggle wiederhergestellt (genau die abgegebene Version,
inkl. Commit-Ausgaben); in `kernel-metadata.json` ist nur `code_file` auf `v6_hgb.ipynb` angepasst.

Push: `.venv/bin/kaggle kernels push -p kaggle/public_701/v6_hgb` (T4, GPU-Kontingent!)
