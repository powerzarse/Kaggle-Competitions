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

## V10 (`v10_w15/`, Kaggle: `linuszarse/ntl-701-fork-v10-w15`)
V9 + zwei Werte, sonst nichts (`build_v10_w15.py`):
- Mischgewicht des GLM-Klassifikators W = 1.5 statt 1.0.
- GLM-Stufe bewertet höchstens 25 % des Tests (`--max-cover 0.25` statt 0.30), weniger, wenn die Zeit nicht reicht.

Simulation in der V9-Kette (Experiment 09, Stack-Ersatz, train.csv):

| Abdeckung | W 1.0 | W 1.25 | W 1.5 | W 2.0 |
|---|---|---|---|---|
| 20 % | 0.6434 | 0.6431 | 0.6453 | 0.6431 |
| 25 % | 0.6472 | 0.6475 | **0.6494** | 0.6468 |
| 30 % | 0.6460 | 0.6449 | 0.6472 | 0.6423 |

Abbruchgrenze der Stufe (25 % geänderte Vorhersagen aller Texte): im Nachbau 11,3 % bei W 1.5 / 25 %.
Lokaler Test: Patch-Zellen von V6 und V9 laufen gegen den echten `v1_stage.py`, `ll_model.json` hat W 1.5.

**Public LB 0.696** (V9: 0.701). Weder W 1.5 noch die Apex-Korrektur helfen auf dem Test.

## V11 (`v11_v6w15/`, Kaggle: `linuszarse/ntl-701-fork-v11-v6w15`)
Dieselben zwei Werte auf V6 statt V9, also **ohne** die Apex-Korrektur (`build_v10_w15.py --variant v11`).
Absicherung für den Fall, dass die Apex-Korrektur auf dem echten Stack schadet (in der Simulation auf unserem
Ersatz-Stack schadet Apex insgesamt, siehe Experiment 09). Simulation, 25 % Abdeckung:

| | W 1.0 | W 1.5 |
|---|---|---|
| V6-Kette (Apex auf Stack-Logits) | 0.6390 (V6) | 0.6405 (V11) |
| V9-Kette (Apex nach der GLM-Stufe) | 0.6472 (V9) | 0.6494 (V10) |

Zusammen mit V6 (0.707) und V9 ergibt das ein 2×2-Raster: Apex-Korrektur ja/nein × W 1.0/1.5.

**Public LB 0.702** (V6: 0.707). Ergebnis des Rasters: siehe `experiments/09_postprocess/README.md`.

## Herkunft der Dateien
`v6_hgb/` wurde am 2026-11-17 per `kaggle kernels pull linuszarse/ntl-701-fork-v6-hgb -m` von Kaggle
wiederhergestellt (genau die abgegebene Version, inkl. Commit-Ausgaben); in `kernel-metadata.json` ist nur
`code_file` auf `v6_hgb.ipynb` angepasst. V9–V11 bauen auf dieser Fassung auf.
Der Generator `build_v6_hgb.py` (mit `public_701.ipynb` und `original-kernel-metadata.json`) kam am 18.11. aus main
dazu. Seine Ausgabe hat denselben Code in allen 33 Zellen; sie unterscheidet sich nur in der Reihenfolge der
Datensätze/Modelle in `kernel-metadata.json` und hat keine Ausgaben. Ein erneuter Lauf überschreibt `v6_hgb/`.

Push: `.venv/bin/kaggle kernels push -p kaggle/public_701/v6_hgb` (T4, GPU-Kontingent!)

## V12–V14: weniger aggressiv (18.11., `build_v12_gentle.py`)
Je genau eine Änderung an der Patch-Zelle von V6 (Basis `v6_hgb/v6_hgb.ipynb`), Apex unverändert.
Simulation und Begründung: `experiments/10_v6_gentle/README.md`.

| Version | Ordner / Kernel | Änderung | Simulation (20 %) |
|---|---|---|---|
| V12 | `v12_w07` / `linuszarse/ntl-701-fork-v12-w07` | W 0.7 (≈ Temperatur aus OOF, T 1.55 → W 0.64) | −0,37 |
| V13 | `v13_glmrest` / `linuszarse/ntl-701-fork-v13-glmrest` | `member_logp`: p(Rest)/16 für jede Nicht-GLM-Klasse | −0,26 |
| V14 | `v14_pool025` / `linuszarse/ntl-701-fork-v14-pool025` | Pool-Gewicht 0.25 statt 0.5 | −0,04 |
| – | `v14_sigmoid` / `linuszarse/ntl-701-fork-v14-sigmoid` | CalibratedClassifierCV sigmoid, innere GroupKFold | −0,59 (nur gebaut) |

Lokaler Test: Alle Patch-Zellen laufen gegen den echten `v1_stage.py` (aus dem Payload von V6), inklusive des
Commit-Smoke-Tests; `member_logp` gibt auf allen 2.699 train-Zeilen normierte Log-Wahrscheinlichkeiten. Commit-Läufe
auf Kaggle (18.11.): V12, V13 und V14 Pool 0.25 (Version 2) OK, Fit 18–19 s. Die Sigmoid-Fassung brach in Version 1 ab
(ältere sklearn auf Kaggle verlangt float64-Gewichte); Version 2 OK, Fit 92 s.

**Public LB (18.11.):** V12 0.704, V13 0.705, V14 0.707 (V6: 0.707). Alle im Rauschen, keine Variante schlägt V6.

Push: `.venv/bin/kaggle kernels push -p kaggle/public_701/v12_w07` (bzw. `v13_glmrest`, `v14_pool025`)

## V15–V17: Familien-Spezialist in Apex (19.11., `build_v15_spec.py`)
V6 + Familien-Spezialist ([Experiment 12](../../experiments/12_family_specialist/README.md)): je Familie (Claude,
GPT, Gemini) TF-IDF + logistische Regression auf bis zu 15.000 Open-Pool-Texten je Klasse, privater Datensatz
`linuszarse/ntl-family-spec` (ohne Pickle, Smoke-Test gegen `reference.npz` im Commit). Apex nutzt
Geschwister-LR × Spezialist^t mit Stärke s; scheitert der Spezialist, verhält sich die Zelle wie V6.

| Version | Ordner / Kernel | t | Stärke | Simulation | Ref | Public LB |
|---|---|---|---|---|---|---|
| V15 | `v15_spec` / `linuszarse/ntl-701-fork-v15-spec` | 1.0 | 0.25 | +0,69 | 58554338 | 0.697 |
| V16 | `v16_spec015` / `linuszarse/ntl-701-fork-v16-spec015` | 1.0 | 0.15 | +0,37 | 58554345 | 0.702 |
| V17 | `v17_spec_t15` / `linuszarse/ntl-701-fork-v17-spec-t15` | 1.5 | 0.20 | +0,74 | 58555074 | 0.697 |

**Public LB (19.11.): V15 0.697, V16 0.702, V17 0.697 (V6 0.707).** Alle schlechter, umso mehr, je stärker der
Spezialist wirkt. Abgegeben am 19.11. um 09:53 / 09:53 / 10:13 UTC. Bau: `.venv/bin/python kaggle/public_701/build_v15_spec.py
--name v15_spec --t 1.0 --strength 0.25`, Push `.venv/bin/kaggle kernels push -p kaggle/public_701/v15_spec`.
`build_v6_knobs.py` (Sinkhorn, Zielmischung, Apex-Stärke) ist gebaut, aber ungenutzt: Die Simulation zeigt dort
keinen Hebel.

