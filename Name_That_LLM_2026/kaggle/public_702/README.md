# Fork: 0.702 | Soft Typography + GLM + Qwen3.8-27B LoRA (lightningv08)

Quelle: https://www.kaggle.com/code/lightningv08/0-702-soft-typography-glm-qwen3-8-27b-lora
Unverändert importiert am 2026-11-17 als `public_702.ipynb` (Credits im Notebook). Codegleich mit alexchilton
„Name That LLM 0.702: All Public + What Failed“ (dort nur zusätzliche Markdown-Zellen und eine Input-Prüfung).

Unterschiede zur 0.701-Basis von V6 (`kaggle/public_701/`):
- GLM-Stufe fest auf die 15 % wahrscheinlichsten GLM-Texte und 90.000 Token-Positionen begrenzt (vorher 20 % / 130.000
  mit Zeit-Deadline); speichert zusätzlich `post-glm.npz`.
- Apex-Nachbearbeitung (Geschwister-Residual + Typografie-Regeln) als eigenes Modul, Typografie **weich**
  (verbotene Klassen × 0.05 statt × 0); GLM-Vorhersagen bleiben wie bisher erhalten.
- Alle Pfade fest verdrahtet (`pinned_assets.py`), der ungenutzte Zusatz-Worker ist abgeschaltet.

## V7 (`v7_hgb/`, Kaggle: `linuszarse/ntl-702-fork-v7-hgb`)
Basis + unser GLM-Likelihood-Klassifikator wie in V5/V6 (HGB auf train + 5.670 Pool-Texten, Pool-Gewicht 0.5,
W = 1.0, Merkmale aus `linuszarse/ntl-glm-ll-train`). Erzeugt mit:

    .venv/bin/python kaggle/public_702/build_v7_hgb.py --W 1.0
    .venv/bin/kaggle kernels push -p kaggle/public_702/v7_hgb

Lokaler Test der Patch-Zelle (gegen `v5_hgb_data/ll_train.npz`): identische Ausgabe wie der Commit-Lauf von V6.

**Public LB 0.704** (17.11., V6 auf der 0.701-Basis: 0.707). Der Unterschied liegt im LB-Rauschen.

## V8 (`v8_postapex/`, Kaggle: `linuszarse/ntl-702-fork-v8-postapex`)
V7 + Apex-Nachbearbeitung auf den Logits **nach** der GLM-Stufe (`post-glm.npz` statt `FINAL_LOGITS`), damit Apex
die Korrekturen der GLM-Stufe nicht überschreibt; GLM-Labels bleiben wie im Original erhalten. Begründung und
Simulation: `experiments/09_postprocess/` (+0,4 Pp. auf train.csv). Erzeugt mit `build_v7_hgb.py --W 1.0 --apex-post-glm`.
Die geänderte Zelle läuft im Commit-Lauf nicht (FAST_COMMIT); lokal mit Dateien im echten Format getestet.

## Öffentliche Fassung (`public_en/`, Kaggle: `linuszarse/glm-likelihood-classifier-on-open-pool`)
Code zellgleich mit V7, nur Zelle 0 auf Englisch (Basis, Credits, LB auf drei Basen, Fehlschläge). Eigener Kernel,
damit keine Vorversionen in der Historie stehen. Erzeugt mit `build_public_en.py`. Commit-Lauf 18.11. OK.
Bleibt privat, bis der Datensatz `linuszarse/ntl-glm-ll-train` öffentlich ist (nur nach Go des Users). Den Titel
erst nach dem Score um die Zahl ergänzen (Kaggle-UI, höchstens 50 Zeichen).
