# 11 – Kimi-K3-Likelihood (ab 18.11.)

Anlass: Am GLM-Klassifikator ist nichts mehr zu holen (Experiment 10, alle Varianten 0.702–0.707). Der GLM-Hebel
funktioniert, weil GLM offene Gewichte hat und wir die Likelihood unter dem echten Modell messen. Kimi-K3 hat
ebenfalls offene Gewichte.

## Was es gibt
- Segmente: `xerneas/kimik3-q1-ntl-segs-a/b/c` (16.11.), Kimi-K3 UD-Q1_0, 93 Schichten in 23 Segmenten
  (je ~20 GB, das letzte 26 GB, zusammen ~465 GB), Lizenz Kimi K3 License (Moonshot AI, MIT-ähnlich).
- Patch `ntl-kimi-k3-segrun.patch` (in jedem der drei Datensätze, sha256 `8506eb28…`): ntl-segrun plus
  Segment-Unterstützung für glm-dsa, glm5-next und kimi-k3. Passt sauber auf llama.cpp `b12247` (lokal mit
  `git apply --check` geprüft); `kimi-k3` und der Typ `Q1_0` (mit CUDA-Kernels) sind dort schon enthalten.
- Kein öffentliches Notebook nutzt die Segmente. Das Team auf Platz 14 (0.756) hat selbst Kimi-GGUFs hochgeladen
  (`ggayoayogg/kimik3-ud-q2kxl-*`).
- Architektur: hybride KDA (linear) + MLA, latente MoE (2,8T Parameter, ~50B aktiv), Cross-Layer-Residuals
  (`attn_res.block_size`). Dadurch wächst der Zustand zwischen den Segmenten mit der Schicht:
  `n_embd * (1 + ceil(l / block_size))`.

## Erwartung
Ersatz-Stack (OOF 0.609): Kimi-Recall 0.55, Hauptverwechslung gpt-5.5 (22 Kimi-Texte als gpt-5.5, 17 gpt-5.5 als
Kimi). Zum Vergleich hob die GLM-Likelihood GLM-5.3 von 0.23 auf 0.60 und GLM-Flash von 0.44 auf 0.74.
Kimi-Recall 0.8 wären +1,4 Pp. balanced accuracy, bevor man abzieht, was an GLM-Abdeckung verloren geht.

## Engpass: 9-h-Limit
Stack ~6,7 h, die GLM-Stufe füllt die restlichen ~2 h (Flash 870 s fest + 0,0141 s je Token-Zeile, GLM-5.3 1140 s
+ 0,0231 s, also ~6 s je Text). Kimi braucht mindestens einen Durchlauf über 465 GB. Ohne Messung lässt sich nicht
sagen, wie viele Texte hineinpassen und wie viel GLM-Abdeckung (20 % → 15 % kostet in der V6-Kette ~0,45 Pp.)
dafür abgegeben werden muss.

## Schritt 1 + 2: `kaggle/kimi_probe/` (Kernel `linuszarse/ntl-kimi-likelihood-train`)
Erzeugt mit `kaggle/build_notebook_kimi_probe.py`. Eine Sitzung (2× T4, Internet an):
1. ntl-segrun bauen (b12247 + Patch, CUDA sm_75, llama/ggml statisch), Binary + CUDA-Bibliotheken nach
   `/kaggle/working/ntl-bin-kimi` (später Kernel-Quelle der Abgabe, die offline laufen muss).
2. Metadaten und Chat-Template aus Segment 0, Präfix = Template mit leerer Nutzernachricht (Rückfall: Kimi-K2-Format).
3. Probe: 50 train-Texte (16 Kimi, je 2 der übrigen Klassen), Lade- und Rechenzeit je Segment getrennt.
4. Gate: Durchlauf ≤ 3.600 s und ≤ 0,10 s je Token-Zeile, sonst Stopp. Danach die übrigen 2.649 train-Texte in
   Häppchen (≤ 900, je ein Durchlauf), so viele wie bis 11 h passen. Ausgabe je Häppchen:
   `out/{tag}_kimi_features.csv`, `_scores.npz` (Token-Scores, für kürzere Schnitte), `_timing.json`;
   am Ende `out/kimi_train_feats_L160.parquet`.

`src/run_pipeline_kimi.py` = Josés `run_pipeline.py` (Apache 2.0) mit vier Änderungen: Zustandsbreite für Kimi,
zwei Zustandsdateien im Wechsel (die Breiten unterscheiden sich, in-place geht nicht), Ladezeit bis „context ready“
je Segment im Timing, `--segs` auch als JSON-Liste. Lokal gegen ein gefälschtes ntl-segrun (gleiche Dateiformate,
prüft die Zustandsbreiten) getestet, auch mit erzwungener Aufteilung in mehrere Häppchen; das ganze Notebook lief
lokal mit gefälschten Segmenten durch.

## Schritt 3 (offen): lokal simulieren
- Kimi-Klassifikator (HGB auf Kimi-Merkmalen, evtl. zusammen mit GLM-Merkmalen), OOF über die GroupKFold-Folds.
- Achtung Apex: In V6 ersetzt Apex jede Nicht-GLM-Vorhersage durch die verfeinerte Stack-Vorhersage. Eine
  Kimi-Stufe vor Apex würde also überschrieben. Entweder läuft sie nach Apex, oder Apex behält auch die
  Kimi-Entscheidungen. Das klärt die Simulation.
- Abbruch, wenn nach Abzug der verlorenen GLM-Abdeckung weniger als +0,8 Pp. bleiben.
