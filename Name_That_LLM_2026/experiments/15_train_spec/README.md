# 15 – Familien-Spezialist nur auf train.csv: LB-Gegenprobe zu V15 (20.11.)

Anlass: V15–V17 (Familien-Spezialist auf dem Open Pool als Apex-Evidenz, Experiment 12) lagen 0,5–1,0 Pp. unter V6,
die Simulation hatte +0,4 bis +0,7 erwartet. Die Geschwister-LR von Apex ist auf train.csv trainiert und hilft auf
dem LB. Zwei Erklärungen sind möglich:
- **Quelle:** Pool-trainierte Evidenz passt nicht zum Test, train-trainierte schon.
- **Eingriff:** Der echte Stack ist innerhalb der Familien stärker als unser Ersatz-Stack, und jede zusätzliche
  Evidenz dieser Größe überstimmt richtige Entscheidungen, egal woher sie kommt.

V20 trennt das: gleicher Eingriff wie V15, aber der Spezialist wird wie die Geschwister-LR zur Laufzeit nur auf
train.csv gefittet.

## Spezialist mit ~150 Texten je Klasse (`oof_train_spec.py`, OOF über die Prompt-Folds)
Innerhalb der Familie (Claude / GPT / Gemini):

| Evidenz | Trefferquote |
|---|---|
| Ersatz-Stack | 0.737 / 0.667 / 0.655 |
| Geschwister-LR (Apex, V6) | 0.402 / 0.384 / 0.433 |
| Pool-Spezialist (V15) | 0.653 / 0.556 / 0.595 |
| train: Wort 1-2 + Zeichen 2-5, C 4 | 0.446 / 0.381 / 0.460 |
| train: andere C (1, 16) bzw. nur Zeichen | 0.355–0.451, nicht besser |

Der train-Spezialist ist kaum stärker als die Geschwister-LR. Der Pool-Spezialist wurde schon auf train.csv gemessen
und überträgt dorthin gut (0.65). Die Erklärung „Quelle" ist damit schwächer als zunächst angenommen.

## Eingriffsstärke (geänderte Vorhersagen gegen V6 in der Kette, Ersatz-Stack, GLM 20 %)
| Spezialist | t 1.0 / 0.15 | 1.0 / 0.25 | 1.0 / 0.35 | 1.5 / 0.20 | 2.0 / 0.35 |
|---|---|---|---|---|---|
| Pool (V15) | 68 (V16) | **118 (V15)** | 156 | 111 (V17) | 214 |
| train C 4 | 31 | 73 | **112 (V20)** | 65 | 150 |

V20 = train C 4, t 1.0, Stärke 0.35: etwa so viele Änderungen wie V15. Die Änderungen überschneiden sich zu rund
der Hälfte (52 gemeinsame). Simulation −0,30 Pp. (unter den 112 geänderten vorher 43 richtig, danach 35). Zum
Vergleich, Apex allein mit höherer Stärke: 0.35 → 84 Änderungen, −0,15; 0.5 → 135, −0,45; ganz ohne Geschwister-LR
+0,22. Die Simulation ist für Nicht-GLM-Entscheidungen zweimal widerlegt, die Zahlen dienen nur zur Einordnung.

Lesart des LB-Ergebnisses:
- V20 ≥ V6 (0.707): Train-Evidenz verträgt der echte Stack, Pool-Evidenz nicht → Quelle zählt, Pool-Daten für
  Nicht-GLM-Klassen meiden.
- V20 ≈ V15 (0.697–0.702): Jeder stärkere Eingriff innerhalb der Familien schadet → Apex-Stärke nicht anfassen,
  die Quelle ist zweitrangig.

## Notebook (`kaggle/public_701/build_v20_trainspec.py`)
- Zelle 9 und Apex-Zelle wie V15 (Patches aus `build_v15_spec.py` übernommen). Neue Zelle vor Apex fittet den
  Spezialisten auf train.csv (3,9 s) und prüft ihn gegen eine lokale Referenz für 20 train-Texte (im Notebook
  hinterlegt, Toleranz 0.02; lokal 5e-7). Scheitert der Spezialist, verhält sich Apex wie V6. Kein Datensatz nötig.
- Lokal gegen die echten Zellen getestet (1.000 Pool-Texte als Ersatz-Test, Ersatz-Logits): 37 Vorhersagen anders
  als V6. Einige wechseln die Familie, auch ohne Typografie-Regel: `refine()` erhält nur die Masse je Familie, das
  argmax über alle 18 Klassen kann trotzdem in eine andere Familie springen (auch bei V6/V15).
- Kernel `linuszarse/ntl-701-fork-v20-trainspec`: Push am 20.11. 20:55 abgelehnt („Maximum weekly GPU quota of
  30.00 hours reached"). Commit-Lauf und Abgabe stehen aus.
