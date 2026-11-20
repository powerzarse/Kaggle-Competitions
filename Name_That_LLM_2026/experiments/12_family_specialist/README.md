# 12 – Familien-Spezialist als Apex-Evidenz (ab 19.11.)

Anlass: Kimi-Likelihood ist vorerst blockiert (Experiment 11). Die xerneas-Segmente nutzen den Tensortyp 66, den
weder llama.cpp noch die Forks von Unsloth oder ik_llama kennen. Die IQ2_XXS-Scheiben von Lian Matsuo brauchen eine
private Laufzeit und 731 GB pro Durchlauf. Apex hat gezeigt, dass Umsortieren innerhalb der Familien auf dem LB wirkt
(lumibee: 0.696 → 0.701). Die Geschwister-LR dort ist schwach (Satzzeichen/Funktionswörter, Stärke 0.10). Der Open
Pool hat pro Familie 44k–70k Texte, ein Vielfaches dessen, was die öffentlichen Modelle gesehen haben.

Wichtig vorab: Die V6-Wiederholung (18.11.) kam wieder auf 0.707. Die Pipeline reproduziert auf etwa ±0,001, kleine
LB-Unterschiede (V12 −0,3, V13 −0,2, V11 −0,5) sind also echt (bezogen auf den Public-Teil).

## Schnelltest: TF-IDF + logistische Regression (`quick_tfidf.py`, `export_spec.py`)
Je Familie Wort-1–2- und Zeichen-2–5-Gramme, multinomiale LR (C 4, class_weight balanced), bis 15.000 Pool-Texte je
Klasse. Bewertet auf train.csv (nicht im Pool, also sauber getrennt).

| Familie | Ersatz-Stack | Geschwister-LR | Spezialist |
|---|---|---|---|
| Claude | 0.737 | 0.402 | 0.652 |
| GPT | 0.667 | 0.384 | 0.555 |
| Gemini | 0.655 | 0.433 | 0.590 |

Allein schwächer als der Stack, aber als zusätzliche Evidenz (LR × Spezialist^t) in der V6-Kette (GLM 20 %, Apex):

| t \ Stärke | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|
| 1.0 | +0,37 | +0,48 | **+0,67** | +0,48 |
| 1.5 | +0,56 | +0,74 | +0,48 | +0,15 |
| 2.0 | +0,67 | +0,78 | +0,19 | −0,11 |

(finaler Spezialist; Δ in Pp. gegen V6). Bootstrap t 1.0 / 0.25: +0,69 [+0,19; +1,22], P(>0) 0,98. Geändert werden
~118 von 2.699 Vorhersagen, die Gewinne liegen vor allem bei GPT-5.6 Sol (0.49 → 0.56), Luna und Opus 4.6.
Zum Vergleich ändern nur die Stärke von Apex (LR allein 0.1 → 0.2) oder der Sinkhorn-Ausgleich (`sim_sinkhorn.py`,
alle Einstellungen ±0,3, bei unausgewogenem Test schaden stärkere Werte) nichts Nennenswertes.

Mehr Daten helfen nicht (`tfidf_variants.py`): ganzer Pool statt höchstens 15k je Klasse gibt innerhalb der Familien
dieselben Werte (0.653 / 0.561 / 0.592 bei C 4, 0.652 / 0.559 / 0.590 bei C 8) und in der Kette +0,37 bzw. +0,48
(t 1.0 / 0.25) statt +0,67. Die Kette schwankt also schon bei kleinen Änderungen am Spezialisten um 0,2–0,3 Pp.;
das ist zusätzlich zum Bootstrap über Prompts zu bedenken.

Export ohne Pickle (Vokabular als JSON-Liste, idf, Koeffizienten), privater Datensatz `linuszarse/ntl-family-spec`;
der Nachbau mit `TfidfVectorizer(vocabulary=…)` + `idf_` stimmt mit sklearn auf 3e-8 überein, `reference.npz`
prüft das im Notebook (auch im Commit).

## Abgaben (`kaggle/public_701/build_v15_spec.py`)
| Version | t | Stärke | Simulation | Public LB |
|---|---|---|---|---|
| V15 `ntl-701-fork-v15-spec` v1 (Ref 58554338) | 1.0 | 0.25 | +0,69 | **0.697** (−1,0) |
| V16 `ntl-701-fork-v16-spec015` v1 (Ref 58554345) | 1.0 | 0.15 | +0,37 | **0.702** (−0,5) |
| V17 `ntl-701-fork-v17-spec-t15` v1 (Ref 58555074) | 1.5 | 0.20 | +0,74 | **0.697** (−1,0) |

Lokal gegen die echten Zellen getestet (train.csv als Test): 112 / 66 / 109 Vorhersagen anders als V6.
Commit-Läufe 19.11. OK, Smoke-Test auf dem Kaggle-Image: Abweichung zur lokalen Referenz 4e-16.
V15 und V16 am 19.11. um 10:53 abgegeben.

## LLM-Spezialist (`kaggle/build_notebook_family_spec.py`, `eval_llm_spec.py`)
Qwen3-4B-Base (24 Schichten, LoRA r16, 8 Tiefen), 14 Klassen, Softmax nur innerhalb der Familie, 3.000 Pool-Texte je
Klasse ohne train.csv-Prompts, eine Epoche (Kernel `linuszarse/ntl-family-spec-qwen3-4b`, 2,5 h auf 1× T4,
4,7 Texte/s; Loss 1.60 → 1.05, fiel noch).

| Familie | Ersatz-Stack | TF-IDF | LLM | LLM × TF-IDF |
|---|---|---|---|---|
| Claude | 0.737 | 0.653 | 0.654 | 0.716 |
| GPT | 0.667 | 0.556 | 0.545 | 0.601 |
| Gemini | 0.655 | 0.595 | 0.537 | 0.602 |

In der Kette bringt der LLM-Spezialist allein höchstens +0,37 (t 1.0 / 0.25), LLM × TF-IDF +0,48 – beides nicht
über TF-IDF allein (+0,67). Die beiden ergänzen sich innerhalb der Familien deutlich, die Kette (Apex greift nur bei
knappen Entscheidungen) setzt das hier aber nicht um. Fortsetzung mit 3.000 neuen Texten je Klasse (Seed 43, LR 7e-5):
Kernel `linuszarse/ntl-family-spec-qwen3-4b-r2`. Ob sich mehr GPU lohnt, entscheiden die LB-Scores von V15–V17.

Zweite Epoche (r2, 19.11., 2,6 h, Loss 1.05 → 0.88; `models/12_family_specialist/llm4b_r2/`):

| Familie | Ersatz-Stack | TF-IDF | LLM r2 | LLM r2 × TF-IDF | Stack + LLM r2 |
|---|---|---|---|---|---|
| Claude | 0.737 | 0.653 | 0.685 | 0.722 | 0.744 |
| GPT | 0.667 | 0.556 | 0.543 | 0.597 | 0.640 |
| Gemini | 0.655 | 0.595 | 0.592 | 0.627 | 0.672 |

Kette (Apex-Evidenz): LLM r2 allein höchstens +0,52 (t 1.0 / 0.35), LLM r2 × TF-IDF +0,67 (t 1.0 / 0.35, Bootstrap
+0,70 [+0,11; +1,33]), also wieder nur so gut wie TF-IDF allein (+0,69).

Andere Einbindung (`sim_pre_glm.py`): Spezialist^λ vor der GLM-Stufe in den Stack einrechnen (nur innerhalb der
Familie umsortieren, Familienmasse bleibt), danach GLM-Stufe und Apex wie V6 bzw. mit Spezialist-Evidenz. Bestes
LLM r2 × TF-IDF λ 0.3: +0,68 [+0,00; +1,45]; TF-IDF allein höchstens +0,37. Gleiche Decke von ~+0,7 Pp. auf
jedem Weg; ein LLM-Spezialist im Abgabe-Notebook (Qwen3-4B, ~5 min für 4.000 Texte) lohnt sich daher nicht.

## Gruppen-Spezialisten über Familiengrenzen (`sim_group_spec.py`, 19.11.)
Die größten Verluste nach der V6-Kette (Ersatz-Stack) außerhalb der Familien: Claude-Fable-5 → GLM (42 %, durch
die GLM-Stufe), Kimi-K3 → GPT-5.5 (14 %), GPT-5.5 → Kimi-K3 / Qwen3.8-Max (je 11 %). Gleicher TF-IDF-Ansatz auf
den Pool-Texten der Gruppe, danach innerhalb der Gruppe neu entscheiden (log p nach GLM-Stufe + t · log Spezialist):

| Gruppe | Stack / nach GLM-Stufe | Spezialist | bestes Δ in der Kette |
|---|---|---|---|
| Kimi-K3, GPT-5.5, Qwen3.8-Max | 0.791 / 0.791 | 0.676 | −0,04 (t 0.5) |
| Claude-Fable-5, GLM-5.3, GLM-5.3-Flash | 0.474 / 0.668 | 0.403 | +0,11 [−0,18; +0,41] (t 0.25) |

TF-IDF ist hier deutlich schwächer als Stack und GLM-Likelihood, kein Hebel. Die Prior-Korrektur für
Gemini-3.7-Flash ist in Experiment 13 beschrieben (verworfen).

## Ergebnis auf dem LB (20.11.): verworfen
V15 0.697, V16 0.702, V17 0.697 gegen V6 0.707 (reproduziert auf ±0,001). Die Wirkung kehrt sich um und wächst mit
der Stärke: Stärke 0.15 −0,5, Stärke 0.20–0.25 −1,0, statt +0,4 bis +0,7 in der Simulation. Das ist nach „Apex nach
der GLM-Stufe“ (Experiment 09, 2× −0,6) die zweite Widerlegung des Ersatz-Stacks bei Nicht-GLM-Entscheidungen.

Mögliche Gründe (nicht geprüft):
- Der echte Stack ist innerhalb der Familien deutlich stärker als unser OOF-Ersatz (Ersatz-Kette 0.638, echter Stack
  ~0.70 auf dem LB). Evidenz, die dem Ersatz hilft, überstimmt beim echten Stack richtige Entscheidungen.
- Der Spezialist ist nur auf dem Open Pool trainiert. Die Geschwister-LR von Apex (auf train.csv trainiert) hilft auf
  dem LB (lumibee 0.696 → 0.701), der Pool-Spezialist schadet. Der Test könnte train.csv näher sein als dem Pool.

Folgerung: Nicht-GLM-Entscheidungen nicht mehr über `stack_proxy` bewerten. Die Richtung (auch LLM-Spezialist,
Gruppen-Spezialisten, Prior-Korrektur) ist beendet.

