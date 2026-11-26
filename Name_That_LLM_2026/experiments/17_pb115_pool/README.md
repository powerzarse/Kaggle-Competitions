# 17 – PB-115-Kopf mit Open-Pool-Daten (25.11.)

Anlass: Öffentlich jetzt **0.724** (`lumibee/0-724-pb-115-likelihood-unlocked-glm-coverage`, v69; unser
unveränderter Fork ist V21, `kaggle/public_724/`). Wir (V6 0.707) sind auf Platz 1.247 von 7.046 gefallen, Bronze
0.719. Der GLM-Kopf von v69 („PB-115“: 54 Zusammenfassungen + 61 token-gleiche Merkmale GLM-5.3 gegen Flash,
0.55 LR + 0.45 · 3 LGB, Hybrid aus 18-Klassen-log-p und marginalen GLM-log-odds, α 0.35, W 1.5, Abdeckung bis 40 %)
wird **nur auf den 2.699 train-Texten** trainiert. Bei uns hat das Hinzunehmen von Pool-Texten den GLM-Kopf zweimal
verbessert (V4 → V5 +1,0, V6 +0,6).

## Format
PB-115 nutzt „Prefix B“ (`v1src/ll_model.json`): `Reasoning Effort: Low`, Nutzer-Turn
„Write a single paragraph of about {n_words} words, with no title or headings.“, 160 Tokens. Unsere Pool-Scores
(Experiment 08/14: `Reasoning Effort: Max`, leerer Nutzer-Turn) passen nicht. Neuer Lauf:
`kaggle/build_notebook_glm_scores_pb.py` → Kernel `linuszarse/ntl-glm-scores-poolpb160` (25.11. gepusht), Auswahl wie
Experiment 08 (6.600 Texte in 7 Häppchen, erwartet ~5 Häppchen in 11 h) + 40 train-Prüftexte in Häppchen 0
(Formatprüfung gegen `lumibee/ntl-prefixb-full-115`: n_tok exakt gleich, lp_mean korreliert).

`v1src/`: Code des 0.724-Notebooks (Payload `run_pipeline.py`/`segio.py`/`features.py`/`v1_stage.py`/`ll_model.json`,
Zelle 27 `prefix_variant_stage.py`, Zelle 28 `v1_token_features.py`, `spec.json` aus dem PB-115-Datensatz).
`features.py` und `segio.py` sind identisch mit Experiment 08.

## Simulation (`simulate_pool.py`, auf unseren alten Prefix-A-Läufen, train 2.699 + Pool 4.734, Schnitt 160)
Kopf mit genau dem PB-115-Rezept, OOF über die Prompt-Folds. Kette: Ersatz-Stack → Auswahl nach GLM-Masse → Hybrid
(α 0.35, W 1.5) → „Apex-ähnlich“ (Nicht-GLM-Vorhersagen fallen auf den Stack zurück, außer Fable-Return-geschützte).
Δ gegen „nur train“ in Pp., Bootstrap 90 % über Prompts:

| Kopf | GLM-3 Kopf | Rec. GLM-5.3 / Flash / Fable | Δ 30 % | Δ 40 % |
|---|---|---|---|---|
| nur train (wie 0.724) | 0.697 | 0.58 / 0.74 / 0.30 | 0 | 0 |
| + Pool w0.5 | 0.751 | 0.69 / 0.83 / 0.18 | −0,30 | −0,30 |
| + Pool w1.0 | 0.752 | 0.71 / 0.85 / 0.11 | −0,73 | −0,73 |
| **+ Pool ausgeglichen w0.5** | 0.724 | 0.61 / 0.79 / 0.29 | **+0,41** [+0,11; +0,74] | **+0,45** [+0,15; +0,78], P>0 0.99 |
| + Pool ausgeglichen w1.0 | 0.739 | 0.62 / 0.80 / 0.27 | +0,37 | +0,41 |

„ausgeglichen“ = jede Klasse bekommt im Pool dieselbe Gesamtmasse (Mittel w). Ohne Ausgleich verschiebt die
GLM-lastige Pool-Auswahl (1.000 je GLM-Klasse, 500 Fable, 250–400 sonst) den Prior Richtung GLM, und Fable wird zu GLM.
Das erklärt vermutlich auch das Scheitern der gepaarten Merkmale in Experiment 16 (dort Pool ohne Ausgleich).
Die Simulation betrifft nur GLM-Entscheidungen, für die sie bisher verlässlich war.

## Gegenprobe auf echten Prefix-B-Daten (`simulate_pb.py`)
train-Merkmale aus `lumibee/ntl-prefixb-full-115` (wie im Notebook), Pool aus `linuszarse/ntl-pb115-pool`.
Scoring-Lauf (25.11., 9,1 h): 5 von 7 Häppchen, 4.734 Pool-Texte (715 je GLM-Klasse, 358 Fable, 180–286 sonst),
Ausgabe `models/17_pb115_pool/poolpb160/`. Formatprüfung an 40 train-Texten: n_tok 40/40 gleich (beide Modelle),
lp_mean Flash r 0.9998, GLM-5.3 r 0.938 – wegen eines Ausreißers (`d66bca0a1674`, GLM-5.3-lp_mean bei lumibee −3,71,
bei uns −2,62, Flash gleich), ohne ihn r 0.9986. Gepaarte Merkmale Median r 0.907. Gleiches Binary/gleiche Segmente wie
V21 zur Testzeit. Datensatz `linuszarse/ntl-pb115-pool` privat hochgeladen.

Ergebnis (`results_pb.csv`, Log `simulate_pb.log`), Δ gegen „nur train“ in Pp., Bootstrap 90 % über Prompts:

| Kopf | GLM-3 Kopf | Rec. GLM-5.3 / Flash / Fable (40 %, Apex) | Δ 40 % ohne Apex | Δ 40 % Apex-ähnlich |
|---|---|---|---|---|
| nur train (wie 0.724) | 0.807 | 0.73 / 0.78 / 0.34 | 0 | 0 |
| + Pool w0.5 | 0.827 | 0.79 / 0.85 / 0.28 | −0,30 | −0,38 [−0,81; +0,07] |
| + Pool ausgeglichen w0.25 | 0.803 | 0.73 / 0.79 / 0.31 | −0,23 | −0,30 [−0,52; −0,07] |
| + Pool ausgeglichen w0.5 | 0.815 | 0.73 / 0.80 / 0.29 | −0,22 | **−0,34** [−0,63; −0,04], P>0 0.02 |
| + Pool ausgeglichen w1.0 | 0.818 | 0.74 / 0.81 / 0.30 | −0,11 | −0,11 [−0,44; +0,26] |

**Verworfen.** Mit echtem Prefix B ist der Kopf nur auf train schon stark (GLM-3 0.807 statt 0.697 beim alten Prefix);
der Pool verbessert GLM-3 kaum noch (+0,8 bis +2 Pp.) und kostet weiter Fable-Recall (0.34 → 0.29), auch mit Ausgleich.
Der Gewinn aus der ersten Simulation war ein Effekt des schwachen Prefix-A-Kopfes.

## Ergebnis
V22 (gebaut, `kaggle/public_724/v22_pbpool/`, lokal Ende-zu-Ende getestet) **nicht gepusht, nicht abgegeben**, weil
die Freigabebedingung (Sim bestätigt den Gewinn) nicht erfüllt ist. V21 (0.724) bleibt Basis.
