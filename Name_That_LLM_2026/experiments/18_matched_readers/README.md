# 18 – Host-ähnliche Texte für die TF-IDF/SVC-Leser des 0.727-Kerns (26.11.)

Anlass: Nachbearbeitung und GLM-Kopf gelten als ausgereizt (Exp. 17, Forum 761507); neues Signal muss in die Leser.
Unsere LB-Lehre: Auf train.csv gefittete Evidenz hilft, auf dem Pool gefittete schadet → Test ähnelt train.
`markbarney/ntl-matched-passages-five-models-1k` (4.964 Texte, 26.11.) + `…-four-models` (597) sind die ersten
externen Daten, die wie der Host erzeugt wurden (ein Nutzer-Turn „{prompt}\n\nWrite a single paragraph of about N
words, with no title or headings.“, Herstellerroute über OpenRouter): gemini-3.7-flash, glm-5.3, glm-5.3-flash,
kimi-k3, qwen3.8-max (+ gemini-3.8-flash aus dem Vier-Modelle-Satz). Ablage `data/external/matched/`.

Ziel: der vorab trainierte Kern `ntl666-prefit` des 0.727-Notebooks (Rezept in dessen `provenance.json`,
Methode areebbb/ntl-we-back-0-666 V14): **svc4** = LinearSVC (C 0.5, balanced) auf 4 TF-IDF-Ansichten
(char 3–5 250k ×1, word 1–3 180k ×0.7, shape 2–6 60k ×0.5, function 1–3 50k ×0.5), train w1 +
`dhruvtibarewal/nlm-matched-clean` (24.443 Pool-v13-Texte, ausgewählt) w0.5 + übriger Pool v13 (166.145) w0.35,
Blendgewicht 12; tfidf (LR, train + matched-clean, 0.25), distort (0.10).

## Vortest 1 (`quick_test.py`, ohne Pool, OOF über die Prompt-Folds, argmax)
| Leser | Daten | bal | Kimi | Qwen | Gem-3.7 | Gem-3.8 |
|---|---|---|---|---|---|---|
| tfidf | ohne markbarney | 0.412 | 0.09 | 0.41 | 0.27 | 0.62 |
| tfidf | + markbarney w0.5 | 0.394 | 0.49 | 0.76 | 0.63 | 0.43 |
| svc4-lite | ohne markbarney | 0.440 | 0.10 | 0.59 | 0.34 | 0.61 |
| svc4-lite | + markbarney w0.5 | 0.440 | 0.46 | 0.84 | 0.77 | 0.37 |

Starke Verschiebung zu den markbarney-Klassen, bal. Acc. gleich → priorunabhängig nachgemessen.

## Vortest 2 (`quick_test2.py`, svc4-lite, Sinkhorn auf softmax(12 · Score))
| Daten | bal Sinkhorn | Macro-AUC | Kimi | Qwen | neu richtig / neu falsch |
|---|---|---|---|---|---|
| ohne markbarney | 0.442 | 0.903 | 0.18 | 0.49 | – |
| + markbarney w0.5 | 0.454 | 0.907 | 0.37 | 0.53 | 176 / 144 |
| + nur Kimi+Qwen w0.5 | 0.455 | 0.908 | 0.36 | 0.53 | 100 / 67 |

Echter, kleiner Gewinn (+1,2 Pp. im Einzelleser), vor allem Kimi (Recall verdoppelt). „Nur Kimi+Qwen“ gleich gut
mit weniger Änderungen (Gemini-3.7-Texte verschieben nur Gemini-3.7 ↔ 3.8).

## Test 3 (`svc4_pool_test.py`): echtes svc4 mit Pool
Vektorisierer fest aus dem Kern, nur die SVC neu; Pool v14 lokal, geschichtete Stichprobe 30 % mit Gewicht
0.35/0.3 (voller Pool passt nicht in 16 GB). Ergebnis: ausstehend.

## Plan
Hält der Gewinn mit Pool: svc4 auf allen Daten + markbarney (Kimi+Qwen) neu fitten (gleiche Vektorisierer → nur
`model.npz` neu), als privaten Datensatz hochladen, im 0.727-Notebook nur `core/svc4/model.npz` ersetzen (V24),
Abgabe nach Rücksprache. Prüfen nur über das LB (Sim für Nicht-GLM-Entscheidungen zweimal widerlegt).
