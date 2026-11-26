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
0.35/0.3 (voller Pool passt nicht in 16 GB). Ergebnis (`models/18_matched_readers/svc4_pool_test.csv`):

| Daten | bal argmax | bal Sinkhorn | Δ Pp. [90 %] | P>0 | Macro-AUC | Kimi | Gem-3.7 |
|---|---|---|---|---|---|---|---|
| ohne markbarney | 0.471 | 0.469 | 0 | – | 0.904 | 0.22 | 0.39 |
| + nur Kimi+Qwen w0.5 | 0.473 | 0.473 | +0,37 [−0,07; +0,85] | 0.90 | 0.907 | 0.25 | 0.39 |
| **+ markbarney w0.5** | 0.471 | **0.480** | **+1,07 [+0,44; +1,81]** | 0.995 | 0.912 | 0.29 | 0.42 |

Mit Pool hilft der volle markbarney-Satz am meisten (der Pool dämpft die Gemini-3.7-Verschiebung aus Vortest 1).
Ohne Sinkhorn (argmax) kein Gewinn; im Notebook steht svc4 im Blend mit vielen Lesern und Sinkhorn 0.30 → wie viel
ankommt, zeigt nur das LB.

## Umsetzung
- `kaggle/svc4_mb/svc4_mb.py` → Kernel `linuszarse/ntl-svc4-markbarney` (Kaggle CPU, kein GPU-Kontingent): svc4 mit
  den Kern-Vektorisierern neu gefittet auf train + matched-clean + aktuellem Pool (v15, 239k; Kern: v13 – v13 lässt
  sich aus v14/v15 nicht rekonstruieren) ohne bzw. mit markbarney; Treuetest gegen das Original-`model.npz` auf den
  (für beide ungesehenen) markbarney-Texten. Lokal mit 3.000 Zeilen je Quelle getestet.
- `kaggle/public_727/build_v24_svc4mb.py` → V24a (ohne, Kontrolle: nur neue Pool-Version) und V24b (mit markbarney):
  V23 + Prüfzelle (läuft im Commit) + Austausch von `core/svc4/model.npz` nach dem Entpacken des Kerns.
- Kaggle-Lauf 26.11. (26 min, Ausgabe `models/18_matched_readers/svc4_mb_out/`): Pool 221.498, markbarney 5.561.
  Treuetest auf den ungesehenen markbarney-Texten: „ohne“ vs. Original argmax gleich 0.80, Zeilen-r 0.977,
  Trefferquote 0.32 vs. 0.31 (das Original erkennt host-ähnliche Texte dieser 5 Klassen kaum). Lokal: Kern mit
  beiden Gewichten fehlerfrei (core_votes endlich), train-argmax zu 99,96 % wie das Original.
- Commit-Läufe V24a/V24b 26.11. 13:43/13:45 OK (Prüfzelle „svc4 weights ok“, keine Fehler; Einhängepfad der
  Kernel-Ausgabe wechselt zwischen `/kaggle/input/notebooks/linuszarse/…` und `/kaggle/input/ntl-svc4-markbarney/`, das
  rekursive glob findet beide).
- Abgabe beider (sauberer A/B-Vergleich) nach Rücksprache.
