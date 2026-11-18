# 10 – Weniger aggressive GLM-Stufe in der V6-Kette (18.11.)

Anlass: Auf dem LB hat mehr Gewicht geschadet (W 1.5 / 25 %: V6 0.707 → V11 0.702, V9 0.701 → V10 0.696), ebenso
Apex nach der GLM-Stufe (zweimal −0,6). Vermutung: Der Klassifikator ist auf dem Test überkonfident und trennt die
16 Nicht-GLM-Klassen schlecht. Heute deshalb Gegenproben in die andere Richtung, alle auf der V6-Basis.

`simulate.py` baut die V6-Kette nach: Stack-Ersatz (unser Ensemble, OOF 0.609) → GLM-Stufe → Apex auf den
Stack-Logits mit harter Typografie (`apply_arrays(..., penalty=0)` aus dem 0.702-Notebook, geprüft gleich dem
Inline-Apex von V6). Klassifikator und Residual OOF über die GroupKFold-Folds.

## Zwei Befunde vorab
- **In V6 wirkt die GLM-Stufe nur über „welche Texte werden GLM“.** Apex ersetzt jede Nicht-GLM-Vorhersage durch die
  verfeinerte Stack-Vorhersage. Nur die Nicht-GLM-Verschiebungen der Stufe zurückzusetzen, ergäbe exakt V6.
  „GLM gegen den Rest“ muss deshalb im Klassifikator ansetzen: p(Rest)/16 für jede Nicht-GLM-Klasse (r2).
  Das ist die Gruppenfassung der bisherigen Mischung (der Klassifikator hat einen gleichverteilten Prior).
  r1 (jede Nicht-GLM-Klasse bekommt p(Rest)) ist zusätzlich um log 16 gegen GLM verschoben.
- **Temperatur = W.** `argmax(zl + w·log_softmax(ll/T)) = argmax(zl + (w/T)·ll)`. Die OOF-Temperatur des
  V6-Klassifikators ist T = 1.55 (NLL 2.596 → 2.509), gleichwertig zu W = 0.64. V12 (W 0.7) ist also praktisch
  der temperaturkalibrierte Klassifikator.

## Ergebnis (balanced accuracy auf train.csv, Δ in Pp. gegen V6 W 1.0)
| Variante | W | 20 % | Δ | 30 % | Δ | GLM-Vorhersagen (20 %) |
|---|---|---|---|---|---|---|
| V6 | 0.5 | 0.6293 | −0,85 | 0.6293 | −1,04 | 324 |
| V6 → **V12** | 0.7 | 0.6342 | −0,37 | 0.6342 | −0,56 | 338 |
| V6 | 0.85 | 0.6360 | −0,19 | 0.6367 | −0,30 | 342 |
| **V6** | 1.0 | 0.6379 | 0 | 0.6397 | 0 | 353 |
| V6 | 1.5 | 0.6397 | +0,19 | 0.6405 | +0,07 | 372 |
| GLM gegen Rest r2 → **V13** | 1.0 | 0.6353 | −0,26 | 0.6360 | −0,37 | 365 |
| GLM gegen Rest r1 | 1.0 | 0.6257 | −1,22 | 0.6257 | −1,41 | 290 |
| Pool-Gewicht 0.25 → **V14** | 1.0 | 0.6375 | −0,04 | 0.6390 | −0,07 | 354 |
| kalibriert sigmoid | 1.0 | 0.6319 | −0,59 | 0.6323 | −0,74 | 323 |
| kalibriert isotonisch | 1.0 | 0.6316 | −0,63 | 0.6319 | −0,78 | 321 |

Bootstrap über Prompts gegen V6 (20 %): W 0.7 −0,38 [−0,63; −0,15]; r2 −0,26 [−0,52; 0,00]; Pool 0.25
−0,05 [−0,33; +0,22]; sigmoid −0,60 [−0,93; −0,30]; isotonisch −0,64 [−0,96; −0,33] (90 %-Intervalle).
Vollständig: `models/10_v6_gentle/results.csv`.

## Lesart
- Die Simulation will in jeder Zeile mehr GLM (höheres W ist besser). Genau diese Richtung hat der LB zweimal
  widerlegt (V11, V10). Die negativen Zahlen für V12/V13 sind deshalb kein Grund, sie nicht abzugeben. Sie sind die
  Gegenprobe zur Simulation.
- Die Kalibrierung (sigmoid/isotonisch, innere GroupKFold) macht den Klassifikator vor allem weicher. Sie senkt die
  GLM-Vorhersagen wie ein W um 0.6 und testet damit dieselbe Richtung wie V12. Pool-Gewicht 0.25 ist die eigene Achse
  (weniger Einfluss der Pool-Verteilung) und in der Simulation neutral. → V14 = Pool 0.25, sigmoid nur gebaut.

## Abgaben 18.11. um 11:00 (Public LB, Scores am Abend)
Außerdem abgegeben: öffentliche Fassung von V7 (`linuszarse/glm-likelihood-classifier-on-open-pool` v1,
Ref 58504553), Code identisch mit V7 (0.704), misst also auch das Rauschen. Abgabe 5 als Reserve zurückgehalten.

| Version | Kernel | Änderung gegen V6 | Public LB |
|---|---|---|---|
| V12 | `linuszarse/ntl-701-fork-v12-w07` v1 (Ref 58504556) | W 0.7 | – |
| V13 | `linuszarse/ntl-701-fork-v13-glmrest` v1 (Ref 58504560) | GLM gegen den Rest (r2) | – |
| V14 | `linuszarse/ntl-701-fork-v14-pool025` v2 (Ref 58504563) | Pool-Gewicht 0.25 | – |
| (Alternative) | `linuszarse/ntl-701-fork-v14-sigmoid` | kalibriert sigmoid | nicht abgegeben |
