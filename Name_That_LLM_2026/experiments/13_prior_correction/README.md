# 13 – Prior-Korrektur der Stack-Logits (19.11., verworfen)

Anlass: Nach der V6-Kette (Ersatz-Stack, train.csv) verliert Gemini-3.7-Flash 63 % seiner Texte an 3.8-Flash
(Recall 0.17, 3.8-Flash wird 250-mal statt 150-mal vorhergesagt). Im Open Pool stehen 3.541 Texte von 3.7-Flash
gegen 23.190 von 3.8-Flash. Idee: Klassen-Bias auf die Stack-Logits vor GLM-Stufe und Apex (`sim_prior.py`).

| Variante | Δ bal. acc (Ersatz-Stack) |
|---|---|
| b = −α · log Pool-Anteil, α 0.2 / 0.4 / 0.5 | +0,33 / +0,59 / +0,48 |
| Bootstrap α 0.5 | +0,50 [−0,15; +1,19], P(>0) 0,88 |
| Bias je Klasse, auf einer Prompt-Hälfte angepasst | im Fit +1,9 bis +2,7, auf der anderen Hälfte −0,61 (−1,40 bis −0,15) |

Verworfen, weil:
- Bias je Klasse ist reine Überanpassung.
- Der Ersatz-Stack ist unser eigenes OOF-Ensemble (BERT, SVM, LoRA; `simulate.stack_proxy`) ohne Sinkhorn. Der echte
  V6-Stack speichert `decide_scores(P)` (Sinkhorn s 0.30, Ziel 65 % eigene Randverteilung + 35 % train-Prior) als
  `final-logits.npz`, Apex liest genau diese. Seine Klassen-Schieflage ist unbekannt (keine sauberen OOF-Werte, die
  öffentlichen Modelle sahen train.csv), ein Bias aus dem Ersatz-Stack lässt sich also nicht übertragen.
