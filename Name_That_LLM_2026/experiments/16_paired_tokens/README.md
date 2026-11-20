# 16 – Gepaarte Token-Merkmale für den V6-Kopf (20.11.)

Idee (offen seit 17.11., Experiment 08): GLM-5.3 und Flash tokenisieren identisch. Je Token lässt sich also
vergleichen, welches Modell das Token wahrscheinlicher findet. Die bisherigen 54 Merkmale vergleichen nur Mittelwerte
je Modell (`d:…`). Gepaarte Merkmale (21, `evaluate.py`): Differenz lp(GLM-5.3) − lp(Flash) je Token als Mittel,
Streuung, t-Statistik, Anteil positiv, Quantile; z-Differenz, Entropie-Differenz, Rangvergleich, Korrelation, getrennt
nach niedriger und hoher Entropie. Sie kosten zur Laufzeit nichts. Am 17.11. fehlten Token-Scores für den Pool; jetzt
liegen sie vor (Experiment 14: train 2.699 Texte, Pool 4.734 Texte, jeweils unser Lauf, Schnitt 160 wie V6).

## Ergebnis (HGB wie V6, Pool-Gewicht 0.5, OOF über die Prompt-Folds, Kette Ersatz-Stack → GLM-Stufe 20 % → Apex)
| Kopf | Kopf 18 Kl. | GLM-Dreiklassen | NLL GLM-Texte | Kette Δ (90 %-Bootstrap) |
|---|---|---|---|---|
| 54 Merkmale, train + Pool (V6-Rezept) | 0.186 | 0.722 | 1.124 | 0 |
| 54 + gepaart, train + Pool | 0.192 | **0.744** | **1.043** | −0,22 [−0,56; +0,11] |
| nur gepaart + n_tok, train + Pool | 0.166 | 0.698 | 1.260 | −0,48 |
| 54 Merkmale, nur train | 0.182 | 0.654 | 2.129 | −0,67 |
| 54 + gepaart, nur train | 0.178 | 0.644 | 2.029 | −0,67 |

Der Kopf wird besser, die Kette nicht. Auch andere W und Abdeckungen ändern das nicht (W 0.7–1.6, 20/25 %:
gepaart immer ≤ V6-Rezept, beste Variante −0,08 bei 25 %). Log-Mischung beider Köpfe −0,11.

Warum: Die gepaarten Merkmale trennen GLM-5.3 von Flash besser (stärkstes Merkmal `p:d_t`, |r| 0.72). In der
Kette ist das nicht der Engpass, die Recalls von GLM-5.3 (0.593 / 0.587) und Flash (0.780 / 0.780) bleiben gleich.
Dafür zieht der Kopf mehr Nicht-GLM-Texte zu GLM (138 → 150), vor allem Claude-Fable-5 (Recall 0.215 → 0.181).
Engpass ist GLM gegen Fable, nicht GLM-5.3 gegen Flash.

Dazu kommt das Laufrauschen: Differenzmerkmale sind zwischen zwei GGUF-Läufen am wenigsten stabil (`d:logrank_mean`
r 0.41, `d:lp_mean` 0.44, Experiment 14). Gepaarte Merkmale sind reine Differenzen; zur Testzeit (anderer Lauf)
dürften sie eher noch schlechter abschneiden als hier (B → B).

Nebenbefund: Das V6-Rezept auf unseren eigenen Läufen (train B + 4.734 Pool-Texte) erreicht in der Kette 0.6379,
genauso viel wie V6 auf estopadillas train-Lauf + Pool aus Experiment 08 (A → A, Experiment 14).

Folgerung: verworfen, kein Notebook gebaut. Die GLM-Stufe bleibt in der Simulation ausgereizt (Experiment 14).
