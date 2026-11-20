# 14 – GLM-Stufe mit 256 statt 160 Tokens (20.11.)

Anlass: `omerfarukmerey/0-706-glm-fable-routing-256-token-likelihood` (Public **0.706**) ändert gegenüber lightningv08s
0.704 nur den Schnitt der GLM-Stufe. Der Kopf (estopadillas LR-Rezept) wird auf den ersten 256 statt 160 Tokens neu
angepasst, das Token-Budget steigt von 90k auf 130k, damit die Abdeckung bei 15 % bleibt. Ergebnis: +0,002 auf dem LB.
Frage: Hilft das auch auf V6 (0.707)? V6 nutzt unseren HGB mit Pool-Daten und deckt nach Restzeit ab, nicht fest.

## Daten
- `kaggle/build_notebook_glm_scores.py --source train` → Kernel `linuszarse/ntl-glm-scores-train512`: alle 2.699
  train-Texte unter GLM-5.3-Flash und GLM-5.3, erste 512 Tokens, **Token-Scores gespeichert** (4,95 h auf 2× T4).
  Ablage: `models/14_glm_256/train512/out/`. Merkmale für jeden Schnitt L: `features.one(s[:L], len(text))`.
- `--source pool` → `linuszarse/ntl-glm-scores-pool512`: dieselbe Pool-Auswahl wie Experiment 08 mit 512 Tokens
  (20.11., 9 h, 5 von 7 Häppchen = 4.734 Texte, Ablage `models/14_glm_256/pool512/out/`). Gedacht für HGB-256 +
  Pool und gepaarte Token-Merkmale, noch nicht ausgewertet.
- Öffentlicher 256er-Kopf: `ll_model_L256_public.json` (aus dem 0.706-Notebook, unverändert).

Prüfungen: Merkmale bei 512 = Merkmale der Pipeline (max. 3e-7). LR-256 auf unseren Merkmalen gegen den öffentlichen
Kopf: Korrelation der Koeffizienten 0.94. 38 % der Texte sind länger als 160 Tokens, 11 % länger als 256.
Unser Lauf (B) gegen estopadillas train-Lauf (A) bei 160 Tokens: Korrelation je Merkmal Median 0.97, am schlechtesten
`d:logrank_mean` 0.41 und `d:lp_mean` 0.44. Das ist das Batch-Rauschen der MoE-GGUF-Läufe.

## Ergebnisse (`evaluate.py`, `evaluate_crossrun.py`)
Kopf allein (OOF, 18 Klassen): LR-160 0.199 → **LR-256 0.216** → LR-320 0.221 (wie im 0.706-Notebook angegeben).
V6-Kopf (HGB-160 + Pool) allein 0.190, aber bester GLM-Dreiklassen-Wert.

V6-Kette (Ersatz-Stack → GLM-Stufe → Apex). Referenz ist V6 so, wie er zur Testzeit läuft: auf Lauf A trainiert, auf
Lauf B bewertet (0.6356; A→A wäre 0.6379, das Laufrauschen kostet also ~0,2 Pp.). Gleiche Zeit = so viele Texte,
wie in die Token-Zeilen von 20 % bei 160 passen: 16,7 % bei 256.

| Kopf | 20 % | gleiche Zeit (16,7 %) |
|---|---|---|
| V6: HGB-160 + Pool | 0 | – |
| V18: LR-256 (öffentlich) | −0,52 | −0,67 |
| V19: 0.5 V6 + 0.5 LR-256 | −0,33 | −0,56 |
| 0.7 V6 + 0.3 LR-256 | −0,26 | −0,52 |
| HGB-256 nur train | −0,15 | −0,41 |
| LR-160 (wie öffentlich vor V5) | −0,59 | – |

- Auf demselben Lauf bringt der längere Schnitt ~+0,2 (LR) bis +0,3 (HGB) bei gleicher Abdeckung. Das passt zum
  +0,2 des 0.706-Notebooks, wo die Abdeckung fest bleibt. Die Sim reproduziert diese öffentliche LB-Änderung.
- In V6 begrenzt die Restzeit die Abdeckung. Bei 256 Tokens sinkt sie auf 16,7 %, und der Gewinn ist weg.
- Unser HGB mit Pool-Daten ist mehr wert als der längere Schnitt. Ihn durch LR-256 zu ersetzen (V18) kostet.

Weitere Proben, alle ohne Hebel:
- HGB auf beiden train-Läufen (A+B) als Schutz gegen Laufrauschen: nicht besser (über Läufe hinweg 0.6356 in beide Richtungen).
- Fable-Routing (Auswahl nach `mass + β·min(p_fable, mass)`, lightningv08 0.702 → 0.704): +0,04 für jedes β. In V6
  setzt Apex alles außer den GLM-Entscheidungen zurück.
- Was die Stufe in V6 ändert: 156 Vorhersagen, ohne Stufe 0.6071 → mit 0.6356. Fable → GLM-5.3: 18 Wechsel, davon
  16 danach richtig (vorher 1). Fable-Recall sinkt 0.29 → 0.22, GLM-5.3 steigt 0.23 → 0.55, Flash 0.44 → 0.74.
  Eine Bremse für Fable → GLM hat keinen Ansatzpunkt.

## Gebaut, nicht abgegeben
`kaggle/public_701/build_v18_l256.py`:
- V18 `linuszarse/ntl-701-fork-v18-l256` (v1, Commit-Lauf OK): Stufe mit 256 Tokens, öffentlicher LR-256-Kopf.
- V19 `v19_mix050`: Log-Mix 0.5 HGB-160 (aus den 256er-Token-Scores nachgerechnet, mit Laufzeitprüfung) + 0.5 LR-256.
  Lokal Ende-zu-Ende getestet (Ersatz-Pipeline, `allow_pickle` für die Objekt-IDs von run_pipeline nötig), kein Commit-Lauf.

Folgerung: Für V6 kein Gewinn durch den längeren Schnitt, solange die Stufe nicht mehr Zeit bekommt. Die GLM-Stufe
von V6 ist in der Simulation ausgereizt (alle Kopf-, Auswahl- und Gewichtsvarianten ≤ +0,1).
