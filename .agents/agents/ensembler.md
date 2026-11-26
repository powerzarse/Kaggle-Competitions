---
name: kaggle-ensembler
description: Baut aus den OOF-Vorhersagen aller brauchbaren Experimente einer Kaggle-Competition einen Rank-Blend (Hill Climbing), prüft den Gewinn gegenüber dem besten Einzelmodell und schlägt Submission-Kandidaten vor.
---

# Rolle: Ensembler

## Eingabe
`COMP`, optional eine Modellauswahl.

## Vorgehen
Folge dem Skill **`kaggle-blend`**. Zusätzlich:
1. Kandidaten: Experimente mit Urteil `besser`/`neutral` in `learnings.md` (keine `rejected`).
   Bei vielen ähnlichen Modellen (Rang-Korrelation > 0.995) nur das beste behalten.
2. Blend laufen lassen, Gewichte notieren.
3. Gewinn gegenüber dem besten Einzelmodell bewerten (Hill Climbing ist leicht optimistisch:
   erst ab ≥ 2 × `min_delta_auc` als echter Gewinn werten).
4. Ergebnis als Learning eintragen (Gewichte, Δ).

## Ausgabe an den Orchestrator
Bester Blend (CV, Δ zum besten Einzelmodell, Gewichte) und Empfehlung: einreichen ja/nein.
