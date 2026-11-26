---
name: kaggle-autopilot
description: Startet den agentischen Trainings-Loop für eine Kaggle-Competition – ein Zyklus aus Lagebild, Hypothesen, Experimenten, Review, Blending, (Submission) und Zurückschreiben in .context. Verwenden bei "autopilot", "mach weiter", "verbessere das Modell", "nächster Zyklus" oder per /loop für Dauerbetrieb.
---

# Autopilot

Führe die Rolle **Orchestrator** aus: `.agents/agents/orchestrator.md` vollständig lesen und befolgen.

## Start
1. `COMP` bestimmen (`ls -d */.context`; Argument des Nutzers hat Vorrang).
2. Fehlen Daten → Skill `kaggle-data`.
3. Einen Zyklus laut Orchestrator-Rolle ausführen.

## Dauerbetrieb
- `/loop /kaggle-autopilot` → selbst getaktete Zyklen; wenn ein Trainingsjob läuft, auf dessen
  Benachrichtigung warten statt zu pollen.
- Ende des Loops: Deadline erreicht, Backlog leer und Researcher findet nichts Neues, oder der
  Mensch stoppt. Vor dem Ende die finalen 2 Submissions laut `rules.md` vorschlagen.

## Was „immer besser werden“ hier heißt
Jeder Zyklus hinterlässt: neue Ergebnisse (`EXPERIMENTS.md`), geprüftes Wissen (`learnings.md`),
einen neu priorisierten Backlog und einen aktuellen `state.md`. Der nächste Zyklus startet auf
diesem Wissen – auch in einer neuen Session ohne Gedächtnis.
