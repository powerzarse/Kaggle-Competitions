# .context – Gedächtnis dieser Competition

Alles, was ein Agent wissen muss, um hier ohne Vorwissen weiterzuarbeiten. Dateien sind die
**einzige Quelle der Wahrheit** zwischen Sessions – was nicht hier steht, ist vergessen.

| Datei | Inhalt | Wer schreibt |
|---|---|---|
| `competition.md` | Fakten: Metrik, Daten, Deadline, Limits | einmalig / bei neuen Erkenntnissen |
| `rules.md` | harte Regeln + Autonomie-Einstellungen (`auto_submit`, Budgets) | nur der Mensch |
| `state.md` | aktueller Stand: Bestwerte, laufende Jobs, Submissions, nächster Schritt | Orchestrator, jeder Zyklus |
| `backlog.md` | priorisierte Hypothesen-Queue | Researcher (neu), Orchestrator (Status) |
| `learnings.md` | was funktioniert hat und was nicht, mit Δ-AUC | Reviewer |
| `journal.md` | kurzes Zyklus-Protokoll (append-only) | Orchestrator |

**Lesereihenfolge für jeden Agenten:** `rules.md` → `state.md` → `learnings.md` → `backlog.md`
(`competition.md` beim ersten Mal).

Zahlen (CV, LB) stehen außerdem maschinenlesbar in `models/*/meta.json` und `EXPERIMENTS.md`;
`python -m src.status` fasst beides zusammen.
