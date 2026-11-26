# Regeln & Autonomie (nur vom Menschen zu ändern)

## Autonomie-Einstellungen
```yaml
auto_submit: false          # true = Orchestrator darf ohne Rückfrage einreichen (siehe Submit-Policy)
max_experiments_per_cycle: 3
max_parallel_jobs: 1        # 4 Kerne → Trainingsjobs nacheinander
min_delta_auc: 0.0002       # kleinere Verbesserungen gelten als Rauschen
submissions_reserved_per_day: 2   # nie das Tageslimit ganz aufbrauchen
language: de
```

## Harte Regeln
1. **Validierung**: ausschließlich die festen Folds aus `src.cv.get_folds()`. Zielgröße = OOF-AUC.
   Keine eigenen Splits, kein Tuning auf dem Public LB.
2. **Kein Leakage**: Target-/Count-Encodings, Scaler, Feature-Selektion usw. nur innerhalb des
   Train-Teils eines Folds fitten. Test-Daten nie fürs Training mit Labels verwenden.
3. **Ein Experiment = ein Ordner** `experiments/NN_<name>/`, eine Änderung je Experiment,
   Hypothese im `note`. Ergebnisse nur über `src.tracking.save_experiment()`.
4. **Reproduzierbarkeit**: bestehende Feature-Sets/Modell-Defaults nie verändern – neue anlegen.
5. **Git**: nie Daten, Modelle, Submissions committen. `.context/`, `EXPERIMENTS.md`, Code schon.
6. **Submissions**: nur Dateien, die die Validierung im Skill `kaggle-submit` bestehen.

## Submit-Policy
- `auto_submit: false` → vor jeder Submission den Menschen fragen.
- `auto_submit: true` → nur einreichen, wenn
  (a) OOF-AUC ≥ beste bisher **eingereichte** OOF-AUC + `min_delta_auc`, und
  (b) danach noch ≥ `submissions_reserved_per_day` Submissions heute übrig sind.
- Letzter Tag: die 2 finalen Submissions = bestes Einzelmodell + bester Blend (nach CV).
