# Learnings

Kurze, überprüfte Erkenntnisse. Jede Zeile: Was wurde geändert → Δ OOF-AUC → Urteil → Warum (Vermutung).
Neueste oben. Der Researcher leitet daraus neue Hypothesen ab; nichts doppelt testen, was hier steht.

## Gesichert
- [2026-11-26] 02_xgb_v1 (H01): XGBoost statt LGBM, v1 → Δ +0.00022 (3/5 Folds besser, 1 gleich) → neutral.
  Rang-Korrelation zu LGBM 0.988 (< 0.99) → als Blend-Partner behalten.
- [2026-11-26] **CV ≈ LB**: 01_lgbm_baseline OOF 0.94631 vs. Public LB 0.94630 → der CV ist
  vertrauenswürdig, auf CV optimieren.
- [2026-11-26] LGBM (lr 0.03, 63 Leaves) auf v1: OOF 0.94631, Fold-Std 0.0033. Referenzpunkt.

## Offen / Hinweise
- Fold-Std (0.0033) ist groß gegenüber typischen Gewinnen → Vergleiche immer gepaart auf
  denselben Folds (OOF-AUC), nicht über Fold-Mittelwerte.
