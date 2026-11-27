# 10_glm_v6 – GLM (LogReg) auf logit(Wert-/Ziffern-TE)

**Hypothese (H15b):** Das Signal ist fast additiv. Ein GLM auf den TE-Logits ist deshalb allein fast so stark wie die flachen
Bäume, macht aber andere Fehler und sollte im Blend deutlich mehr bringen als eine weitere GBDT-Variante.

**Änderung:**
- `src/models.py`: neue Factory `glm_vte`. Sie nutzt `vte_frames` wie `lgbm_vte`/`xgb_vte` (m=20, 5 innere Folds, Seed 42).
  Danach clippt sie die 54 TE-Spalten auf [1e-4, 1−1e-4], bildet logit(TE) und nimmt die 6 booleschen Flags sowie die
  5 Kategorien als One-Hot dazu. Das ergibt 65 Eingangsspalten, nach One-Hot 95. Daraus baut sie eine Pipeline aus
  ColumnTransformer (OneHotEncoder(handle_unknown="ignore") + StandardScaler) und `LogisticRegression` (lbfgs, max_iter 3000).
  Rohe Zahlen und Cents gehen nicht ein, weil sie als exakter Wert bzw. Ziffer schon im TE stecken. Feature-Set `v6` ist unverändert.
- **C-Wahl (Fold 0):** C 0.1 → 0.94904, C 1 → 0.94906, C 10 → 0.94906. Bei 560k Zeilen ist die Regularisierung praktisch
  irrelevant, deshalb habe ich **C = 1.0** genommen. lbfgs konvergiert in 31–40 Iterationen.

**Leakage-Schutz:** Das TE läuft fold-intern (X_tr bekommt OOF-Werte über innere Folds, Val und Test das Mapping aus ganz X_tr).
Scaler und One-Hot werden in der Pipeline nur auf X_tr gefittet. Labels von Val werden nicht genutzt, es gibt kein Early Stopping.

**Ergebnis:** OOF-AUC **0.95010** (Fold-Mittel 0.95013 ± 0.00335).
Folds [0.94906, 0.95064, 0.95516, 0.95101, 0.94477] gegenüber 07 [0.94926, 0.95115, 0.95508, 0.95142, 0.94435]
→ **Δ −0.00009 zu 07** (2/5 Folds besser). Das lineare Modell ist damit praktisch so stark wie die flachen Bäume.
Rang-Korrelation 0.982 zu 07 und 09, 0.989 zu 08.

**Blend-Probe (Rank-Mittel der OOF-Ränge, In-Sample, nichts gespeichert):**
- 07 + 10: w(10) 0.1/0.2/0.3/0.4/0.5 → 0.95036/0.95048/0.95056/0.95060/**0.95061** (Δ bis +0.00042, alle 5/5 Folds)
- 07 + 09 + 10: gleichgewichtet (⅓ je Modell) **0.95062** (+0.00043, 5/5), 0.4/0.2/0.4 bzw. 0.3/0.3/0.4 → **0.95064** (+0.00044, 5/5)
- 07 + 09 allein bringt nur +0.00011 (w 0.5). Der Gewinn kommt fast vollständig von 10; 09 legt nur ~+0.00002 obendrauf.

**Laufzeit:** 97 s für den vollen 5-Fold-Lauf (TE ~4 s je Fold, LogReg ~10 s). Keine Warnungen im Log.
