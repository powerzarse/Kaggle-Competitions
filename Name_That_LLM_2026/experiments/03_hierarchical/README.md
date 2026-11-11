# 03 – Hierarchisch: erst Familie, dann Modell

Die EDA zeigt, dass die Familie (~68 %) gut zu erkennen ist, das Modell innerhalb der Familie aber schwer.
1. Familienklassifikator (6 Klassen)
2. Je Familie ein eigener Klassifikator nur auf deren Modelle (z. B. claude-opus-4-6 vs. -4-8 vs. -5 …)
3. P(Modell) = P(Familie) · P(Modell | Familie)

Hinweis: GLM wird oft als Claude erkannt (gleiche Typografie) und gehört daher in den Claude-Spezialisten mit hinein.

## Ergebnis Lauf 1 (GroupKFold 5, `run.py`)
Stufe 1 und Spezialisten = `tfidf_logreg` (Wort- + Char-TF-IDF + Stilo); Gruppen: claude_glm / gpt_kimi_qwen / gemini

| Variante | Balanced Acc. | Familien-Acc. | Log-Loss |
|---|---|---|---|
| flach (Baseline) | 0.352 | 0.672 | 2.00 |
| hierarchisch, weich | 0.351 | **0.691** | 1.99 |
| hierarchisch, hart | 0.348 | 0.688 | 5.60 |
| Blend 0.3·hier + 0.7·flach | 0.355 | 0.684 | 1.98 |

**Fazit:** kein relevanter Gewinn. Die Streuung zwischen den Folds liegt bei ca. ±1,3 Pp., der Unterschied ist Rauschen.
- Die Gruppe wird besser erkannt (+2 Pp. Familien-Accuracy), aber innerhalb der Gruppe bleiben die Spezialisten auf dem Niveau des flachen Modells.
- Grund: Die Spezialisten nutzen **dieselben Features**. TF-IDF trennt die Modelle innerhalb einer Familie schlicht nicht besser, nur weil weniger Klassen übrig sind.
- Die schwachen Klassen bleiben schwach: claude-fable-5 ~0.10, glm-5.3 0.12, gpt-5.6-sol 0.13.

**Wann wieder aufgreifen:** wenn es Spezialisten mit *anderen* Signalen gibt, z. B. einen feinjustierten Transformer je Gruppe oder gruppenspezifische Features. Die Struktur (`src/hierarchical.py`) nimmt beliebige Estimator für `group_model` und `specialist` an.
