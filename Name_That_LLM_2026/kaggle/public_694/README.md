# Fork: 0.694 | GLM-5.3 Likelihood + Qwen3.8-27B LoRA

Quelle: https://www.kaggle.com/code/lightningv08/0-694-glm-5-3-likelihood-qwen3-8-27b-lora (Apache 2.0, Credits im Notebook)
Unverändert importiert am 2026-11-14; Inputs (Datasets, Notebooks, Modelle) wie im Original in `kernel-metadata.json`.

Hebel für Verbesserungen (Zelle 15, `v1_stage.py`):
- `--max-cover` 0.20 (Anteil gescorter Testtexte) und Token-Budget 130.000
- nur die ersten 160 Tokens je Text (an das Training des Likelihood-Klassifikators gebunden)
- `--select-classes glm-5.3,glm-5.3-flash` → analoge Stufe für kimi-k3 denkbar

Push: `.venv/bin/kaggle kernels push -p kaggle/public_694` (2× T4, GPU-Kontingent!)

## Varianten und Public LB
| Ordner | Änderung | Public LB |
|---|---|---|
| `v1_uncertain/` | GLM-Stufe wählt unsichere Texte (glm, glm-flash, fable), Abdeckung 0.30 | kein Score |
| `v2_gsq03/` | GSQ-27B-Anteil 0.12 → 0.03 (V2b: Assert repariert) | kein Score |
| `v3_both/` | V1 + V2 (V3b: Assert repariert) | kein Score |
| `v4_nogsq/` | GSQ-27B-Stufe aus (`gsq_share 0`), damit das 9-h-Limit hält | 0.685 |
| `v5_hgb/` | V4 + GLM-Klassifikator HGB auf train + Pool (`build_v5_hgb.py`, Daten in `v5_hgb_data/`) | 0.695 |

Nachfolger auf der 0.701-Basis: `kaggle/public_701/` (V6, 0.707).
