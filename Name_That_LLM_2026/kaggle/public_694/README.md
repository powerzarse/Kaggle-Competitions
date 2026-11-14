# Fork: 0.694 | GLM-5.3 Likelihood + Qwen3.8-27B LoRA

Quelle: https://www.kaggle.com/code/lightningv08/0-694-glm-5-3-likelihood-qwen3-8-27b-lora (Apache 2.0, Credits im Notebook)
Unverändert importiert am 2026-11-14; Inputs (Datasets, Notebooks, Modelle) wie im Original in `kernel-metadata.json`.

Hebel für Verbesserungen (Zelle 15, `v1_stage.py`):
- `--max-cover` 0.20 (Anteil gescorter Testtexte) und Token-Budget 130.000
- nur die ersten 160 Tokens je Text (an das Training des Likelihood-Klassifikators gebunden)
- `--select-classes glm-5.3,glm-5.3-flash` → analoge Stufe für kimi-k3 denkbar

Push: `.venv/bin/kaggle kernels push -p kaggle/public_694` (2× T4, GPU-Kontingent!)
