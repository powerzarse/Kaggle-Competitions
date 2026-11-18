"""Erzeugt kaggle/public_702/public_en/ – das zur Veröffentlichung gedachte Notebook: Code genau wie V7, nur die
Beschreibung (Zelle 0) auf Englisch, mit Ergebnissen und Fehlschlägen. Eigener Kernel, damit die Versionshistorie
keine Vorversionen enthält.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_702/build_v7_hgb.py --W 1.0     # V7 neu erzeugen (bytegleich zum Repo)
    .venv/bin/python kaggle/public_702/build_public_en.py
    .venv/bin/kaggle kernels push -p kaggle/public_702/public_en    # privat; öffentlich erst nach Go (Datensatz zuerst)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "v7_hgb"
OUT = HERE / "public_en"
SLUG = "glm-likelihood-classifier-on-open-pool"
TITLE = "GLM likelihood classifier on Open Pool"   # Score später im Kaggle-UI voranstellen (max. 50 Zeichen)

INTRO = """# GLM likelihood classifier on Open-Pool texts

**One change to a public notebook, scored on three bases.** The base is unchanged: lightningv08's *0.702 | Soft Typography + GLM + Qwen3.8-27B LoRA* (code-identical to alexchilton's *Name That LLM 0.702: All Public + What Failed*, from lumibee's *NTL Kiro v44 Apex Grandmaster* line). All credits of the base are kept at the bottom.

## What changed
The GLM stage by José Padilla (estopadilla) scores the test texts with the most GLM probability under the real GLM-5.3 and GLM-5.3-Flash weights (jamesliebel's segment GGUFs). It turns the token log-probs into 54 likelihood features (first 160 tokens) and blends a small classifier into the stack: `argmax(z + W/T' · log p_classifier)` on the scored texts.

In the public notebooks that classifier is a logistic regression fitted on the 2,699 texts of train.csv. Here it is a `HistGradientBoostingClassifier` fitted on train.csv **plus 5,670 texts from crtproduction's Open Pool**, which I scored under both GLMs with the same `ntl-segrun` pipeline in a separate Kaggle run. Pool rows are weighted so that every class has the same total weight, half that of a train.csv class.

- Features: dataset [`linuszarse/ntl-glm-ll-train`](https://www.kaggle.com/datasets/linuszarse/ntl-glm-ll-train) (`ll_train.npz`: X 8,369 × 54, labels, sample weights).
- The classifier is fitted inside the notebook (about 17 s, no pickled model in the dataset). The cell starting with `# ours (V7)` patches `member_logp` in `v1_stage.py`.
- Everything else is as in the base: selection (top 15 %, 90,000 tokens), temperature, the max-changed guard, Apex, and blend weight W = 1.0.

## Public leaderboard
| Base | Base score | + this classifier |
|---|---|---|
| 0.694 line without GSQ (my own run of the base) | 0.685 | 0.695 |
| alexchilton 0.701 (GLM coverage by time left, up to 30 %) | 0.701 | **0.707** |
| lightningv08 0.702 (this notebook, fixed 15 %) | 0.702 | 0.704 |

The same code can move by up to 0.008 between submissions (alexchilton measured this), so one gain alone proves little. Three gains on three bases are consistent.

Locally (train.csv, GroupKFold by prompt, my own weaker stack as a stand-in with OOF 0.609, GLM stage on 20 % of the texts): 0.637 with the public logistic regression, 0.645 with this classifier.

## What did not work
Each change was submitted against the same 0.701 base, once with and once without the other change:

| Change | Public LB |
|---|---|
| Apex on the logits *after* the GLM stage, so it keeps the stage's non-GLM changes | 0.707 → 0.701 and 0.702 → 0.696 |
| Blend weight W = 1.5, GLM coverage capped at 25 % | 0.707 → 0.702 and 0.701 → 0.696 |

Both had looked good in my local simulation (+0.004 to +0.0055 and +0.002). My reading: the classifier recognises GLM well, but on the test it separates the other 16 classes poorly. The original Apex resets every non-GLM prediction to the stack and so filters that out. Simulations with a stand-in stack are worth trusting only for the GLM-or-not decision.

Also not worth it:
- More Open-Pool texts: the local learning curve is flat after about 4,000 texts (0 → 0.6383, 1,871 → 0.6409, 3,798 → 0.6449, 5,670 → 0.6453).
- Paired token features (GLM-5.3 vs Flash on the same tokens): +0.0026 locally on train.csv alone, less than the pool gives.
"""


def main():
    src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    nb = json.loads((SRC / "v7_hgb.ipynb").read_text())
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    assert src(nb["cells"][0]).startswith("# Fork V7"), "v7_hgb.ipynb hat sich geändert"
    nb["cells"][0]["source"] = INTRO
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    OUT.mkdir(exist_ok=True)
    name = f"{SLUG}.ipynb"
    (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    assert 5 <= len(TITLE) <= 50
    meta.update(id=f"linuszarse/{SLUG}", title=TITLE, code_file=name, is_private=True)
    assert "linuszarse/ntl-glm-ll-train" in meta["dataset_sources"]
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Kontrolle: alle Code-Zellen identisch mit V7
    a = [src(c) for c in json.loads((SRC / "v7_hgb.ipynb").read_text())["cells"]][1:]
    b = [src(c) for c in nb["cells"]][1:]
    assert a == b
    print(f"geschrieben: {OUT / name}, Kernel linuszarse/{SLUG}, Titel {TITLE!r}; Code-Zellen identisch mit V7")


if __name__ == "__main__":
    main()
