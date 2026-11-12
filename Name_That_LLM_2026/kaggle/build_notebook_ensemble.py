"""Erzeugt kaggle/ensemble/submission_ensemble.ipynb – Abgabe: SVM + ModernBERT (Stufe 2) [+ Qwen-LoRA] + Sinkhorn.

- SVM: wird im Notebook auf Open Pool + train.csv trainiert (Code aus src/, wie kaggle/svm_pool).
- ModernBERT: fertiges Stufe-2-Modell aus dem Output von linuszarse/name-that-llm-modernbert-stage2.
- Mischung: W_SVM * softmax(SVM-Score * T) + (1 - W_SVM) * BERT, danach Sinkhorn-Ausgleich (Stärke SINKHORN).
  Gewichte aus den OOF-Vorhersagen auf train.csv bestimmen.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_ensemble.py --w-svm 0.4 --svm-t 5 --sinkhorn 0
    # mit Qwen-LoRA als drittem Mitglied (Output von kaggle/lora, Ordner final/):
    .venv/bin/python kaggle/build_notebook_ensemble.py --w-svm 0.2 --w-lora 0.4 --lora-base 4b
    .venv/bin/kaggle kernels push -p kaggle/ensemble
"""
import argparse
import json
from pathlib import Path

from build_notebook_lora import BASES
from build_notebook_svm import ROOT, code, md, strip_module

OUT_DIR = ROOT / "kaggle" / "ensemble"
BERT_KERNEL = "linuszarse/name-that-llm-modernbert-stage2"


def lora_classes_src() -> str:
    """LoRALinear, add_lora und Heads wörtlich aus dem Trainings-Notebook (eine Quelle, kein Auseinanderlaufen)."""
    src = (ROOT / "kaggle" / "build_notebook_lora.py").read_text()
    a = src[src.index("LORA_TARGETS = ("):src.index("def load_trunk():")]
    b = src[src.index("class Heads(nn.Module):"):src.index("lm = load_trunk()")]
    return (a + b).replace('\\"\\"\\"', '"""').strip()


LORA_CELL = """import json, math
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel

LORA_DIR = find("final/meta.json").parent
meta = json.load(open(LORA_DIR / "meta.json"))
LORA_R, LORA_ALPHA, LORA_DROPOUT = meta.get("lora_r", 16), meta.get("lora_alpha", 32), 0.0
KEEP_LAYERS, DEPTHS, K = meta["keep_layers"], meta["depths"], len(meta["classes"])

def find_base(hidden):
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if c.get("model_type") == "qwen3" and c.get("hidden_size") == hidden:
            return Path(cfg).parent
    raise FileNotFoundError("Qwen3-Basismodell als Input einbinden")

__CLASSES__

del bert; torch.cuda.empty_cache()
BASE_DIR = find_base(meta["base_hidden"])
ltok = AutoTokenizer.from_pretrained(BASE_DIR)
if ltok.pad_token is None:
    ltok.pad_token = ltok.eos_token
ltok.padding_side = "right"
cfg = AutoConfig.from_pretrained(BASE_DIR)
cfg.num_hidden_layers, cfg.use_cache = KEEP_LAYERS, False
if getattr(cfg, "layer_types", None) is not None:
    cfg.layer_types = cfg.layer_types[:KEEP_LAYERS]
lm = AutoModel.from_pretrained(BASE_DIR, config=cfg, dtype=torch.float16 if dev.type == "cuda" else torch.float32).to(dev)
lm.norm = nn.Identity()
n_lora = add_lora(lm)
res = lm.load_state_dict(torch.load(LORA_DIR / "lora.pt", map_location="cpu"), strict=False)
assert not res.unexpected_keys and not [k for k in res.missing_keys if "lora_" in k], "LoRA-Gewichte passen nicht zum Modell"
heads = Heads(cfg.hidden_size).to(dev)
heads.load_state_dict(torch.load(LORA_DIR / "heads.pt", map_location="cpu"))
lm.eval(); heads.eval()

texts = list(test["text"]); order = np.argsort([len(t) for t in texts])
P_lora = np.zeros((len(texts), K), np.float32)
with torch.no_grad():
    for i in range(0, len(texts), 32):
        sel = order[i:i + 32]
        b = ltok([texts[j] for j in sel], truncation=True, max_length=meta["max_len"], padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            hs = lm(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True).hidden_states
        P_lora[sel] = torch.softmax(heads.plain(heads.features(hs, b["attention_mask"])), -1).cpu().numpy()
P_lora = pd.DataFrame(P_lora, columns=meta["classes"])[classes].values
del lm; torch.cuda.empty_cache()
print(f"LoRA ({n_lora} Schichten) fertig nach {(time.time() - T0) / 60:.1f} min")"""


def build(w_svm, svm_t, sinkhorn, real_weight, C, w_lora=0.0, lora_base="4b"):
    features_src = strip_module((ROOT / "src" / "features.py").read_text())
    linear_src = strip_module((ROOT / "src" / "linear_large.py").read_text())
    cells = [
        md("# Name That LLM – Ensemble: LinearSVC + ModernBERT (Stufe 2) + Sinkhorn\n\n"
           "Inputs: Competition `name-that-llm`, Dataset `crtproduction/name-that-llm-open-pool`, "
           f"Notebook-Output `{BERT_KERNEL}` (Ordner `model_stage2`). Kein Internet."),
        code(f"""W_SVM = {w_svm}          # Anteil SVM in der Mischung
W_LORA = {w_lora}        # Anteil Qwen-LoRA (0 = nicht verwendet); Rest: ModernBERT
SVM_T = {svm_t}          # Temperatur: softmax(SVM-Score * SVM_T)
SINKHORN = {sinkhorn}    # 0 = aus, 1 = voller Klassenausgleich
REAL_WEIGHT = {real_weight}
C = {C}
SEED = 42"""),
        code("""import glob, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

T0 = time.time()
def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"{pattern} nicht gefunden – Input eingebunden?"
    return Path(hits[0])

DATA_DIR = find("name-that-llm/train.csv").parent
POOL_PATH = find("ntl_open_pool.parquet")
BERT_DIR = find("model_stage2/config.json").parent
OUT_DIR = Path("/kaggle/working")
print(DATA_DIR, POOL_PATH, BERT_DIR, "| GPU:", torch.cuda.is_available())"""),
        md("## Features & Modell (aus `src/features.py` und `src/linear_large.py`)"),
        code(features_src),
        code(linear_src),
        md("## Daten"),
        code("""train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
pool = pd.read_parquet(POOL_PATH, columns=["text", "model"])
classes = np.sort(train["model"].unique())
assert set(pool["model"]) == set(classes)
print("train", train.shape, "| pool", pool.shape, "| test", test.shape)"""),
        md("## 1. LinearSVC (Pool + train.csv)"),
        code("""vec = LargeVectorizer()
X_pool = vec.fit_transform(pool["text"])
X_train = vec.transform(train["text"])
X = sp.vstack([X_pool, X_train], format="csr")
y = np.concatenate([pool["model"].values, train["model"].values])
svm = make_svm(C).fit(X, y, sample_weight=sample_weights(len(pool), len(train), REAL_WEIGHT))
del X, X_pool
dec = svm.decision_function(vec.transform(test["text"]))
P_svm = pd.DataFrame(softmax(dec * SVM_T, axis=1), columns=svm.classes_)[classes].values
print(f"SVM fertig nach {(time.time() - T0) / 60:.1f} min")"""),
        md("## 2. ModernBERT Stufe 2"),
        code("""dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(BERT_DIR)
bert = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).float().to(dev).eval()
bert_classes = [bert.config.id2label[i] for i in range(bert.config.num_labels)]
assert set(bert_classes) == set(classes)

enc = tok(list(test["text"]), truncation=True, max_length=384)
order = np.argsort([len(x) for x in enc["input_ids"]])
items = [{k: enc[k][i] for k in enc} for i in order]
dl = DataLoader(items, batch_size=64, collate_fn=DataCollatorWithPadding(tok, pad_to_multiple_of=8))
out = []
with torch.no_grad():
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(torch.softmax(bert(**b).logits.float(), -1).cpu().numpy())
P_bert = np.empty((len(test), len(bert_classes)), dtype=np.float32)
P_bert[order] = np.concatenate(out)
P_bert = pd.DataFrame(P_bert, columns=bert_classes)[classes].values
print(f"BERT fertig nach {(time.time() - T0) / 60:.1f} min")"""),
        *([md("## 3. Qwen-LoRA (Stufe 1 + 2)"), code(LORA_CELL.replace("__CLASSES__", lora_classes_src()))] if w_lora else []),
        md("## Mischung + Sinkhorn-Ausgleich"),
        code("""P = W_SVM * P_svm + (1 - W_SVM - W_LORA) * P_bert
if W_LORA:
    P = P + W_LORA * P_lora

def sinkhorn(P, strength, iters=100):
    Q = P.copy(); target = np.full(P.shape[1], len(P) / P.shape[1])
    for _ in range(iters):
        Q *= target / Q.sum(0); Q /= Q.sum(1, keepdims=True)
    return np.exp((1 - strength) * np.log(P + 1e-12) + strength * np.log(Q + 1e-12))

if SINKHORN and len(test) >= 50:      # nicht auf den 3 Beispieltexten
    P = sinkhorn(P, SINKHORN)
pred = classes[P.argmax(1)]
print("Übereinstimmung SVM/BERT:", round(float((P_svm.argmax(1) == P_bert.argmax(1)).mean()), 3))"""),
        md("## Submission"),
        code("""sub = pd.DataFrame({"id": test["id"], "model": pred})
assert list(sub.columns) == ["id", "model"]
assert sub.notna().all().all(), "Nullwerte"
assert sub["id"].is_unique, "doppelte IDs"
assert len(sub) == len(test) and set(sub["id"]) == set(test["id"]), "nicht jede Test-ID genau einmal"
assert set(sub["model"]) <= set(classes) and len(classes) == 18, "unbekanntes Label"
sub.to_parquet(OUT_DIR / "submission.parquet", index=False)
print(sub["model"].value_counts().head(10), f"\\ngesamt {(time.time() - T0) / 60:.1f} min")
sub.head()"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "submission_ensemble.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": "linuszarse/name-that-llm-ensemble",
        "title": "Name That LLM Ensemble",
        "code_file": "submission_ensemble.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["crtproduction/name-that-llm-open-pool"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [BERT_KERNEL] + ([f"linuszarse/name-that-llm-lora-qwen3-{lora_base}"] if w_lora else []),
        "model_sources": [BASES[lora_base][0]] if w_lora else [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"geschrieben: {OUT_DIR}/submission_ensemble.ipynb (+ kernel-metadata.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-svm", type=float, default=0.5)
    ap.add_argument("--svm-t", type=float, default=5.0)
    ap.add_argument("--sinkhorn", type=float, default=0.6)
    ap.add_argument("--real-weight", type=float, default=5)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--w-lora", type=float, default=0.0, help="Anteil der Qwen-LoRA (0 = ohne)")
    ap.add_argument("--lora-base", choices=BASES, default="4b")
    a = ap.parse_args()
    assert 0 <= a.w_svm + a.w_lora <= 1
    build(a.w_svm, a.svm_t, a.sinkhorn, a.real_weight, a.C, a.w_lora, a.lora_base)
