"""Erzeugt kaggle/bert_stage2/train_bert_stage2.ipynb – Stufe 2: ModernBERT aus Stufe 1 (Pool)
auf der echten train.csv weitertrainieren.

1. 5-Fold-CV (GroupKFold nach prompt_id, identische Folds wie lokal) -> OOF je Epoche
   => zeigt, wie viel Stufe 2 bringt und wie viele Epochen optimal sind.
2. Finales Modell auf der kompletten train.csv -> /kaggle/working/model_stage2 (für die Abgabe).

Input: Output des Stufe-1-Notebooks (linuszarse/name-that-llm-modernbert-open-pool, Ordner model/).

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_bert_stage2.py --epochs 3 --lr 2e-5
    .venv/bin/kaggle kernels push -p kaggle/bert_stage2
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "kaggle" / "bert_stage2"
STAGE1_KERNEL = "linuszarse/name-that-llm-modernbert-open-pool"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def fold_map() -> dict:
    from src.cv import get_folds
    from src.data import load_train
    df = load_train()
    folds = get_folds(df)
    m = dict(zip(df["prompt_id"], folds.tolist()))
    assert df["prompt_id"].map(m).tolist() == folds.tolist()
    return m


def build(epochs, lr, batch_size, max_len, full_fit):
    fmap = fold_map()
    cells = [
        md("# Name That LLM – ModernBERT Stufe 2 (Pool → train.csv)\n\n"
           "Startet vom Stufe-1-Modell (nur Pool trainiert) und trainiert auf der echten `train.csv` weiter. "
           "5-Fold-CV nach Prompt für die Messung, danach ein finales Modell auf allen Daten."),
        code(f"""EPOCHS = {epochs}
LR = {lr}
BATCH_SIZE = {batch_size}
MAX_LEN = {max_len}
WARMUP = 0.1
WEIGHT_DECAY = 0.01
FULL_FIT = {full_fit}          # finales Modell auf allen train.csv-Texten trainieren und speichern
SEED = 42
# Zuordnung prompt_id -> Fold (identisch mit src/cv.get_folds, lokal erzeugt)
FOLD_OF_PROMPT = {json.dumps(fmap)}"""),
        code("""import glob, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

assert torch.cuda.is_available(), "GPU aktivieren"
device = torch.device("cuda")

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"{pattern} nicht gefunden"
    return Path(hits[0])

TRAIN_CSV = find("name-that-llm/train.csv")
STAGE1 = find("model/config.json").parent
STAGE1_PROBA = find("train_proba.parquet")
OUT = Path("/kaggle/working")
print(TRAIN_CSV, STAGE1)

real = pd.read_csv(TRAIN_CSV)
folds = real["prompt_id"].map(FOLD_OF_PROMPT).values
assert not np.isnan(folds.astype(float)).any()
tokenizer = AutoTokenizer.from_pretrained(STAGE1)
probe = AutoModelForSequenceClassification.from_pretrained(STAGE1)
classes = np.array([probe.config.id2label[i] for i in range(probe.config.num_labels)])
del probe
label_id = {c: i for i, c in enumerate(classes)}
y = real["model"].map(label_id).values
FAMILIES = ("claude", "gpt", "gemini", "glm", "kimi", "qwen")
family = lambda m: next(f for f in FAMILIES if m.startswith(f))
print(len(real), "Texte | Folds:", np.bincount(folds))"""),
        code("""class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)
        self.labels = labels
    def __len__(self):
        return len(self.enc["input_ids"])
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item

collate = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
ds_all = TextDataset(real["text"], y)

@torch.no_grad()
def predict(model, idx):
    model.eval()
    dl = DataLoader(torch.utils.data.Subset(ds_all, idx), batch_size=64, collate_fn=collate)
    out = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(**b).logits.float(), -1).cpu().numpy())
    model.train()
    return np.concatenate(out)

def train(tr_idx, va_idx=None):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(STAGE1).float().to(device)
    no_decay = ("bias", "LayerNorm.weight", "norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=LR)
    dl = DataLoader(torch.utils.data.Subset(ds_all, tr_idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    steps = len(dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(optim, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    per_epoch = []
    if va_idx is not None:
        per_epoch.append(predict(model, va_idx))          # Epoche 0 = reines Stufe-1-Modell
    model.train()
    for epoch in range(EPOCHS):
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**b).loss
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update(); sched.step()
        if va_idx is not None:
            per_epoch.append(predict(model, va_idx))
    return model, per_epoch"""),
        md("## 5-Fold-CV"),
        code("""t0 = time.time()
oof = np.zeros((EPOCHS + 1, len(real), len(classes)), dtype=np.float32)
for k in range(5):
    tr_idx, va_idx = np.where(folds != k)[0], np.where(folds == k)[0]
    model, per_epoch = train(tr_idx, va_idx)
    for e, p in enumerate(per_epoch):
        oof[e, va_idx] = p
    accs = [balanced_accuracy_score(y[va_idx], p.argmax(1)) for p in per_epoch]
    print(f"Fold {k}: " + " | ".join(f"Ep{e} {a:.4f}" for e, a in enumerate(accs)) + f" | {(time.time()-t0)/60:.1f} min", flush=True)
    del model; torch.cuda.empty_cache()

rows = []
for e in range(EPOCHS + 1):
    pred = classes[oof[e].argmax(1)]
    rows.append({"epoch": e, "balanced_accuracy": balanced_accuracy_score(real["model"], pred),
                 "family_accuracy": np.mean([family(a) == family(b) for a, b in zip(real["model"], pred)])})
    out = pd.DataFrame(oof[e], columns=classes); out.insert(0, "id", real["id"].values)
    out.to_parquet(OUT / f"oof_stage2_ep{e}.parquet", index=False)
res = pd.DataFrame(rows); res.to_csv(OUT / "cv_stage2.csv", index=False)
print("OOF gesamt (Epoche 0 = nur Stufe 1):"); res"""),
        md("## Finales Modell auf allen train.csv-Texten"),
        code("""if FULL_FIT:
    model, _ = train(np.arange(len(real)))
    model.save_pretrained(OUT / "model_stage2")
    tokenizer.save_pretrained(OUT / "model_stage2")
    print("gespeichert:", OUT / "model_stage2", f"| gesamt {(time.time()-t0)/60:.1f} min")"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train_bert_stage2.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": "linuszarse/name-that-llm-modernbert-stage2",
        "title": "Name That LLM ModernBERT Stage2",
        "code_file": "train_bert_stage2.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False,   # alles kommt aus den Inputs -> testet gleich die Offline-Fähigkeit
        "dataset_sources": [],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [STAGE1_KERNEL], "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"geschrieben: {OUT_DIR}/train_bert_stage2.ipynb (+ kernel-metadata.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--no-full-fit", action="store_true")
    a = ap.parse_args()
    build(a.epochs, a.lr, a.batch_size, a.max_len, not a.no_full_fit)
