"""Erzeugt kaggle/bert_pool/train_bert_pool.ipynb – ModernBERT auf dem Open Pool (Kaggle-GPU).

Das Notebook trainiert NUR auf Pool-Texten und testet auf der kompletten echten train.csv
(gleiches Protokoll wie Teil A in experiments/06_external_data/run_fast.py -> direkt mit der
SVM vergleichbar: SVM 3000/Modell 0.449, 6000/Modell 0.489).

Ausgaben in /kaggle/working:
    model/                   Gewichte + Tokenizer (Input für ein späteres Abgabe-Notebook)
    train_proba.parquet      Wahrscheinlichkeiten auf train.csv (für das Ensemble mit der SVM)
    log.csv                  Metriken je Epoche

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_bert.py --per-class 3000 --epochs 2   # Testlauf
    .venv/bin/kaggle kernels push -p kaggle/bert_pool
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle" / "bert_pool"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def build(model_name, per_class, epochs, lr, batch_size, max_len, eval_every):
    cells = [
        md("# Name That LLM – ModernBERT auf dem Open Pool\n\n"
           "Training nur auf *Name That LLM Open Pool* (crtproduction, CC BY-SA 4.0), "
           "Test auf der kompletten echten `train.csv`. Speichert Gewichte und Wahrscheinlichkeiten für das Ensemble."),
        code(f"""# Parameter
MODEL_NAME = "{model_name}"
PER_CLASS = {per_class or None}   # Pool-Texte je Modell (None = kompletter Pool)
EPOCHS = {epochs}
LR = {lr}
BATCH_SIZE = {batch_size}
MAX_LEN = {max_len}               # Pool-Texte: max. ~370 Tokens
EVAL_EVERY = {eval_every}         # zusätzlich alle N Schritte auf train.csv auswerten (0 = nur je Epoche)
WARMUP = 0.06
WEIGHT_DECAY = 0.01
SEED = 42"""),
        code("""import glob, math, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KEINE")
assert torch.cuda.is_available(), "GPU im Notebook aktivieren"
device = torch.device("cuda")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"{pattern} nicht gefunden"
    return Path(hits[0])

TRAIN_CSV = find("name-that-llm/train.csv")
POOL_PATH = find("ntl_open_pool.parquet")
OUT = Path("/kaggle/working")
print(TRAIN_CSV, POOL_PATH)"""),
        md("## Daten"),
        code("""real = pd.read_csv(TRAIN_CSV)
pool = pd.read_parquet(POOL_PATH, columns=["text", "model"]).sample(frac=1, random_state=SEED)
if PER_CLASS:
    pool = pool.groupby("model").head(PER_CLASS)
pool = pool.reset_index(drop=True)

classes = np.sort(real["model"].unique())
label_id = {c: i for i, c in enumerate(classes)}
assert set(pool["model"]) == set(classes)
FAMILIES = ("claude", "gpt", "gemini", "glm", "kimi", "qwen")
family = lambda m: next(f for f in FAMILIES if m.startswith(f))

# Klassengewichte (Pool ist unbalanciert, Metrik = Balanced Accuracy)
counts = pool["model"].value_counts().reindex(classes).values
class_w = torch.tensor(len(pool) / (len(classes) * counts), dtype=torch.float32, device=device)
print("Pool:", len(pool), "| train.csv:", len(real))
print(pool["model"].value_counts().to_string())"""),
        code("""tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(torch.utils.data.Dataset):
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

t = time.time()
ds_tr = TextDataset(pool["text"], pool["model"].map(label_id).values)
ds_real = TextDataset(real["text"])
print(f"Tokenisierung {time.time() - t:.0f}s")

collate = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=2)
# Auswertung nach Länge sortiert -> wenig Padding
order = np.argsort([len(x) for x in ds_real.enc["input_ids"]])
dl_real = DataLoader(torch.utils.data.Subset(ds_real, order), batch_size=BATCH_SIZE * 4, collate_fn=collate)"""),
        md("## Modell & Training"),
        code("""model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(classes),
    id2label=dict(enumerate(classes)), label2id=label_id,
).float().to(device)   # fp32-Gewichte; Rechnen in fp16 per autocast

no_decay = ("bias", "LayerNorm.weight", "norm.weight")
params = [
    {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
]
optim = torch.optim.AdamW(params, lr=LR)
steps = len(dl_tr) * EPOCHS
sched = get_linear_schedule_with_warmup(optim, int(WARMUP * steps), steps)
scaler = torch.amp.GradScaler("cuda")
loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

@torch.no_grad()
def predict_real():
    model.eval()
    out = []
    for batch in dl_real:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
    proba = np.empty((len(real), len(classes)), dtype=np.float32)
    proba[order] = np.concatenate(out)
    model.train()
    return proba

def score(proba):
    pred = classes[proba.argmax(1)]
    return {"balanced_accuracy": balanced_accuracy_score(real["model"], pred),
            "family_accuracy": np.mean([family(a) == family(b) for a, b in zip(real["model"], pred)])}

print(f"{steps} Schritte ({len(dl_tr)} je Epoche)")"""),
        code("""log, step, t0 = [], 0, time.time()
model.train()
for epoch in range(1, EPOCHS + 1):
    running = []
    for batch in dl_tr:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**batch).logits
        loss = loss_fn(logits.float(), labels)
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim); scaler.update(); sched.step()
        running.append(loss.item()); step += 1

        if step % 200 == 0:
            print(f"Epoche {epoch} Schritt {step}/{steps} | loss {np.mean(running[-200:]):.4f} "
                  f"| {(time.time() - t0) / 60:.1f} min", flush=True)
        if (EVAL_EVERY and step % EVAL_EVERY == 0) or step == epoch * len(dl_tr):
            m = score(predict_real())
            log.append({"epoch": epoch, "step": step, "train_loss": np.mean(running[-500:]),
                        "minutes": (time.time() - t0) / 60, **m})
            pd.DataFrame(log).to_csv(OUT / "log.csv", index=False)
            print(f"  >>> train.csv: balanced_acc={m['balanced_accuracy']:.4f} "
                  f"| family_acc={m['family_accuracy']:.4f}", flush=True)"""),
        md("## Speichern"),
        code("""proba = predict_real()
print("Endergebnis train.csv:", score(proba))
out = pd.DataFrame(proba, columns=classes)
out.insert(0, "id", real["id"].values)
out.to_parquet(OUT / "train_proba.parquet", index=False)

model.save_pretrained(OUT / "model")
tokenizer.save_pretrained(OUT / "model")
pd.DataFrame(log)"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train_bert_pool.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": "linuszarse/name-that-llm-modernbert-open-pool",
        "title": "Name That LLM ModernBERT Open Pool",
        "code_file": "train_bert_pool.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True,  # Training-Notebook: lädt ModernBERT von Hugging Face
        "dataset_sources": ["crtproduction/name-that-llm-open-pool"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [], "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"geschrieben: {OUT_DIR}/train_bert_pool.ipynb (+ kernel-metadata.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--per-class", type=int, default=3000, help="0 = kompletter Pool")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--eval-every", type=int, default=0)
    a = ap.parse_args()
    build(a.model, a.per_class, a.epochs, a.lr, a.batch_size, a.max_len, a.eval_every)
