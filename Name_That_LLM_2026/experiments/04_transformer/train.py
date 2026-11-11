"""Experiment 04: Transformer-Finetuning (DeBERTa-v3 / ModernBERT) mit denselben
GroupKFold-Folds wie alle anderen Modelle.

Aufruf aus dem Ordner Name_That_LLM_2026/:

    # 1) Kurzer Funktionstest (~1-2 min): kleiner Datenausschnitt, 1 Epoche, Fold 0
    .venv/bin/python experiments/04_transformer/train.py --smoke

    # 2) Ein Fold zum Einschätzen von Laufzeit & Score
    .venv/bin/python experiments/04_transformer/train.py --folds 0

    # 3) Alle 5 Folds -> OOF für Ensemble + gemittelte Test-Vorhersage
    .venv/bin/python experiments/04_transformer/train.py

    # Alternative: DeBERTa-v3 (auf dem Mac ~2x langsamer, mehr Speicher)
    .venv/bin/python experiments/04_transformer/train.py --model microsoft/deberta-v3-base --lr 2e-5

Speicher (Mac, 16 GB): ModernBERT-base, Batch 8, Gradient Checkpointing ~4.4 GB GPU,
~1.7 s/Schritt -> ~7-8 min/Epoche, ~30 min/Fold. Ein GPU-Speicherlimit verhindert, dass
macOS in den Swap läuft (dann lieber ein klarer Out-of-Memory-Fehler).

Folds können auch einzeln nacheinander gestartet werden (--folds 0, dann --folds 1 ...);
sobald alle 5 Folds eines Runs vorliegen, wird die OOF-Datei automatisch geschrieben.

Ausgaben je Run in models/04_transformer/<run_name>/:
    config.json, log.csv (Metriken je Epoche), fold{k}_val.npy, fold{k}_test.npy,
    oof_<run_name>.parquet (wenn alle Folds fertig), test_<run_name>.parquet
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# MPS-Speicherlimit (Anteil des empfohlenen Maximums) – muss vor dem torch-Import gesetzt sein
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)
from transformers.utils import logging as hf_logging

from src.config import MODEL_DIR
from src.cv import N_SPLITS, evaluate, get_folds, save_oof
from src.data import TARGET, TEXT_COL, load_test, load_train

hf_logging.set_verbosity_error()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument("--folds", default="all", help="'all' oder z.B. '0' / '0,1,2'")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5, help="ModernBERT ~5e-5, DeBERTa ~2e-5")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-len", type=int, default=512, help="Texte haben max. ~400 Tokens")
    p.add_argument("--no-grad-ckpt", action="store_true", help="Gradient Checkpointing aus (schneller, mehr Speicher)")
    p.add_argument("--warmup", type=float, default=0.1, help="Anteil Warmup-Schritte")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-weights", action="store_true", help="Modellgewichte je Fold speichern (~700 MB/Fold)")
    p.add_argument("--smoke", action="store_true", help="Schneller Funktionstest")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in tqdm(loader, desc="  predict", leave=False):
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def train_fold(k, args, df, test, folds, classes, tokenizer, device, run_dir):
    tr_idx, va_idx = np.where(folds != k)[0], np.where(folds == k)[0]
    label_id = {c: i for i, c in enumerate(classes)}
    y = df[TARGET].map(label_id).to_numpy()

    collate = DataCollatorWithPadding(tokenizer)
    ds_tr = TextDataset(df[TEXT_COL].values[tr_idx], y[tr_idx], tokenizer, args.max_len)
    ds_va = TextDataset(df[TEXT_COL].values[va_idx], y[va_idx], tokenizer, args.max_len)
    ds_te = TextDataset(test[TEXT_COL].values, None, tokenizer, args.max_len)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size * 2, collate_fn=collate)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size * 2, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(classes),
        id2label=dict(enumerate(classes)), label2id=label_id,
        # transformers v5 lädt sonst im Checkpoint-dtype (fp16) -> NaN-Loss beim Training
        dtype=torch.float32,
    ).to(device)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()

    no_decay = ("bias", "LayerNorm.weight", "norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=args.lr)
    steps = math.ceil(len(dl_tr) / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"\n===== Fold {k} | train {len(tr_idx)} | valid {len(va_idx)} | {steps} Optimizer-Schritte =====")
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), []
        bar = tqdm(dl_tr, desc=f"  Epoche {epoch}/{args.epochs}")
        for step, batch in enumerate(bar, 1):
            batch = {kk: v.to(device) for kk, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(dl_tr):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                sched.step()
            running.append(loss.item() * args.grad_accum)
            bar.set_postfix(loss=f"{np.mean(running[-50:]):.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        val_proba = predict(model, dl_va, device)
        m = evaluate(df[TARGET].values[va_idx], val_proba, classes)
        recalls = pd.Series(recall_score(y[va_idx], val_proba.argmax(1), labels=range(len(classes)),
                                         average=None, zero_division=0), index=classes)
        print(f"  Epoche {epoch}: train_loss={np.mean(running):.4f} | "
              f"balanced_acc={m['balanced_accuracy']:.4f} | family_acc={m['family_accuracy']:.4f} | "
              f"log_loss={m['log_loss']:.4f} | {time.time() - t0:.0f}s")
        print("    schwächste Recalls:", ", ".join(f"{c}={r:.2f}" for c, r in recalls.nsmallest(4).items()))
        log_rows.append({"fold": k, "epoch": epoch, "train_loss": np.mean(running), **m, "seconds": time.time() - t0})
        pd.DataFrame(log_rows).to_csv(run_dir / f"log_fold{k}.csv", index=False)

    # OOF/Test = letzte Epoche (keine Epochenauswahl auf dem Valid-Fold -> ehrliche OOF)
    np.save(run_dir / f"fold{k}_val.npy", val_proba)
    np.save(run_dir / f"fold{k}_test.npy", predict(model, dl_te, device))
    if args.save_weights:
        model.save_pretrained(run_dir / f"fold{k}_model")
        tokenizer.save_pretrained(run_dir / f"fold{k}_model")

    del model, optim
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return m


def collect(run_dir, run_name, df, test, folds, classes):
    """Schreibt OOF (wenn alle Folds da sind) und mittelt Test-Vorhersagen."""
    done = [k for k in range(N_SPLITS) if (run_dir / f"fold{k}_val.npy").exists()]
    if done:
        test_proba = np.mean([np.load(run_dir / f"fold{k}_test.npy") for k in done], axis=0)
        out = pd.DataFrame(test_proba, columns=classes)
        out.insert(0, "id", test["id"].values)
        out.to_parquet(run_dir / f"test_{run_name}.parquet", index=False)
    if len(done) < N_SPLITS:
        print(f"\nFertige Folds: {done} – OOF wird geschrieben, sobald alle {N_SPLITS} Folds vorliegen.")
        return
    oof = np.zeros((len(df), len(classes)))
    for k in range(N_SPLITS):
        oof[folds == k] = np.load(run_dir / f"fold{k}_val.npy")
    save_oof(run_name, df, oof, classes, run_dir)
    m = evaluate(df[TARGET].values, oof, classes)
    print("\n===== OOF gesamt (alle Folds) =====")
    print({kk: round(v, 4) for kk, v in m.items()})


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    df, test = load_train(), load_test()
    folds = get_folds(df)  # vor dem Smoke-Subset, damit Folds identisch bleiben
    classes = np.sort(df[TARGET].unique())

    if args.smoke:
        keep = df.groupby(TARGET).sample(12, random_state=args.seed).index
        df, folds = df.loc[keep].reset_index(drop=True), folds[keep]
        args.epochs, args.folds, args.max_len = 1, "0", 256
        args.run_name = args.run_name or "smoke"

    run_name = args.run_name or f"{args.model.split('/')[-1]}_len{args.max_len}_lr{args.lr:g}_ep{args.epochs}"
    run_dir = MODEL_DIR / "04_transformer" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    fold_ids = range(N_SPLITS) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    print(f"Device: {device} | Modell: {args.model} | Run: {run_name}")
    print(f"Train: {len(df)} Texte, {len(classes)} Klassen | Folds: {list(fold_ids)} | Ausgabe: {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    t_start = time.time()
    for k in fold_ids:
        train_fold(k, args, df, test, folds, classes, tokenizer, device, run_dir)
        print(f"  Fold {k} fertig nach insgesamt {(time.time() - t_start) / 60:.1f} min")

    if not args.smoke:
        collect(run_dir, run_name, df, test, folds, classes)
    print(f"\nFertig in {(time.time() - t_start) / 60:.1f} min. Ergebnisse: {run_dir}")


if __name__ == "__main__":
    main()
