"""Erzeugt kaggle/lora/train_lora.ipynb – LoRA auf einem Qwen3-Base-LLM, zweistufig (Pool -> train.csv).

Rezept (nach dem öffentlichen "Two-stage LoRA"-Ansatz, LB 0.59 als Einzelmodell-Blend):
- Nur die ersten KEEP_LAYERS Schichten des LLM (Stil steckt in frühen/mittleren Schichten).
- LoRA (r=16, alpha=32) auf allen linearen Schichten.
- Merkmale: maskierter Mittelwert der Hidden States in mehreren Tiefen, je eigene LayerNorm, verkettet.
- Zwei lineare Köpfe: "plain" und "centered" (Merkmale minus Mittelwert eines Tripels mit demselben Prompt).
  Der centered-Kopf ist nur Trainingssignal (lernt Stil statt Thema); Vorhersagen nutzen den plain-Kopf.
- Stufe 1: Tripel aus dem Open Pool (gleicher Prompt, verschiedene Modelle), klassengewichtete CE.
- Stufe 2: 2 Epochen auf train.csv; 5-Fold-CV (gleiche Folds wie lokal) + finales Modell auf allen Daten.

Ausgaben (/kaggle/working): stage1_train_proba.parquet (Protokoll A), oof_stage2.parquet, cv.csv,
    final/ (LoRA-Adapter + Köpfe + meta.json) für das Abgabe-Notebook.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_lora.py --base 4b --triplets 8000
    .venv/bin/kaggle kernels push -p kaggle/lora
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "kaggle" / "lora"

BASES = {  # Kaggle-Model-Quelle, Hidden Size (zum Finden des Ordners), Layer insgesamt
    "4b": ("qwen-lm/qwen-3/Transformers/4b-base/1", 2560, 36),
    "8b": ("qwen-lm/qwen-3/Transformers/8b-base/1", 4096, 36),
}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def fold_map() -> dict:
    from src.cv import get_folds
    from src.data import load_train
    df = load_train()
    return dict(zip(df["prompt_id"], get_folds(df).tolist()))


def build(base, keep, triplets, max_len, tps, cv, full_fit):
    model_source, hidden, _ = BASES[base]
    cells = [
        md(f"# Name That LLM – LoRA auf Qwen3-{base.upper()}-Base (zweistufig)\n\n"
           "Stufe 1: Tripel aus dem Open Pool (gleicher Prompt). Stufe 2: train.csv, 5-Fold-CV + finales Modell."),
        code(f"""BASE_HIDDEN = {hidden}          # identifiziert den Basismodell-Ordner unter /kaggle/input
KEEP_LAYERS = {keep}
DEPTHS = [d for d in (2, 4, 6, 8, 12, 16, 20, 24, 28, 32) if d <= KEEP_LAYERS]
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
MAX_LEN = {max_len}
TRIPLETS_PER_STEP = {tps}        # 3 Texte je Tripel
STAGE1_TRIPLETS = {triplets}     # Anzahl Pool-Tripel in Stufe 1 (1 Epoche)
LR1, HEAD_LR1 = 1e-4, 1e-3
LR2, HEAD_LR2, EPOCHS2 = 5e-5, 5e-4, 2
RUN_CV = {cv}
FULL_FIT = {full_fit}
SEED = 42
FOLD_OF_PROMPT = {json.dumps(fold_map())}"""),
        code("""import glob, json, math, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import torch.nn.functional as F

assert torch.cuda.is_available()
dev = torch.device("cuda")
T0 = time.time()
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT = Path("/kaggle/working")
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, pattern
    return Path(hits[0])

def find_base():
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if c.get("model_type") == "qwen3" and c.get("hidden_size") == BASE_HIDDEN:
            return Path(cfg).parent
    raise FileNotFoundError("Qwen3-Basismodell als Input einbinden")

BASE_DIR = find_base()
real = pd.read_csv(find("name-that-llm/train.csv"))
pool = pd.read_parquet(find("ntl_open_pool.parquet"), columns=["text", "model", "prompt_canon"])
classes = np.sort(real["model"].unique()); K = len(classes)
label_id = {c: i for i, c in enumerate(classes)}
real["y"] = real["model"].map(label_id); pool["y"] = pool["model"].map(label_id)
real["fold"] = real["prompt_id"].map(FOLD_OF_PROMPT)
FAMILIES = ("claude", "gpt", "gemini", "glm", "kimi", "qwen")
family = lambda m: next(f for f in FAMILIES if m.startswith(f))
log("Basis:", BASE_DIR, "| GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))"""),
        md("## Tripel bauen"),
        code("""def pool_triplets(n, seed):
    \"\"\"n Tripel (Indizes) aus Pool-Prompts mit >=3 verschiedenen Modellen; Klassen möglichst ausgeglichen.\"\"\"
    rng = np.random.default_rng(seed)
    groups = [g for _, g in pool.groupby("prompt_canon") if g["model"].nunique() >= 3]
    rng.shuffle(groups)
    cls_count = np.zeros(K)
    out = []
    while len(out) < n:
        for g in groups:
            g = g.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates("model")
            # seltene Klassen bevorzugen
            g = g.iloc[np.argsort(cls_count[g["y"].values], kind="stable")[:3]]
            out.append(g.index.values); cls_count[g["y"].values] += 1
            if len(out) >= n:
                break
    return out

def real_triplets(df, seed):
    \"\"\"Alle Texte einer train.csv-Teilmenge, je Prompt gemischt in Dreiergruppen.\"\"\"
    rng = np.random.default_rng(seed)
    out = []
    for _, g in df.groupby("prompt_id"):
        idx = g.index.values.copy(); rng.shuffle(idx)
        out += [idx[i:i + 3] for i in range(0, len(idx) - 2, 3)]
    rng.shuffle(out)
    return out

t = time.time()
P_TRIPS = pool_triplets(STAGE1_TRIPLETS, SEED)
p_idx = np.concatenate(P_TRIPS)
cw = np.bincount(pool.loc[p_idx, "y"], minlength=K).astype(float)
CLASS_W = torch.tensor(cw.sum() / (K * np.maximum(cw, 1)), dtype=torch.float32, device=dev)
log(f"{len(P_TRIPS)} Pool-Tripel ({len(p_idx)} Texte), Klassen min/max {cw.min():.0f}/{cw.max():.0f}")"""),
        md("## Modell: LLM-Rumpf + LoRA + Mehrtiefen-Pooling + zwei Köpfe"),
        code("""tok = AutoTokenizer.from_pretrained(BASE_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

class LoRALinear(nn.Module):
    \"\"\"Eigene LoRA-Schicht (statt peft): y = W x + (B A dropout(x)) * alpha/r; A, B in fp32.\"\"\"
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale, self.drop = alpha / r, nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scale

def add_lora(model):
    for p in model.parameters():
        p.requires_grad = False
    targets = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] in LORA_TARGETS and isinstance(m, nn.Linear)]
    for name, mod in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], LoRALinear(mod, LORA_R, LORA_ALPHA, LORA_DROPOUT))
    return len(targets)

def load_trunk():
    cfg = AutoConfig.from_pretrained(BASE_DIR)
    cfg.num_hidden_layers, cfg.use_cache = KEEP_LAYERS, False
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[:KEEP_LAYERS]
    m = AutoModel.from_pretrained(BASE_DIR, config=cfg, dtype=torch.float16).to(dev)
    m.norm = nn.Identity()
    n = add_lora(m)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    train_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"LoRA auf {n} Schichten | trainierbar: {train_p / 1e6:.1f} M von {sum(p.numel() for p in m.parameters()) / 1e9:.2f} Mrd.")
    return m

class Heads(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in DEPTHS])
        self.plain = nn.Linear(len(DEPTHS) * h, K)
        self.centered = nn.Linear(len(DEPTHS) * h, K)
    def features(self, hidden_states, mask):
        m = mask.unsqueeze(-1).float().clone()
        m[:, 0] = 0                      # erstes Token auslassen (wie die öffentliche l24-Inferenz)
        feats = [norm((hidden_states[d].float() * m).sum(1) / m.sum(1)) for d, norm in zip(DEPTHS, self.norms)]
        return torch.cat(feats, -1)

lm = load_trunk()
heads = Heads(lm.config.hidden_size).to(dev)

def encode(texts):
    b = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return b["input_ids"].to(dev), b["attention_mask"].to(dev)

def forward_feats(texts):
    ids, mask = encode(texts)
    with torch.autocast("cuda", dtype=torch.float16):
        out = lm(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    return heads.features(out.hidden_states, mask)

@torch.no_grad()
def predict(texts, bs=32):
    lm.eval(); heads.eval()
    texts = list(texts); order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), K), np.float32)
    for i in range(0, len(texts), bs):
        sel = order[i:i + bs]
        out[sel] = torch.softmax(heads.plain(forward_feats([texts[j] for j in sel])), -1).cpu().numpy()
    lm.train(); heads.train()
    return out

def score(y, proba):
    pred = classes[proba.argmax(1)]
    return {"balanced_accuracy": balanced_accuracy_score(classes[y], pred),
            "family_accuracy": float(np.mean([family(a) == family(b) for a, b in zip(classes[y], pred)]))}"""),
        code("""def train_on(trips, df, epochs, lr, head_lr, class_w=None, tag=""):
    params = [{"params": [p for p in lm.parameters() if p.requires_grad], "lr": lr},
              {"params": list(heads.parameters()), "lr": head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = math.ceil(len(trips) / TRIPLETS_PER_STEP) * epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss(weight=class_w)
    lm.train(); heads.train()
    step, bad, run = 0, 0, []
    for ep in range(epochs):
        order = trips if ep == 0 else [trips[i] for i in np.random.permutation(len(trips))]
        for i in range(0, len(order), TRIPLETS_PER_STEP):
            batch = order[i:i + TRIPLETS_PER_STEP]
            idx = np.concatenate(batch)
            y = torch.tensor(df.loc[idx, "y"].values, device=dev)
            f = forward_feats(df.loc[idx, "text"].values)
            sizes = [len(b) for b in batch]
            fc = torch.cat([g - g.mean(0, keepdim=True) for g in torch.split(f, sizes)])
            loss = ce(heads.plain(f), y) + ce(heads.centered(fc), y)
            if not torch.isfinite(loss):
                bad += 1; opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            run.append(loss.item()); step += 1
            if step % 100 == 0 or step == 10:
                log(f"{tag} Schritt {step}/{steps} | loss {np.mean(run[-100:]):.4f} | nicht-endlich: {bad}")
    return bad

def snapshot():
    return ({k: v.detach().cpu().clone() for k, v in lm.state_dict().items() if "lora_" in k},
            {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()})

def restore(snap):
    lm.load_state_dict(snap[0], strict=False); heads.load_state_dict(snap[1])"""),
        md("## Stufe 1: Pool"),
        code("""init = snapshot()
train_on(P_TRIPS, pool, 1, LR1, HEAD_LR1, CLASS_W, tag="Stufe 1")
s1 = snapshot()
p1 = predict(real["text"].values)
m1 = score(real["y"].values, p1)
log("Stufe 1 auf train.csv (Protokoll A):", m1)
o = pd.DataFrame(p1, columns=classes); o.insert(0, "id", real["id"].values)
o.to_parquet(OUT / "stage1_train_proba.parquet", index=False)"""),
        md("## Stufe 2: train.csv (5-Fold-CV)"),
        code("""rows = [{"stage": "1 (nur Pool)", **m1}]
if RUN_CV:
    oof = np.zeros((len(real), K), np.float32)
    for k in range(5):
        restore(s1)
        tr = real[real["fold"] != k]
        va = np.where(real["fold"].values == k)[0]          # Positionen
        train_on(real_triplets(tr, SEED + k), real, EPOCHS2, LR2, HEAD_LR2, tag=f"Fold {k}")
        oof[va] = predict(real["text"].values[va])
        log(f"Fold {k}:", score(real["y"].values[va], oof[va]))
    m2 = score(real["y"].values, oof)
    rows.append({"stage": "1+2 (OOF)", **m2})
    log("Stufe 2 OOF:", m2)
    o = pd.DataFrame(oof, columns=classes); o.insert(0, "id", real["id"].values)
    o.to_parquet(OUT / "oof_stage2.parquet", index=False)
pd.DataFrame(rows).to_csv(OUT / "cv.csv", index=False)
pd.DataFrame(rows)"""),
        md("## Finales Modell (Stufe 1 + Stufe 2 auf allen train.csv-Texten)"),
        code("""if FULL_FIT:
    restore(s1)
    train_on(real_triplets(real, SEED), real, EPOCHS2, LR2, HEAD_LR2, tag="Final")
    fin = OUT / "final"; fin.mkdir(exist_ok=True)
    lora_sd = {k: v.detach().cpu() for k, v in lm.state_dict().items() if "lora_" in k}
    torch.save(lora_sd, fin / "lora.pt")
    # zusätzlich im peft-Format der öffentlichen l24-Pipeline (adapter/ + heads.pt + meta.json mit labels)
    from safetensors.torch import save_file
    (fin / "adapter").mkdir(exist_ok=True)
    save_file({f"base_model.model.{k.rsplit('.', 1)[0]}.{k.rsplit('.', 1)[1]}.weight": v.contiguous() for k, v in lora_sd.items()},
              str(fin / "adapter" / "adapter_model.safetensors"))
    json.dump({"r": LORA_R, "lora_alpha": LORA_ALPHA, "peft_type": "LORA"}, open(fin / "adapter" / "adapter_config.json", "w"))
    torch.save(heads.state_dict(), fin / "heads.pt")
    json.dump({"base_hidden": BASE_HIDDEN, "keep_layers": KEEP_LAYERS, "depths": DEPTHS, "max_len": MAX_LEN,
               "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "classes": classes.tolist(), "labels": classes.tolist(),
               "cfg": {"keep": KEEP_LAYERS, "layers": DEPTHS, "max_len": MAX_LEN, "eos_id": None}}, open(fin / "meta.json", "w"), indent=2)
    log("gespeichert:", fin)"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train_lora.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": f"linuszarse/name-that-llm-lora-qwen3-{base}",
        "title": f"Name That LLM LoRA Qwen3 {base}",
        "code_file": "train_lora.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["crtproduction/name-that-llm-open-pool"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [], "model_sources": [model_source],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"geschrieben: {OUT_DIR}/train_lora.ipynb (+ kernel-metadata.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=BASES, default="4b")
    ap.add_argument("--keep-layers", type=int, default=24)
    ap.add_argument("--triplets", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--triplets-per-step", type=int, default=6)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--no-full-fit", action="store_true")
    a = ap.parse_args()
    build(a.base, a.keep_layers, a.triplets, a.max_len, a.triplets_per_step, not a.no_cv, not a.no_full_fit)
