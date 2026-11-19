"""Erzeugt kaggle/family_spec/family_spec.ipynb – LLM-Familien-Spezialist (Experiment 12), Kaggle 1x T4.

Ziel: die Geschwister innerhalb von Claude (5), GPT (5) und Gemini (4) besser trennen als der Stack. Die
TF-IDF-Fassung (export_spec.py) bringt als Apex-Evidenz in der Simulation +0,7 Pp., trennt aber innerhalb der
Familien nur 0.65 / 0.56 / 0.59. Hier dasselbe Gerüst wie kaggle/build_notebook_lora.py (Qwen3-Base, erste 24
Schichten, LoRA r16 auf allen linearen Schichten, maskiertes Mittel in 8 Tiefen), aber:
- 14 Klassen (die drei Familien), Softmax/CE nur innerhalb der Familie des Texts (andere Familien maskiert);
- nur Pool-Texte, ohne kb-Proxy und ohne Texte zu train.csv-Prompts (train.csv bleibt sauber getrennt);
- zufällige Batches, eine Epoche, Abbruch, wenn die Zeit (BUDGET_H) nicht reicht.
Ausgaben (/kaggle/working): train_spec_proba.parquet (alle 2.699 train.csv-Texte, 14 Klassen), eval.json,
    final/ (lora.pt, heads.pt, meta.json) für das Abgabe-Notebook.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_family_spec.py --per-class 3000
    .venv/bin/kaggle kernels push -p kaggle/family_spec
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle" / "family_spec"
BASES = {"4b": ("qwen-lm/qwen-3/Transformers/4b-base/1", 2560), "8b": ("qwen-lm/qwen-3/Transformers/8b-base/1", 4096)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def build(base, per_class, max_len, batch, budget_h, lr, seed=42, resume=None):
    model_source, hidden = BASES[base]
    cells = [
        md(f"# Name That LLM – Familien-Spezialist (Qwen3-{base.upper()}-Base + LoRA)\n\n"
           "Trennt die Geschwister in Claude, GPT und Gemini; trainiert nur auf dem Open Pool (crtproduction, "
           "CC BY-SA 4.0), Softmax innerhalb der Familie. Bewertung auf train.csv (nicht im Training)."),
        code(f"""BASE_HIDDEN = {hidden}
KEEP_LAYERS = 24
DEPTHS = [2, 4, 6, 8, 12, 16, 20, 24]
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
MAX_LEN = {max_len}
BATCH = {batch}
PER_CLASS = {per_class}
LR, HEAD_LR = {lr}, 1e-3
BUDGET_H = {budget_h}
SEED = {seed}
RESUME = {bool(resume)}
FAMS = ('claude', 'gpt', 'gemini')"""),
        code("""import glob, json, math, os, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

assert torch.cuda.is_available()
dev = torch.device('cuda'); T0 = time.time()
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT = Path('/kaggle/working')
log = lambda *a: print(f'[{(time.time() - T0) / 60:6.1f} min]', *a, flush=True)

def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True)); assert hits, pattern; return Path(hits[0])

def find_base():
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if c.get('model_type') == 'qwen3' and c.get('hidden_size') == BASE_HIDDEN:
            return Path(cfg).parent
    raise FileNotFoundError('Qwen3-Basismodell als Input einbinden')

BASE_DIR = find_base()
real = pd.read_csv(find('name-that-llm/train.csv'))
import pyarrow.parquet as pq
POOL_FILE = find('ntl_open_pool.parquet')
FLAGS = ['flag_kb_proxy_after_ruling', 'flag_near_train_prompt', 'flag_train_prompt_regen']
have = set(pq.read_schema(POOL_FILE).names)
pool = pd.read_parquet(POOL_FILE, columns=['text', 'model'] + [c for c in FLAGS if c in have])
log('Pool-Spalten vorhanden:', [c for c in FLAGS if c in have], '| fehlen:', [c for c in FLAGS if c not in have])
CLASSES = np.array(sorted(c for c in real.model.unique() if c.startswith(FAMS))); K = len(CLASSES)
fam_of = np.array([next(i for i, f in enumerate(FAMS) if c.startswith(f)) for c in CLASSES])
FAM_MASK = torch.tensor(np.stack([fam_of == f for f in range(len(FAMS))]), device=dev)   # (3, K)
keep = pool.model.isin(CLASSES)
for c in FLAGS:
    if c in pool:
        keep &= ~pool[c].fillna(False).astype(bool)
P = pool[keep]
P = pd.concat([g.sample(min(len(g), PER_CLASS), random_state=SEED) for _, g in P.groupby('model')])
P = P.sample(frac=1, random_state=SEED).reset_index(drop=True)
P['y'] = P.model.map({c: i for i, c in enumerate(CLASSES)}); P['f'] = fam_of[P.y.values]
log('Basis', BASE_DIR, '| GPUs', torch.cuda.device_count(), '| Pool', len(P), P.model.value_counts().to_dict())"""),
        code("""tok = AutoTokenizer.from_pretrained(BASE_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'right'
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale, self.drop = alpha / r, nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scale

def load_trunk():
    cfg = AutoConfig.from_pretrained(BASE_DIR)
    cfg.num_hidden_layers, cfg.use_cache = KEEP_LAYERS, False
    if getattr(cfg, 'layer_types', None) is not None:
        cfg.layer_types = cfg.layer_types[:KEEP_LAYERS]
    m = AutoModel.from_pretrained(BASE_DIR, config=cfg, dtype=torch.float16).to(dev)
    m.norm = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    for name, mod in [(n, x) for n, x in m.named_modules() if n.split('.')[-1] in LORA_TARGETS and isinstance(x, nn.Linear)]:
        parent = m.get_submodule(name.rsplit('.', 1)[0])
        setattr(parent, name.rsplit('.', 1)[1], LoRALinear(mod, LORA_R, LORA_ALPHA, LORA_DROPOUT))
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.enable_input_require_grads()
    return m

class Head(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in DEPTHS])
        self.out = nn.Linear(len(DEPTHS) * h, K)
    def forward(self, hidden_states, mask):
        m = mask.unsqueeze(-1).float().clone(); m[:, 0] = 0
        feats = [n((hidden_states[d].float() * m).sum(1) / m.sum(1)) for d, n in zip(DEPTHS, self.norms)]
        return self.out(torch.cat(feats, -1))

lm = load_trunk(); head = Head(lm.config.hidden_size).to(dev)
_prev = sorted(glob.glob('/kaggle/input/**/final/lora.pt', recursive=True))
if RESUME and _prev:   # Fortsetzung: LoRA und Kopf aus dem vorherigen Lauf (Kernel-Quelle)
    _miss = lm.load_state_dict(torch.load(_prev[0], map_location=dev), strict=False)
    assert not [k for k in _miss.missing_keys if 'lora_' in k], 'LoRA-Gewichte fehlen'
    head.load_state_dict(torch.load(Path(_prev[0]).parent / 'heads.pt', map_location=dev))
    log('fortgesetzt von', _prev[0])
else:
    assert not RESUME, 'RESUME gesetzt, aber kein final/lora.pt eingebunden'

def logits_of(texts):
    b = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')
    ids, mask = b['input_ids'].to(dev), b['attention_mask'].to(dev)
    with torch.autocast('cuda', dtype=torch.float16):
        out = lm(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    return head(out.hidden_states, mask)

def fam_logsoftmax(z, f):
    # Softmax nur innerhalb der Familie f (andere Klassen -inf)
    return torch.log_softmax(z.masked_fill(~FAM_MASK[f], float('-inf')), -1)

@torch.no_grad()
def predict(texts, bs=32):
    lm.eval(); head.eval()
    texts = list(texts); order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), K), np.float32)
    for i in range(0, len(texts), bs):
        sel = order[i:i + bs]
        out[sel] = logits_of([texts[j] for j in sel]).float().cpu().numpy()
    lm.train(); head.train()
    return out
log('Modell geladen', f'{sum(p.numel() for p in lm.parameters() if p.requires_grad) / 1e6:.1f} M trainierbar')"""),
        code("""params = [{'params': [p for p in lm.parameters() if p.requires_grad], 'lr': LR}, {'params': list(head.parameters()), 'lr': HEAD_LR}]
opt = torch.optim.AdamW(params, weight_decay=0.01)
steps = math.ceil(len(P) / BATCH)
sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler('cuda')
lm.train(); head.train(); run, bad, t_train = [], 0, time.time()
for s in range(steps):
    b = P.iloc[s * BATCH:(s + 1) * BATCH]
    z = logits_of(b.text.values).float()
    y = torch.tensor(b.y.values, device=dev); f = torch.tensor(b.f.values, device=dev)
    loss = F.nll_loss(fam_logsoftmax(z, f), y)
    if not torch.isfinite(loss):
        bad += 1; opt.zero_grad(set_to_none=True); continue
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for g in params for p in g['params']], 1.0)
    scaler.step(opt); scaler.update(); sch.step(); run.append(loss.item())
    if (s + 1) % 100 == 0 or s == 9:
        rate = (s + 1) * BATCH / (time.time() - t_train)
        log(f'Schritt {s + 1}/{steps} | loss {np.mean(run[-100:]):.4f} | {rate:.1f} Texte/s | nicht-endlich {bad}')
    # Zeitgrenze: Rest der Sitzung für die Vorhersage auf train.csv (~2.699 Texte) und das Speichern freihalten
    if time.time() - T0 > (BUDGET_H - 0.4) * 3600:
        log(f'Zeitgrenze: Abbruch nach {s + 1} von {steps} Schritten'); break
log('Training fertig')"""),
        code("""fin = OUT / 'final'; fin.mkdir(exist_ok=True)
torch.save({k: v.detach().cpu() for k, v in lm.state_dict().items() if 'lora_' in k}, fin / 'lora.pt')
torch.save(head.state_dict(), fin / 'heads.pt')
json.dump({'base_hidden': BASE_HIDDEN, 'keep_layers': KEEP_LAYERS, 'depths': DEPTHS, 'max_len': MAX_LEN, 'lora_r': LORA_R,
           'lora_alpha': LORA_ALPHA, 'classes': CLASSES.tolist(), 'families': list(FAMS), 'n_train': int(min(len(P), (s + 1) * BATCH))},
          open(fin / 'meta.json', 'w'), indent=2)
z = predict(real.text.values)
o = pd.DataFrame(z, columns=CLASSES); o.insert(0, 'id', real.id.values); o.to_parquet(OUT / 'train_spec_logits.parquet', index=False)
ev = {}
for fi, fam in enumerate(FAMS):
    m = real.model.str.startswith(fam).values
    cl = CLASSES[fam_of == fi]
    pred = cl[z[m][:, fam_of == fi].argmax(1)]
    ev[fam] = float((pred == real.model.values[m]).mean())
ev['hours'] = (time.time() - T0) / 3600
json.dump(ev, open(OUT / 'eval.json', 'w'), indent=1)
log('Trefferquote innerhalb der Familie auf train.csv:', ev)"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for c in cells:
        if c["cell_type"] == "code":
            compile(c["source"], "cell", "exec")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "family_spec.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": f"linuszarse/ntl-family-spec-qwen3-{base}" + ("-r2" if resume else ""),
        "title": f"NTL Family Spec Qwen3 {base}" + (" r2" if resume else ""),
        "code_file": "family_spec.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["crtproduction/name-that-llm-open-pool"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [resume] if resume else [], "model_sources": [model_source],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"geschrieben: {OUT_DIR}/family_spec.ipynb | {per_class} je Klasse, MAX_LEN {max_len}, Batch {batch}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=BASES, default="4b")
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--budget-h", type=float, default=11.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None, help="Kernel mit final/lora.pt, z. B. linuszarse/ntl-family-spec-qwen3-4b")
    a = ap.parse_args()
    build(a.base, a.per_class, a.max_len, a.batch, a.budget_h, a.lr, a.seed, a.resume)
