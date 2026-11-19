"""Erzeugt kaggle/kimi_probe/kimi_probe.ipynb – Kimi-K3-Likelihood auf Kaggle (2x T4): bauen, messen, train bewerten.

Experiment 11, Schritt 1 + 2 in einer Sitzung:
1. ntl-segrun aus llama.cpp b12247 + `ntl-kimi-k3-segrun.patch` (xerneas, liegt in den Segment-Datensätzen) bauen,
   CUDA für die T4 (sm_75), statisch gegen llama/ggml gelinkt; Binary + CUDA-Bibliotheken nach
   /kaggle/working/ntl-bin-kimi (später als Kernel-Quelle für die Abgabe).
2. Metadaten und Chat-Template aus Segment 0 lesen, Präfix bestimmen (Template mit leerer Nutzernachricht).
3. Probe: 50 train-Texte (16 Kimi-K3, je 2 der übrigen Klassen) unter allen 23 Segmenten, Ladezeit und Rechenzeit
   je Segment getrennt gemessen.
4. Gate: Nur wenn ein Durchlauf (Laden aller Segmente) höchstens GATE_LOAD_S dauert und die Rechenzeit je Token
   höchstens GATE_ROW_S, passt eine Kimi-Stufe überhaupt ins 9-h-Limit. Dann werden die übrigen train-Texte in
   Häppchen bewertet, so viele wie bis BUDGET_H passen (jedes fertige Häppchen liegt sofort in /kaggle/working/out).

Code der Stufe: segio.py, features.py (unverändert) und run_pipeline_kimi.py (run_pipeline.py mit drei Änderungen
für Kimi, siehe dort), alle von José Padilla (estopadilla, Apache 2.0). Segmente: xerneas (Kimi K3 License).

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_kimi_probe.py
    .venv/bin/kaggle kernels push -p kaggle/kimi_probe
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle" / "kimi_probe"
SRC = ROOT / "experiments" / "11_kimi_likelihood" / "src"
SEED = 11
PATCH_SHA = "8506eb283fd6f8c6e70d2384cf3b8ef72655d23fe49b3552ce03468d7b3427b1"
N_PROBE_KIMI, N_PROBE_OTHER = 16, 2


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def select():
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    tr = tr.sample(frac=1, random_state=SEED).reset_index(drop=True)
    probe = pd.concat([g.head(N_PROBE_KIMI if m == "kimi-k3" else N_PROBE_OTHER) for m, g in tr.groupby("model")])
    rest = tr[~tr.id.isin(probe.id)].copy()
    # reihum über die Klassen: jedes Präfix der Liste ist annähernd klassenbalanciert
    rest["k"] = rest.groupby("model").cumcount()
    rest = rest.sort_values(["k", "model"], kind="stable")
    return probe.id.astype(str).tolist(), rest.id.astype(str).tolist()


CONFIG = """MAX_TEXT_TOKENS = 160          # wie die GLM-Stufe; die Token-Scores werden gespeichert, kürzere Schnitte gehen später
SEQ_BATCH = 8
BUDGET_H = 11.0                # Kaggle-GPU-Sitzung max. 12 h
GATE_LOAD_S = 3600.0           # ein Durchlauf (alle 23 Segmente laden) darf höchstens so lange dauern
GATE_ROW_S = 0.10              # Rechenzeit je Token-Zeile (GLM-5.3: 0.023, Flash: 0.014)
CHUNK_MAX = 900                # Texte je Häppchen (jedes Häppchen liest das Modell einmal)
LLAMA_TAG = 'b12247'
PATCH_SHA = %r
PROBE_IDS = %s
REST_IDS = %s
import time; T0 = time.time()"""

SETUP = r"""import glob, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out[-6000:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f'exit {r.returncode}: {cmd}')
    return r

def find_segs(pattern):
    seen = {}
    for p in sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True)):
        seen.setdefault(os.path.basename(p), p)
    return [seen[k] for k in sorted(seen)]

sh('nvidia-smi -L; nproc; free -g | head -2; df -h /kaggle/working /kaggle/temp /tmp 2>/dev/null; which nvcc && nvcc --version | tail -2; cmake --version | head -1', check=False)
KIMI = find_segs('kimik3-q1-seg*-of-23-*.gguf')
assert len(KIMI) == 23, f'{len(KIMI)} Kimi-Segmente gefunden, 23 erwartet'
print('Kimi-Segmente:', round(sum(os.path.getsize(p) for p in KIMI) / 2**30), 'GiB')
patches = sorted(glob.glob('/kaggle/input/**/ntl-kimi-k3-segrun.patch', recursive=True))
PATCH = next(p for p in patches if hashlib.sha256(open(p, 'rb').read()).hexdigest() == PATCH_SHA)
TRAIN = sorted(glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True))[0]
train = pd.read_csv(TRAIN).astype({'id': str}).set_index('id')
assert set(PROBE_IDS + REST_IDS) == set(train.index) and len(PROBE_IDS) + len(REST_IDS) == len(train)
OUT = Path('/kaggle/working/out'); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path('/kaggle/temp/work'); WORK.mkdir(parents=True, exist_ok=True)
train[['model']].reset_index().to_csv(OUT / 'train_labels.csv', index=False)
print('Patch', PATCH, '| train', TRAIN, len(train))"""

BUILD = r"""# ---- ntl-segrun bauen: llama.cpp b12247 + Kimi-Patch, CUDA sm_75, llama/ggml statisch ----
t = time.time()
SRCDIR = Path('/kaggle/temp/llama.cpp')
if not SRCDIR.exists():
    sh(f'git clone -q --depth 1 --branch {LLAMA_TAG} https://github.com/ggml-org/llama.cpp {SRCDIR}')
    sh(f'cd {SRCDIR} && git apply {PATCH} && git status --short | head -20')
if shutil.which('cmake') is None:
    sh('pip install -q cmake ninja')
# Flags wie José (estopadilla/ntl-glm-likelihood-train): GGML_CUDA_NO_VMM=ON, weil Kaggle kein unversioniertes
# libcuda.so hat (sonst fehlt CUDA::cuda_driver, der Generate-Schritt scheitert und erst der Linker bricht ab,
# so geschehen in Version 1 nach 27 min). Alle Treiber-API-Aufrufe in ggml-cuda hängen an GGML_USE_VMM.
sh(f'cd {SRCDIR} && cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_NO_VMM=ON '
   f'-DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF '
   f'-DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release > /tmp/cmake.log 2>&1; '
   f'grep -n -A12 "CMake Error" /tmp/cmake.log | head -60; tail -3 /tmp/cmake.log', check=False)
_log = open('/tmp/cmake.log').read()
assert 'CMake Error' not in _log and 'Generate step failed' not in _log, 'cmake fehlgeschlagen (siehe oben)'
sh(f'cd {SRCDIR} && cmake --build build -j$(nproc) --target ntl-segrun > /tmp/build.log 2>&1; '
   f'grep -n -i "error\\|undefined reference" /tmp/build.log | head -30; tail -3 /tmp/build.log', check=False)
built = SRCDIR / 'build' / 'bin' / 'ntl-segrun'
assert built.is_file(), 'Build fehlgeschlagen (siehe /tmp/build.log oben)'
BIN = Path('/kaggle/working/ntl-bin-kimi'); (BIN / 'lib').mkdir(parents=True, exist_ok=True)
shutil.copy(built, BIN / 'ntl-segrun'); (BIN / 'ntl-segrun').chmod(0o755)
# CUDA-Laufzeit neben das Binary (wie José): das Abgabe-Image hat ein älteres Toolkit; libcublasLt lädt libcublas
import re
def _deps(p):
    return dict(re.findall(r'(libcu\w+\.so\.\d+) => (\S+)', subprocess.run(['ldd', str(p)], capture_output=True, text=True).stdout))
_libs = _deps(BIN / 'ntl-segrun')
print('CUDA-Abhängigkeiten:', _libs)
for k, v in list(_libs.items()):
    if k.startswith('libcublas.'):
        _libs.update({n: q for n, q in _deps(v).items() if n.startswith('libcublasLt')})
for name, p in sorted(_libs.items()):
    if name.startswith(('libcudart', 'libcublas')):
        shutil.copy(os.path.realpath(p), BIN / 'lib' / name)
(BIN / 'BUILD.json').write_text(json.dumps({'llama_tag': LLAMA_TAG, 'patch_sha256': PATCH_SHA,
                                            'binary_sha256': hashlib.sha256(open(BIN / 'ntl-segrun', 'rb').read()).hexdigest()}, indent=1))
ENV = dict(os.environ, LD_LIBRARY_PATH=str(BIN / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
sh(f'ls -la {BIN} {BIN}/lib; cat {BIN}/BUILD.json')
print(f'Build {time.time() - t:.0f} s', flush=True)"""

META = r"""# ---- Metadaten und Chat-Template aus Segment 0; Präfix = Template mit leerer Nutzernachricht ----
sys.path.insert(0, str(SRCDIR / 'gguf-py'))
from gguf import GGUFReader
rd = GGUFReader(KIMI[0])
def val(f):
    try:
        return f.contents()
    except Exception:
        return None
meta = {}
for k, f in rd.fields.items():
    if k.startswith('tokenizer.ggml.') and k not in ('tokenizer.ggml.model', 'tokenizer.ggml.pre', 'tokenizer.ggml.bos_token_id',
                                                     'tokenizer.ggml.eos_token_id', 'tokenizer.ggml.add_bos_token'):
        continue
    v = val(f)
    meta[k] = v if not isinstance(v, (list, tuple)) or len(v) <= 8 else f'[{len(v)} values]'
TEMPLATE = meta.get('tokenizer.chat_template') or ''
json.dump({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in meta.items()},
          open(OUT / 'kimi_meta.json', 'w'), indent=1)
for k, v in meta.items():
    if k != 'tokenizer.chat_template':
        print(k, '=', v)
print('---- chat template ----'); print(TEMPLATE[:4000])
PREFIX = None
try:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    def _raise(msg):
        raise ValueError(msg)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals['raise_exception'] = _raise
    PREFIX = env.from_string(TEMPLATE).render(messages=[{'role': 'user', 'content': ''}], add_generation_prompt=True,
                                              enable_thinking=False, thinking=False, tools=None,
                                              bos_token='', eos_token='')
except Exception as e:
    print('Template nicht renderbar:', repr(e))
if not PREFIX or 'assistant' not in PREFIX:
    PREFIX = ('<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>'
              '<|im_user|>user<|im_middle|><|im_end|><|im_assistant|>assistant<|im_middle|>')
if PREFIX.rstrip().endswith('<think>'):
    PREFIX = PREFIX.rstrip() + '</think>'
print('PREFIX =', repr(PREFIX))
(OUT / 'kimi_prefix.txt').write_text(PREFIX)"""

RUN = r"""sys.path.insert(0, '/kaggle/temp/v1')
def run_set(tag, ids):
    csv = WORK / f'{tag}.csv'
    train.loc[ids, ['text']].reset_index().to_csv(csv, index=False)
    cmd = [sys.executable, '/kaggle/temp/v1/run_pipeline_kimi.py', '--bin', str(BIN), '--segs', json.dumps(KIMI),
           '--csv', str(csv), '--out', str(WORK / tag), '--tag', 'kimi', '--work', str(WORK / 'state'),
           '--seq-batch', str(SEQ_BATCH), '--state-dtype', 'auto', '--max-text-tokens', str(MAX_TEXT_TOKENS),
           '--prefix', PREFIX, '--prefetch']
    t = time.time()
    with open(OUT / f'{tag}_pipeline.txt', 'w') as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    print(open(OUT / f'{tag}_pipeline.txt').read()[-3000:], flush=True)
    assert code == 0, f'{tag}: Exit {code}, siehe {tag}_pipeline.txt'
    for f in ('kimi_features.csv', 'kimi_timing.json', 'kimi_scores.npz'):
        shutil.copy(WORK / tag / f, OUT / f'{tag}_{f}')
    shutil.rmtree(WORK / tag, ignore_errors=True)
    return time.time() - t, json.load(open(OUT / f'{tag}_kimi_timing.json'))

def auc_report(tag_list):
    from sklearn.metrics import roc_auc_score
    f = pd.concat([pd.read_csv(OUT / f'{t}_kimi_features.csv', dtype={'id': str}) for t in tag_list]).set_index('id')
    y = (train.loc[f.index, 'model'] == 'kimi-k3').to_numpy()
    rows = [(c, roc_auc_score(y, f[c])) for c in f.columns if f[c].std() > 0]
    rows.sort(key=lambda r: -abs(r[1] - 0.5))
    print(f'{len(f)} Texte ({y.sum()} Kimi): AUC Kimi gegen den Rest je Merkmal')
    for c, a in rows[:10]:
        print(f'  {c:16s} {a:.3f}')
    g = f.join(train['model']).groupby('model')[['lp_mean', 'z_mean', 'top1', 'bpc']].mean().round(3)
    print(g.sort_values('lp_mean', ascending=False).to_string())

# ---- Probe ----
dt, tm = run_set('probe', PROBE_IDS)
ch = tm['chunks']
load = sum(sum(x or 0 for x in c['seg_load_seconds']) for c in ch)
total = sum(sum(c['seg_seconds']) for c in ch)
rows = sum(c['n_rows'] for c in ch)
row_s = max(total - load, 0) / rows
PROBE = {'n_texts': tm['n_texts'], 'rows': rows, 'passes': len(ch), 'load_s': load, 'total_s': total, 'row_s': row_s,
         'load_per_pass_s': load / len(ch), 'wall_s': dt}
json.dump(PROBE, open(OUT / 'probe_summary.json', 'w'), indent=1)
print(json.dumps(PROBE, indent=1))
for n_texts in (200, 400, 800):   # Kimi-Stufe im Bewertungslauf: ein Durchlauf + Rechenzeit (160 Token/Text)
    print(f'  Schätzung Testlauf {n_texts} Texte: {(PROBE["load_per_pass_s"] + row_s * n_texts * 165) / 60:.0f} min')
auc_report(['probe'])
GO_ON = PROBE['load_per_pass_s'] <= GATE_LOAD_S and row_s <= GATE_ROW_S
print('GATE:', 'weiter mit allen train-Texten' if GO_ON else 'STOPP: Kimi passt nicht ins Zeitlimit', flush=True)"""

REST = r"""# ---- übrige train-Texte, so viele wie bis BUDGET_H passen ----
done_tags = ['probe']
if GO_ON:
    rem = list(REST_IDS)
    tpc = PROBE['rows'] / train.loc[PROBE_IDS, 'text'].str.len().sum()
    ci = 0
    while rem:
        left = BUDGET_H * 3600 - (time.time() - T0)
        # Zeilen grob aus den Zeichen, Kosten = ein Durchlauf + Rechenzeit, 20 % Puffer
        rows_i = np.minimum(train.loc[rem, 'text'].str.len().to_numpy() * tpc, MAX_TEXT_TOKENS) + 20
        cost = 1.2 * (PROBE['load_per_pass_s'] + PROBE['row_s'] * np.cumsum(rows_i))
        n = int(min(CHUNK_MAX, (cost <= left).sum()))
        if n < 50:
            print(f'Stopp: {left/3600:.2f} h übrig, reicht nicht für 50 weitere Texte', flush=True)
            break
        tag = f'c{ci}'
        dt, tm = run_set(tag, rem[:n])
        rem = rem[n:]
        done_tags.append(tag)
        json.dump({'done_tags': done_tags, 'remaining': len(rem), 'hours': (time.time() - T0) / 3600},
                  open(OUT / 'progress.json', 'w'), indent=1)
        print(f'{tag}: {n} Texte in {dt/60:.0f} min, noch {len(rem)}, gesamt {(time.time()-T0)/3600:.2f} h', flush=True)
        ci += 1
feats = pd.concat([pd.read_csv(OUT / f'{t}_kimi_features.csv', dtype={'id': str}) for t in done_tags])
feats = feats.set_index('id').join(train['model'])
feats.reset_index().to_parquet(OUT / 'kimi_train_feats_L160.parquet', index=False)
print(feats.shape, f'gesamt {(time.time()-T0)/3600:.2f} h'); print(feats['model'].value_counts().to_string())
auc_report(done_tags)
shutil.rmtree(WORK, ignore_errors=True)"""


def build():
    probe, rest = select()
    srcs = {n: (SRC / n).read_text() for n in ("segio.py", "features.py", "run_pipeline_kimi.py")}
    cells = [
        md("# Name That LLM – Kimi-K3-Likelihood: bauen, messen, train bewerten\n\n"
           "Baut `ntl-segrun` mit Kimi-K3-Unterstützung (llama.cpp b12247 + `ntl-kimi-k3-segrun.patch` von xerneas), "
           "misst auf 50 train-Texten Lade- und Rechenzeit aller 23 Segmente (Kimi-K3 UD-Q1_0) und bewertet danach, "
           "falls das in ein 9-h-Abgabelimit passen kann, die übrigen train-Texte (erste 160 Tokens).\n\n"
           "Code der Stufe (`segio.py`, `features.py`, `run_pipeline.py` als Vorlage): José Padilla (estopadilla), "
           "Apache 2.0. Segmente und Patch: xerneas (Kimi K3 License, Moonshot AI). llama.cpp / ggml: MIT."),
        code(CONFIG % (PATCH_SHA, json.dumps(probe), json.dumps(rest))),
        code(SETUP),
        code(BUILD),
        code(META),
        code("Path('/kaggle/temp/v1').mkdir(parents=True, exist_ok=True)"),
        *[code(f"%%writefile /kaggle/temp/v1/{n}\n{s}") for n, s in srcs.items()],
        code(RUN),
        code(REST),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for c in cells:
        if c["cell_type"] == "code" and not c["source"].startswith("%%"):
            compile(c["source"], "cell", "exec")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "kimi_probe.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": "linuszarse/ntl-kimi-likelihood-train",
        "title": "NTL Kimi Likelihood Train",
        "code_file": "kimi_probe.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["xerneas/kimik3-q1-ntl-segs-a", "xerneas/kimik3-q1-ntl-segs-b",
                            "xerneas/kimik3-q1-ntl-segs-c"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"geschrieben: {OUT_DIR}/kimi_probe.ipynb | Probe {len(probe)} Texte, Rest {len(rest)}")


if __name__ == "__main__":
    build()
