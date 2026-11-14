"""Erzeugt kaggle/glm_pool/glm_pool.ipynb – GLM-Likelihood-Merkmale für Open-Pool-Texte (Kaggle, 2x T4).

Zweck: mehr Trainingsdaten für den Likelihood-Klassifikator der GLM-Stufe im öffentlichen 0.694-Notebook
(dort nur auf den 2.699 train.csv-Texten trainiert). Jeder ausgewählte Pool-Text wird unter GLM-5.3-Flash und
GLM-5.3 bewertet (erste 160 Tokens, gleiche Einstellungen wie die Abgabe-Stufe); Ausgabe sind die Merkmale je Text.
Das Training des Klassifikators passiert danach lokal (experiments/08_glm_likelihood/evaluate.py).

- Auswahl (lokal, deterministisch, ID-Liste im Notebook): ohne kb-Proxy-Texte, stilnähere Hälfte je Klasse
  (style_score), ein Text je prompt_group; Schwerpunkt GLM + Verwechslungsklassen. Siehe COUNTS.
- Arbeit in Häppchen (stratifiziert); vor jedem Häppchen wird geprüft, ob die Zeit bis BUDGET_H reicht.
  Jedes fertige Häppchen liegt sofort in /kaggle/working/out, auch wenn spätere nicht mehr passen.
- Code der Stufe (segio.py, features.py, run_pipeline.py) unverändert aus dem öffentlichen Notebook
  (Apache 2.0, José Padilla / estopadilla), Binaries aus dem Output von estopadilla/ntl-glm-likelihood-train.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_glm_pool.py
    .venv/bin/kaggle kernels push -p kaggle/glm_pool
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle" / "glm_pool"
V1SRC = ROOT / "experiments" / "08_glm_likelihood" / "v1src"
POOL = ROOT / "data" / "external" / "open_pool" / "ntl_open_pool.parquet"
SEED = 42
N_CHUNKS = 7
BUDGET_H = 11.0          # Kaggle-GPU-Sitzung max. 12 h; Rest = Puffer für Ausgabe
COUNTS = {"glm-5.3": 1000, "glm-5.3-flash": 1000, "claude-fable-5": 500, "kimi-k3": 400,
          "claude-opus-4-6": 300, "claude-opus-4-8": 300, "claude-opus-5": 300, "claude-sonnet-5": 300}
DEFAULT_COUNT = 250      # alle übrigen Klassen (Gemini, GPT, Qwen)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def select():
    df = pd.read_parquet(POOL, columns=["id", "model", "prompt_group", "style_score", "flag_kb_proxy_after_ruling"])
    df = df[~df.flag_kb_proxy_after_ruling.astype(bool)]
    rng = np.random.default_rng(SEED)
    picks = []
    for m, g in df.groupby("model"):
        g = g[g.style_score >= g.style_score.median()]                     # stilnähere Hälfte
        g = g.sample(frac=1, random_state=SEED).drop_duplicates("prompt_group")  # ein Text je Prompt
        n = min(COUNTS.get(m, DEFAULT_COUNT), len(g))
        picks.append(g.head(n)[["id", "model"]])
    sel = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
    # stratifizierte Häppchen: innerhalb jeder Klasse reihum verteilen
    sel["chunk"] = sel.groupby("model").cumcount() % N_CHUNKS
    sel = sel.sort_values(["chunk"], kind="stable").reset_index(drop=True)
    return sel


def build():
    sel = select()
    print(sel.groupby("model").size().to_string(), "\ngesamt", len(sel), "| Häppchen", sel.chunk.value_counts().sort_index().tolist())
    chunks = [sel[sel.chunk == k]["id"].astype(str).tolist() for k in range(N_CHUNKS)]
    srcs = {n: (V1SRC / n).read_text() for n in ("segio.py", "features.py", "run_pipeline.py")}
    cells = [
        md("# Name That LLM – GLM-Likelihood-Merkmale für Open-Pool-Texte\n\n"
           "Bewertet ausgewählte Texte aus dem Open Pool (crtproduction, CC BY-SA 4.0) unter **GLM-5.3-Flash** und "
           "**GLM-5.3** (erste 160 Tokens, Einstellungen wie die GLM-Stufe der öffentlichen 0.694-Abgabe). "
           "Ausgabe: `out/pool_c*_{flash,glm53}_features.csv` als Trainingsdaten für den Likelihood-Klassifikator.\n\n"
           "Code der Stufe (`segio.py`, `features.py`, `run_pipeline.py`) und Binaries (`ntl-segrun`): José Padilla "
           "(estopadilla), Apache 2.0. GLM-Segmente: jamesliebel."),
        code(f"""MAX_TEXT_TOKENS = 160
BUDGET_S = {BUDGET_H} * 3600   # keine neuen Häppchen mehr, wenn die geschätzte Dauer darüber hinausgeht
SEQ_BATCH = 8
PREFIX = '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|assistant|><think></think>'
CHUNKS = {json.dumps(chunks)}
import time; T0 = time.time()"""),
        code("""import glob, json, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

def find_one(pattern):
    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True))
    assert hits, f'Input fehlt: {pattern}'
    return hits[0]

def find_segs(pattern):
    seen = {}
    for p in sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True)):
        seen.setdefault(os.path.basename(p), p)          # dieselbe Datei kann aus zwei Mirrors kommen
    return [seen[k] for k in sorted(seen)]

FLASH = find_segs('flash_iq3xxs-seg*-of-06-*.gguf')
GLM53 = find_segs('glm53_iq3xxs-seg*-of-19-*.gguf')
assert len(FLASH) == 6 and len(GLM53) == 19, (len(FLASH), len(GLM53))
POOL = find_one('ntl_open_pool.parquet')
segrun = find_one('ntl-bin/ntl-segrun')
ART = Path(segrun).parent
for f in ('lib/libcublas.so.13', 'lib/libcublasLt.so.13', 'lib/libcudart.so.13'):
    assert (ART / f).is_file(), ART / f
Path('/kaggle/temp/v1').mkdir(parents=True, exist_ok=True)   # Ziel der %%writefile-Zellen
BIN = Path('/kaggle/temp/ntl-bin'); BIN.mkdir(parents=True, exist_ok=True)
shutil.copyfile(segrun, BIN / 'ntl-segrun'); (BIN / 'ntl-segrun').chmod(0o755)
ENV = dict(os.environ, LD_LIBRARY_PATH=str(ART / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
subprocess.run('nvidia-smi -L; df -h /kaggle/temp /kaggle/working | tail -2', shell=True)
print('Segmente ok, POOL', POOL, '| BIN', ART)"""),
        *[code(f"%%writefile /kaggle/temp/v1/{n}\n{s}") for n, s in srcs.items()],
        code("""pool = pd.read_parquet(POOL, columns=['id', 'text', 'model']).astype({'id': str}).set_index('id')
wanted = [i for c in CHUNKS for i in c]
missing = [i for i in wanted if i not in pool.index]
assert not missing, f'{len(missing)} IDs fehlen im Pool (andere Dataset-Version?)'
OUT = Path('/kaggle/working/out'); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path('/kaggle/temp/work'); WORK.mkdir(parents=True, exist_ok=True)
pool.loc[wanted, ['model']].reset_index().to_csv(OUT / 'pool_labels.csv', index=False)
print(len(wanted), 'Texte in', len(CHUNKS), 'Häppchen')"""),
        code("""def run_tag(ci, tag, segs, csv):
    cmd = [sys.executable, '/kaggle/temp/v1/run_pipeline.py', '--bin', str(BIN), '--segs', json.dumps(segs),
           '--csv', str(csv), '--out', str(WORK / f'c{ci}'), '--tag', tag, '--work', str(WORK / 'state'),
           '--seq-batch', str(SEQ_BATCH), '--state-dtype', 'auto', '--max-text-tokens', str(MAX_TEXT_TOKENS),
           '--prefix', PREFIX]
    t = time.time()
    with open(OUT / f'c{ci}_{tag}_pipeline.txt', 'w') as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    assert code == 0, f'Häppchen {ci} {tag}: Exit {code}, siehe c{ci}_{tag}_pipeline.txt'
    for f in (f'{tag}_features.csv', f'{tag}_timing.json'):
        shutil.copy(WORK / f'c{ci}' / f, OUT / f'pool_c{ci}_{f}')
    return time.time() - t

sys.path.insert(0, '/kaggle/temp/v1')
log = []
for ci, ids in enumerate(CHUNKS):
    elapsed = time.time() - T0
    est = max((r['flash_s'] + r['glm53_s']) for r in log) * 1.15 if log else 0
    if elapsed + est > BUDGET_S:
        print(f'Stopp vor Häppchen {ci}: {elapsed/3600:.2f} h vergangen, Schätzung {est/3600:.2f} h', flush=True)
        break
    csv = WORK / f'c{ci}.csv'
    pool.loc[ids, ['text']].reset_index().to_csv(csv, index=False)
    r = {'chunk': ci, 'n': len(ids)}
    r['flash_s'] = run_tag(ci, 'flash', FLASH, csv)
    r['glm53_s'] = run_tag(ci, 'glm53', GLM53, csv)
    log.append(r)
    json.dump(log, open(OUT / 'progress.json', 'w'), indent=1)
    print(f"Häppchen {ci}: {len(ids)} Texte, Flash {r['flash_s']/60:.0f} min, GLM-5.3 {r['glm53_s']/60:.0f} min, "
          f"gesamt {(time.time()-T0)/3600:.2f} h", flush=True)"""),
        code("""import glob as g
parts = []
for ci in range(len(CHUNKS)):
    fs = {t: OUT / f'pool_c{ci}_{t}_features.csv' for t in ('flash', 'glm53')}
    if all(f.exists() for f in fs.values()):
        a, b = (pd.read_csv(fs[t], dtype={'id': str}).set_index('id').add_prefix(f'{t}:') for t in ('flash', 'glm53'))
        parts.append(a.join(b, how='inner'))
feats = pd.concat(parts)
feats = feats.join(pool[['model']])
feats.reset_index().to_parquet(OUT / 'pool_feats_L160.parquet', index=False)
print(feats.shape, f'gesamt {(time.time()-T0)/3600:.2f} h'); print(feats['model'].value_counts().to_string())"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "glm_pool.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": "linuszarse/ntl-glm-likelihood-pool",
        "title": "NTL GLM Likelihood Pool",
        "code_file": "glm_pool.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["crtproduction/name-that-llm-open-pool",
                            "jamesliebel/glm53-iq3xxs-segs-a-mirror", "jamesliebel/glm53-iq3xxs-segs-b-mirror",
                            "jamesliebel/glm53-flash-iq3xxs-segs-mirror"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": ["estopadilla/ntl-glm-likelihood-train"],
        "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"geschrieben: {OUT_DIR}/glm_pool.ipynb (+ kernel-metadata.json)")


if __name__ == "__main__":
    build()
