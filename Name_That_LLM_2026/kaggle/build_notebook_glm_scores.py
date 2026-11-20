"""Erzeugt ein Kaggle-Notebook, das Texte unter GLM-5.3-Flash und GLM-5.3 bewertet und die Token-Scores speichert.

Nachfolger von build_notebook_glm_pool.py (Experiment 08, nur 160-Token-Merkmale). Hier werden die Token-Scores je
Text gespeichert (`*_scores.npz`: Log-Wahrscheinlichkeit, Entropie, m2, Rang je Token), mit langem Schnitt (512). Daraus
lassen sich lokal Merkmale für jeden kürzeren Schnitt (160, 256, 320) exakt nachrechnen (`features.one(s[:L], len(text))`,
das Modell ist kausal, `bpc`/`tok_per_char` nutzen die volle Textlänge). Anlass: das öffentliche 0.706-Notebook
(omerfarukmerey) liest in der GLM-Stufe 256 statt 160 Tokens (+0,002 auf dem LB, Kopf allein OOF 0.190 → 0.215).

Quellen:
- `train`: alle 2.699 Texte aus train.csv (2 Häppchen).
- `pool`: dieselbe Pool-Auswahl wie Experiment 08 (select() aus build_notebook_glm_pool.py, 7 Häppchen), so viele
  Häppchen, wie in BUDGET_H passen.

Code der Stufe (segio.py, features.py, run_pipeline.py) unverändert aus dem öffentlichen Notebook (Apache 2.0,
José Padilla / estopadilla), Binaries aus dem Output von estopadilla/ntl-glm-likelihood-train.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_glm_scores.py --source train
    .venv/bin/kaggle kernels push -p kaggle/glm_scores_train
    .venv/bin/python kaggle/build_notebook_glm_scores.py --source pool
    .venv/bin/kaggle kernels push -p kaggle/glm_scores_pool
"""
import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V1SRC = ROOT / "experiments" / "08_glm_likelihood" / "v1src"
BUDGET_H = 11.0          # Kaggle-GPU-Sitzung max. 12 h; Rest = Puffer für Ausgabe


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def pool_chunks():
    spec = importlib.util.spec_from_file_location("glm_pool", ROOT / "kaggle" / "build_notebook_glm_pool.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sel = mod.select()
    return [sel[sel.chunk == k]["id"].astype(str).tolist() for k in range(mod.N_CHUNKS)]


def train_chunks(n_chunks=2):
    ids = pd.read_csv(ROOT / "data" / "train.csv", usecols=["id"]).id.astype(str).tolist()
    return [ids[k::n_chunks] for k in range(n_chunks)]


def build(source, max_tokens):
    chunks = train_chunks() if source == "train" else pool_chunks()
    out_dir = ROOT / "kaggle" / f"glm_scores_{source}"
    slug = f"ntl-glm-scores-{source}{max_tokens}"
    print(source, "Häppchen", [len(c) for c in chunks], "gesamt", sum(len(c) for c in chunks))
    srcs = {n: (V1SRC / n).read_text() for n in ("segio.py", "features.py", "run_pipeline.py")}
    text_load = (
        "src = pd.read_csv(find_one('name-that-llm/train.csv'), usecols=['id', 'text', 'model'])"
        if source == "train" else
        "src = pd.read_parquet(find_one('ntl_open_pool.parquet'), columns=['id', 'text', 'model'])")
    cells = [
        md(f"# Name That LLM – GLM-Token-Scores ({source}, erste {max_tokens} Tokens)\n\n"
           f"Bewertet {'alle train.csv-Texte' if source == 'train' else 'ausgewählte Open-Pool-Texte (crtproduction, CC BY-SA 4.0)'} "
           "unter **GLM-5.3-Flash** und **GLM-5.3** (Einstellungen wie die GLM-Stufe der öffentlichen Abgaben) und "
           "speichert die Token-Scores je Text (`out/c*_{flash,glm53}_scores.npz`), damit sich Merkmale für jeden "
           "kürzeren Schnitt nachrechnen lassen.\n\n"
           "Code der Stufe (`segio.py`, `features.py`, `run_pipeline.py`) und Binaries (`ntl-segrun`): José Padilla "
           "(estopadilla), Apache 2.0. GLM-Segmente: jamesliebel."),
        code(f"""MAX_TEXT_TOKENS = {max_tokens}
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
segrun = find_one('ntl-bin/ntl-segrun')
ART = Path(segrun).parent
for f in ('lib/libcublas.so.13', 'lib/libcublasLt.so.13', 'lib/libcudart.so.13'):
    assert (ART / f).is_file(), ART / f
Path('/kaggle/temp/v1').mkdir(parents=True, exist_ok=True)   # Ziel der %%writefile-Zellen
BIN = Path('/kaggle/temp/ntl-bin'); BIN.mkdir(parents=True, exist_ok=True)
shutil.copyfile(segrun, BIN / 'ntl-segrun'); (BIN / 'ntl-segrun').chmod(0o755)
ENV = dict(os.environ, LD_LIBRARY_PATH=str(ART / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
subprocess.run('nvidia-smi -L; df -h /kaggle/temp /kaggle/working | tail -2', shell=True)
print('Segmente ok | BIN', ART)"""),
        *[code(f"%%writefile /kaggle/temp/v1/{n}\n{s}") for n, s in srcs.items()],
        code(text_load + """
src = src.astype({'id': str}).set_index('id')
wanted = [i for c in CHUNKS for i in c]
missing = [i for i in wanted if i not in src.index]
assert not missing, f'{len(missing)} IDs fehlen (andere Dataset-Version?)'
OUT = Path('/kaggle/working/out'); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path('/kaggle/temp/work'); WORK.mkdir(parents=True, exist_ok=True)
src.loc[wanted, ['model']].reset_index().to_csv(OUT / 'labels.csv', index=False)
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
    for f in (f'{tag}_scores.npz', f'{tag}_features.csv', f'{tag}_timing.json'):
        shutil.copy(WORK / f'c{ci}' / f, OUT / f'c{ci}_{f}')
    return time.time() - t

log = []
for ci, ids in enumerate(CHUNKS):
    elapsed = time.time() - T0
    est = max((r['flash_s'] + r['glm53_s']) for r in log) * 1.15 if log else 0
    if elapsed + est > BUDGET_S:
        print(f'Stopp vor Häppchen {ci}: {elapsed/3600:.2f} h vergangen, Schätzung {est/3600:.2f} h', flush=True)
        break
    csv = WORK / f'c{ci}.csv'
    src.loc[ids, ['text']].reset_index().to_csv(csv, index=False)
    r = {'chunk': ci, 'n': len(ids)}
    r['flash_s'] = run_tag(ci, 'flash', FLASH, csv)
    r['glm53_s'] = run_tag(ci, 'glm53', GLM53, csv)
    log.append(r)
    json.dump(log, open(OUT / 'progress.json', 'w'), indent=1)
    print(f"Häppchen {ci}: {len(ids)} Texte, Flash {r['flash_s']/60:.0f} min, GLM-5.3 {r['glm53_s']/60:.0f} min, "
          f"gesamt {(time.time()-T0)/3600:.2f} h", flush=True)
print('fertig:', [r['chunk'] for r in log], f'{(time.time()-T0)/3600:.2f} h')"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    datasets = ["jamesliebel/glm53-iq3xxs-segs-a-mirror", "jamesliebel/glm53-iq3xxs-segs-b-mirror",
                "jamesliebel/glm53-flash-iq3xxs-segs-mirror"]
    if source == "pool":
        datasets.insert(0, "crtproduction/name-that-llm-open-pool")
    meta = {
        "id": f"linuszarse/{slug}",
        "title": f"NTL GLM Scores {source.capitalize()}{max_tokens}",   # Kaggle leitet daraus den Slug ab
        "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": datasets,
        "competition_sources": ["name-that-llm"],
        "kernel_sources": ["estopadilla/ntl-glm-likelihood-train"],
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"geschrieben: {out_dir}/{slug}.ipynb (+ kernel-metadata.json), Kernel {meta['id']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["train", "pool"], required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    build(a.source, a.max_tokens)
