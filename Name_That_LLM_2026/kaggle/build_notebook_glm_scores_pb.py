"""Erzeugt ein Kaggle-Notebook, das Open-Pool-Texte im PB-115-Format unter GLM-5.3-Flash und GLM-5.3 bewertet.

Anlass (Experiment 17): Das öffentliche 0.724-Notebook (lumibee v69, unser V21-Fork) trainiert seinen GLM-Kopf
(PB-115 = 54 Zusammenfassungen + 61 token-gleiche Differenzen, LR + LGB) nur auf den 2.699 train.csv-Texten. Bei uns
hat das Hinzunehmen von Pool-Texten den GLM-Kopf zweimal verbessert (V4 → V5 +1,0, V6 +0,6). Unsere bisherigen
Pool-Scores (Experiment 08/14) passen aber nicht: PB-115 nutzt „Prefix B“ mit einem Nutzer-Turn je Text und 160 Tokens.

Unterschiede zu build_notebook_glm_scores.py:
- PREFIX/USER_TEMPLATE/MAX_TEXT_TOKENS genau wie `ll_model.json` des 0.724-Notebooks (Reasoning Effort: Low,
  „Write a single paragraph of about {n_words} words, …“, 160 Tokens).
- `run_pipeline.py` (S13/S14, mit `--user-template`) aus dem Payload des 0.724-Notebooks; `--segs` ist ein Glob.
- Binary `ntl-segrun` aus dem Output von estopadilla/ntl-glm-prefix-ab, ausgewählt über denselben sha256 wie dort.
- Häppchen 0 enthält zusätzlich CHECK_N train.csv-Texte. Daran prüft die letzte Zelle unsere Merkmale gegen
  lumibees train-Merkmale (`lumibee/ntl-prefixb-full-115`): n_tok muss exakt gleich sein (gleicher Prefix, gleiche
  Tokenisierung), lp_mean stark korreliert (Laufrauschen der MoE-GGUFs ist erwartet).

Pool-Auswahl wie Experiment 08 (select() aus build_notebook_glm_pool.py, 5.670 Texte in 7 Häppchen), so viele
Häppchen, wie in BUDGET_H passen (Schätzung nach Experiment 14: ~1,6 h je Häppchen à 810 Texte).

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/build_notebook_glm_scores_pb.py
    .venv/bin/kaggle kernels push -p kaggle/glm_scores_poolpb
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
from build_notebook_glm_scores import code, md, pool_chunks  # noqa: E402

V1SRC = ROOT / "experiments" / "17_pb115_pool" / "v1src"
OUT_DIR = ROOT / "kaggle" / "glm_scores_poolpb"
SLUG = "ntl-glm-scores-poolpb160"
BUDGET_H = 11.0          # Kaggle-GPU-Sitzung max. 12 h; Rest = Puffer für Ausgabe
CHECK_N = 40
SEGRUN_SHA = "a9e071e1529b46f2c79b11a1c0466a83e897bfd01bd0d3fc28f3a859626f25b0"   # wie im 0.724-Notebook
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:fc720c15545eb096a510e1f085b17fc118387a007599424d953a4f327b71cacf"


def build():
    ll = json.loads((V1SRC / "ll_model.json").read_text())
    prefix, user_template, max_tokens = ll["prefix"], ll["user_template"], ll["max_text_tokens"]
    assert "{user}" in prefix and max_tokens == 160, (prefix, max_tokens)
    chunks = pool_chunks()
    train_ids = pd.read_csv(ROOT / "data" / "train.csv", usecols=["id"]).id.astype(str)
    check_ids = train_ids.sample(n=CHECK_N, random_state=0).tolist()
    print("Pool-Häppchen", [len(c) for c in chunks], "gesamt", sum(len(c) for c in chunks), "+", CHECK_N, "train")
    srcs = {n: (V1SRC / n).read_text() for n in ("segio.py", "features.py", "run_pipeline.py")}
    cells = [
        md("# Name That LLM – GLM-Token-Scores im PB-115-Format (Open Pool, Prefix B, 160 Tokens)\n\n"
           "Bewertet ausgewählte Open-Pool-Texte (crtproduction, CC BY-SA 4.0) unter **GLM-5.3-Flash** und "
           "**GLM-5.3** mit genau den Einstellungen der PB-115-Stufe (Prefix B mit Nutzer-Turn je Text, 160 Tokens) "
           "und speichert Merkmale und Token-Scores je Text (`out/c*_{flash,glm53}_{features.csv,scores.npz}`).\n\n"
           "Code der Stufe (`segio.py`, `features.py`, `run_pipeline.py`) und Binaries (`ntl-segrun`): José Padilla "
           "(estopadilla), Apache 2.0. Prefix B: prvsiyan. PB-115: lumibee. GLM-Segmente: jamesliebel."),
        code(f"""MAX_TEXT_TOKENS = {max_tokens}
BUDGET_S = {BUDGET_H} * 3600   # keine neuen Häppchen mehr, wenn die geschätzte Dauer darüber hinausgeht
SEQ_BATCH = 8
PREFIX = {prefix!r}
USER_TEMPLATE = {user_template!r}
SEGRUN_SHA = {SEGRUN_SHA!r}
CHECK_IDS = {json.dumps(check_ids)}
CHUNKS = {json.dumps(chunks)}
import time; T0 = time.time()"""),
        code("""import glob, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

def find_one(pattern):
    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True))
    assert hits, f'Input fehlt: {pattern}'
    return hits[0]

def seg_dir(pattern, n, name):
    seen = {}
    for p in sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True)):
        seen.setdefault(os.path.basename(p), p)          # dieselbe Datei kann aus zwei Mirrors kommen
    assert len(seen) == n, (name, len(seen))
    d = Path(f'/kaggle/temp/segs-{name}'); d.mkdir(parents=True, exist_ok=True)
    for b, p in seen.items():
        if not (d / b).exists(): (d / b).symlink_to(p)
    return str(d / '*.gguf')

FLASH = seg_dir('flash_iq3xxs-seg*-of-06-*.gguf', 6, 'flash')
GLM53 = seg_dir('glm53_iq3xxs-seg*-of-19-*.gguf', 19, 'glm53')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

cands = sorted(glob.glob('/kaggle/input/**/ntl-bin/ntl-segrun', recursive=True))
ok = [p for p in cands if sha(p) == SEGRUN_SHA and Path(p).with_name('lib').joinpath('libcublasLt.so.13').exists()]
assert ok, f'kein ntl-segrun mit dem erwarteten sha256 und lib/ unter {cands}'
ART = Path(ok[0]).parent
Path('/kaggle/temp/v1').mkdir(parents=True, exist_ok=True)   # Ziel der %%writefile-Zellen
BIN = Path('/kaggle/temp/ntl-bin'); BIN.mkdir(parents=True, exist_ok=True)
shutil.copyfile(ok[0], BIN / 'ntl-segrun'); (BIN / 'ntl-segrun').chmod(0o755)
ENV = dict(os.environ, LD_LIBRARY_PATH=str(ART / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
subprocess.run('nvidia-smi -L; df -h /kaggle/temp /kaggle/working | tail -2', shell=True)
print('Segmente ok | BIN', ART)"""),
        *[code(f"%%writefile /kaggle/temp/v1/{n}\n{s}") for n, s in srcs.items()],
        code("""pool = pd.read_parquet(find_one('ntl_open_pool.parquet'), columns=['id', 'text', 'model'])
train = pd.read_csv(find_one('name-that-llm/train.csv'), usecols=['id', 'text', 'model'])
pool = pool.astype({'id': str}).set_index('id')
train = train.astype({'id': str}).set_index('id')
wanted = [i for c in CHUNKS for i in c]
missing = [i for i in wanted if i not in pool.index]
assert not missing, f'{len(missing)} Pool-IDs fehlen (andere Dataset-Version?)'
assert not set(CHECK_IDS) & set(pool.index)
src = pd.concat([pool.loc[wanted], train.loc[CHECK_IDS]])
CHUNKS[0] = CHECK_IDS + CHUNKS[0]          # Prüftexte zuerst: Häppchen 0 ist immer dabei
OUT = Path('/kaggle/working/out'); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path('/kaggle/temp/work'); WORK.mkdir(parents=True, exist_ok=True)
lab = src.loc[[i for c in CHUNKS for i in c], ['model']].reset_index()
lab['source'] = np.where(lab.id.isin(CHECK_IDS), 'train', 'pool')
lab.to_csv(OUT / 'labels.csv', index=False)
print(len(lab), 'Texte in', len(CHUNKS), 'Häppchen, davon', len(CHECK_IDS), 'train-Prüftexte')"""),
        code("""def run_tag(ci, tag, segs, csv):
    cmd = [sys.executable, '/kaggle/temp/v1/run_pipeline.py', '--bin', str(BIN), '--segs', segs,
           '--csv', str(csv), '--out', str(WORK / f'c{ci}'), '--tag', tag, '--work', str(WORK / 'state'),
           '--seq-batch', str(SEQ_BATCH), '--state-dtype', 'auto', '--max-text-tokens', str(MAX_TEXT_TOKENS),
           '--prefix', PREFIX, '--user-template', USER_TEMPLATE]
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
    if ci == 0:   # Formatprüfung gegen lumibees train-Merkmale (PB-115), sobald Häppchen 0 fertig ist
        rep = {}
        for tag in ('flash', 'glm53'):
            ours = pd.read_csv(OUT / f'c0_{tag}_features.csv', dtype={'id': str}).set_index('id').loc[CHECK_IDS]
            ref = pd.read_csv(find_one(f'ntl-prefixb-full-115/{tag}_features.csv'), dtype={'id': str}).set_index('id').loc[CHECK_IDS]
            rep[tag] = {'n_tok_equal': int((ours.n_tok == ref.n_tok).sum()),
                        'corr_lp_mean': float(np.corrcoef(ours.lp_mean, ref.lp_mean)[0, 1]),
                        'mad_lp_mean': float((ours.lp_mean - ref.lp_mean).abs().median())}
        json.dump(rep, open(OUT / 'format_check.json', 'w'), indent=1)
        print('Formatprüfung:', rep, flush=True)
        assert all(v['n_tok_equal'] == len(CHECK_IDS) for v in rep.values()), 'Tokenisierung weicht ab: falscher Prefix?'
print('fertig:', [r['chunk'] for r in log], f'{(time.time()-T0)/3600:.2f} h')"""),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": f"linuszarse/{SLUG}",
        "title": "NTL GLM Scores PoolPB160",   # Kaggle leitet daraus den Slug ab
        "code_file": f"{SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "docker_image": DOCKER,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["crtproduction/name-that-llm-open-pool", "lumibee/ntl-prefixb-full-115",
                            "jamesliebel/glm53-iq3xxs-segs-a-mirror", "jamesliebel/glm53-iq3xxs-segs-b-mirror",
                            "jamesliebel/glm53-flash-iq3xxs-segs-mirror"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": ["estopadilla/ntl-glm-prefix-ab"],
        "model_sources": [],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"geschrieben: {OUT_DIR}/{SLUG}.ipynb (+ kernel-metadata.json), Kernel {meta['id']}")


if __name__ == "__main__":
    build()
