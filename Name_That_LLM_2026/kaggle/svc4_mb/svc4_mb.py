"""Experiment 18 (Kaggle CPU): svc4 des 0.727-Kerns neu fitten, ohne und mit markbarney-Texten.

Rezept wie `ntl666-prefit/provenance.json` (areebbb/ntl-we-back-0-666 V14): LinearSVC(C 0.5, balanced,
random_state 42, max_iter 5000) auf den 4 TF-IDF-Ansichten des Kerns (Vektorisierer unverändert aus dem Kern),
train w1 + dhruvtibarewal/nlm-matched-clean w0.5 + Open Pool (hier aktuelle Version statt v13) ohne train/matched-clean
w0.35; Variante „mit“: + markbarney (fünf Modelle × 1.000 Prompts + vier Modelle) w0.5.
Ausgabe: svc4_ohne_model.npz, svc4_mit_model.npz (Format wie core/svc4/model.npz), report.json mit Treuetest
gegen das Original-model.npz.
"""
import gc
import glob
import json
import os
import re
import shutil
import time
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.svm import LinearSVC

T0 = time.time()
BASE = os.environ.get("NTL_BASE", "")          # nur für lokale Tests
SMALL = int(os.environ.get("NTL_SMALL", "0"))   # nur für lokale Tests: Zeilen je Quelle
WORK = Path(BASE + "/kaggle/temp/svc4mb")
OUT = Path(BASE + "/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
FW = frozenset('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())


def log(*a):
    try:
        import psutil
        mem = f"{psutil.Process().memory_info().rss / 1e9:5.1f} GB"
    except Exception:
        mem = ""
    print(f"[{time.time() - T0:6.0f}s {mem}]", *a, flush=True)


def first(pattern):
    hits = sorted(glob.glob(BASE + pattern, recursive=True))
    assert hits, pattern
    return hits[0]


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(t or ""))).strip()


def shape_view(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_view(t):
    return " ".join(w.lower() if w.lower() in FW or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))


# --- Kern entpacken (wie Zelle 5 des 0.727-Notebooks) ---
root = os.path.dirname(first("/kaggle/input/**/ntl666-cache-parts.json"))
meta = json.loads(open(f"{root}/ntl666-cache-parts.json").read())
zpath = WORK / "prefit.zip"
with open(zpath, "wb") as out:
    for part in meta["parts"]:
        with open(f"{root}/{part['name']}", "rb") as f:
            shutil.copyfileobj(f, out, 16 * 1024 * 1024)
with zipfile.ZipFile(zpath) as z:
    for e in z.infolist():
        p = PurePosixPath(e.filename)
        assert not p.is_absolute() and ".." not in p.parts and p.parts[0] == "ntl666-prefit"
    z.extractall(WORK)
zpath.unlink()
PREFIT = WORK / "ntl666-prefit"
svc_meta = json.loads((PREFIT / "core/svc4/metadata.json").read_text())
assert svc_meta["labels"] == LABELS
VECS = [joblib.load(PREFIT / "core/svc4" / v["file"]) for v in svc_meta["views"]]
PREPS = {"char": lambda s: s, "word": lambda s: s, "shape": shape_view, "function": function_view}
log("Kern entpackt", [(v["view"], v["weight"]) for v in svc_meta["views"]])

# --- Daten ---
data = next(BASE + r for r in ("/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm")
            if os.path.isfile(f"{BASE}{r}/train.csv"))
tr = pd.read_csv(f"{data}/train.csv")[["text", "model"]]
mc = pd.read_parquet(first("/kaggle/input/**/nlm_matched_clean.parquet"))[["text", "model"]]
pool = pd.read_parquet(first("/kaggle/input/**/ntl_open_pool.parquet"), columns=["text", "model"])
mb_files = sorted(glob.glob(BASE + "/kaggle/input/**/passages.csv", recursive=True))
mb_files = [f for f in mb_files if "matched-passages" in f]
assert len(mb_files) == 2, mb_files
mb = pd.concat([pd.read_csv(f)[["text", "model"]] for f in mb_files], ignore_index=True)
for d in (tr, mc, pool, mb):
    d["text"] = d.text.fillna("").astype(str)
keys = set(tr.text.map(norm))
mc = mc[mc.model.isin(LABELS) & ~mc.text.map(norm).isin(keys)].drop_duplicates("text")
keys |= set(mc.text.map(norm))
pool = pool[pool.model.isin(LABELS) & ~pool.text.map(norm).isin(keys)].drop_duplicates("text")
keys |= set(pool.text.map(norm))
mb = mb[mb.model.isin(LABELS) & ~mb.text.map(norm).isin(keys)].drop_duplicates("text")
if SMALL:
    mc, pool, mb = (d.sample(min(SMALL, len(d)), random_state=0) for d in (mc, pool, mb))
SRC = {"train": (tr, 1.0), "matched_clean": (mc, 0.5), "pool": (pool, 0.35), "markbarney": (mb, 0.5)}
report = {"rows": {k: len(v[0]) for k, v in SRC.items()}, "weights": {k: v[1] for k, v in SRC.items()},
          "markbarney_classes": mb.model.value_counts().to_dict(), "files": mb_files}
log("Zeilen", report["rows"])


# --- Merkmale je Quelle, auf Platte ---
def transform(texts):
    return sp.hstack([v.transform([PREPS[m["view"]](t) for t in texts]) * m["weight"]
                      for v, m in zip(VECS, svc_meta["views"])], format="csr", dtype=np.float32)


for name, (d, _) in SRC.items():
    for i in range(0, len(d), 50000):
        sp.save_npz(WORK / f"X_{name}_{i // 50000:03d}.npz", transform(d.text.iloc[i:i + 50000].tolist()), compressed=False)
    log("Merkmale", name)
X_train = transform(tr.text.tolist())


def load(names):
    X = sp.vstack([sp.load_npz(p) for n in names for p in sorted(WORK.glob(f"X_{n}_*.npz"))], format="csr")
    y = np.concatenate([SRC[n][0].model.map(L2I).to_numpy(np.int64) for n in names])
    w = np.concatenate([np.full(len(SRC[n][0]), SRC[n][1]) for n in names])
    return X, y, w


orig = np.load(PREFIT / "core/svc4/model.npz")
S_orig = X_train @ orig["coef"].T + orig["intercept"]
X_mb, y_mb, _ = load(["markbarney"])            # für Original und „ohne“ ungesehen → Treuetest
S_orig_mb = X_mb @ orig["coef"].T + orig["intercept"]
report["original_markbarney_acc"] = float((S_orig_mb.argmax(1) == y_mb).mean())
report["fits"] = {}
for tag, names in (("ohne", ["train", "matched_clean", "pool"]), ("mit", ["train", "matched_clean", "pool", "markbarney"])):
    X, y, w = load(names)
    X = X.astype(np.float64)
    gc.collect()
    log(tag, "Fit", X.shape, f"nnz {X.nnz / 1e6:.0f} M")
    t = time.time()
    m = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(X, y, sample_weight=w)
    assert m.classes_.tolist() == list(range(len(LABELS)))
    del X
    gc.collect()
    np.savez(OUT / f"svc4_{tag}_model.npz", coef=m.coef_.astype(np.float64), intercept=m.intercept_.astype(np.float64),
             classes=m.classes_.astype(np.int64))
    S = X_train @ m.coef_.T + m.intercept_
    S_mb = X_mb @ m.coef_.T + m.intercept_
    yt = tr.model.map(L2I).to_numpy()
    report["fits"][tag] = dict(
        seconds=round(time.time() - t), n_iter=int(np.max(m.n_iter_)),
        markbarney_argmax_agree_with_original=float((S_mb.argmax(1) == S_orig_mb.argmax(1)).mean()),
        markbarney_mean_row_corr_with_original=float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(S_mb, S_orig_mb)])),
        markbarney_acc=float((S_mb.argmax(1) == y_mb).mean()),
        argmax_agree_with_original=float((S.argmax(1) == S_orig.argmax(1)).mean()),
        mean_row_corr_with_original=float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(S, S_orig)])),
        train_insample_acc=float((S.argmax(1) == yt).mean()))
    log(tag, report["fits"][tag])
report["original_train_insample_acc"] = float((S_orig.argmax(1) == tr.model.map(L2I).to_numpy()).mean())
(OUT / "report.json").write_text(json.dumps(report, indent=2))
shutil.rmtree(WORK, ignore_errors=True)
log("fertig", json.dumps(report["fits"]))
