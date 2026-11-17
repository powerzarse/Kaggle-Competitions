"""Experiment 09: Zusammenspiel GLM-Stufe <-> Apex-Nachbearbeitung im 0.702-Notebook, lokal auf train.csv.

Kette wie in kaggle/public_702 (V7):
  Stack-Logits -> GLM-Stufe (Top 15 % nach GLM-Masse, zl + W/T' * ll, unser HGB-Klassifikator)
  -> Apex (Geschwister-Residual auf den Stack-Logits + weiche Typografie-Regeln; GLM-Labels bleiben erhalten).
Problem: Apex rechnet mit den Stack-Logits *vor* der GLM-Stufe und überschreibt jede Nicht-GLM-Vorhersage.
Was die GLM-Stufe von GLM weg verschiebt, geht damit verloren.

Ersatz für den unbekannten Stack: unser Ensemble (BERT 0.4 / SVM 0.4 / LoRA 0.2, OOF ~0.60). Alles, was lernt
(HGB-Klassifikator, Residual), ist OOF über dieselben GroupKFold-Folds. Die Zahlen vergleichen Varianten
der Nachbearbeitung, sie schätzen nicht den LB.

Code der Basis wird direkt aus kaggle/public_702/public_702.ipynb geladen (Apache 2.0, Credits dort).

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python experiments/09_postprocess/simulate.py
"""
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

from src.cv import get_folds
from src.data import load_train

NB = ROOT / "kaggle" / "public_702" / "public_702.ipynb"
NPZ = ROOT / "kaggle" / "public_694" / "v5_hgb_data" / "ll_train.npz"
OOF = ROOT / "models" / "07_kaggle"
OUT = ROOT / "models" / "09_postprocess"
GLM = ["glm-5.3", "glm-5.3-flash"]
HGB = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
           min_samples_leaf=40, early_stopping=False)


def load_modules():
    """pngy_residual, apex_postprocess und v1_stage aus den %%writefile-Zellen des Notebooks."""
    cells = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
             for c in json.loads(NB.read_text())["cells"]]
    mods = {}
    for name in ("pngy_residual.py", "apex_postprocess.py", "/kaggle/temp/v1/v1_stage.py"):
        body = [c for c in cells if c.startswith("%%writefile " + name)]
        assert len(body) == 1, name
        mod = types.ModuleType(Path(name).stem)
        sys.modules[mod.__name__] = mod
        exec(compile(body[0].split("\n", 1)[1], name, "exec"), mod.__dict__)
        mods[mod.__name__] = mod
    model_cell = [c for c in cells if "ll_model.json').write_text(" in c]
    assert len(model_cell) == 1
    s = model_cell[0]
    ll_model = json.loads(s[s.index("write_text('") + len("write_text('"):s.rindex("')")])
    return mods["pngy_residual"], mods["apex_postprocess"], mods["v1_stage"], ll_model


def stack_proxy(ids, labels):
    load = lambda f: pd.read_parquet(OOF / f).set_index("id").loc[ids, labels].to_numpy(np.float64)
    P = (0.4 * load("bert_stage2_oof.parquet") + 0.4 * softmax(load("svm_oof_decision.parquet") * 5, 1)
         + 0.2 * load("lora_qwen3_4b_24k_oof.parquet"))
    return np.log(np.clip(P, 1e-9, 1))


def oof_ll(X, y, w, n_train, folds, labels):
    """HGB wie in V5–V7: train (Gewicht 1) + kompletter Pool (Gewicht aus npz), OOF auf den train-Folds."""
    Xt, yt = X[:n_train], y[:n_train]
    Xp, yp, wp = X[n_train:], y[n_train:], w[n_train:]
    out = np.zeros((n_train, len(labels)))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        clf = HistGradientBoostingClassifier(random_state=0, **HGB).fit(
            np.vstack([Xt[tr], Xp]), np.concatenate([yt[tr], yp]), sample_weight=np.concatenate([np.ones(tr.sum()), wp]))
        p = clf.predict_proba(Xt[va])
        out[va] = np.log(np.clip(p[:, [list(clf.classes_).index(c) for c in labels]], 1e-9, 1))
    return out


def oof_evidence(residual_mod, texts, y, folds, labels):
    ev = np.ones((len(texts), len(labels)))
    x = residual_mod.features(texts)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        model = residual_mod.Residual(labels).fit(np.asarray(texts, dtype=object)[tr], y[tr])
        for _, est in model.models:
            p = est.predict_proba(x[va])
            for j, lab in enumerate(est.classes_):
                ev[va, labels.index(lab)] = p[:, j]
    return ev, residual_mod.family_columns(labels)


def glm_stage(zl, ll, labels, entropy_ref, match_temperature, W=1.0, cover=0.15):
    mass = np.exp(zl[:, [labels.index(c) for c in GLM]]).sum(1)
    sel = np.sort(np.argsort(-mass, kind="stable")[:int(cover * len(zl))])
    tp = match_temperature(zl, entropy_ref)
    post = zl.copy()
    post[sel] += W / tp * ll[sel]
    return post, sel


def bootstrap(y, a, b, groups, n=2000, seed=0):
    """Differenz balanced accuracy (b - a), Bootstrap über Prompts."""
    rng = np.random.default_rng(seed)
    g = pd.factorize(groups)[0]
    idx = [np.flatnonzero(g == i) for i in range(g.max() + 1)]
    d = []
    for _ in range(n):
        s = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        d.append(balanced_accuracy_score(y[s], b[s]) - balanced_accuracy_score(y[s], a[s]))
    d = np.asarray(d)
    return d.mean(), np.quantile(d, 0.05), np.quantile(d, 0.95), (d > 0).mean()


def main():
    res_mod, apex, v1, ll_model = load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    folds = get_folds(real)
    L = np.asarray(labels)

    with np.load(NPZ, allow_pickle=False) as d:
        X, yy, w = d["X"].astype(np.float64), d["y"].astype(str), d["w"]
    assert (yy[:len(real)] == y).all()

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "oof_cache.npz"
    if cache.exists():
        with np.load(cache) as c:
            ll, ev = c["ll"], c["ev"]
        families = res_mod.family_columns(labels)
    else:
        ll = oof_ll(X, yy, w, len(real), folds, labels)
        ev, families = oof_evidence(res_mod, texts, y, folds, labels)
        np.savez(cache, ll=ll, ev=ev)

    zl = log_softmax(stack_proxy(real.id, labels), 1)
    post, sel = glm_stage(zl, ll, labels, ll_model["entropy_ref"], v1.match_temperature)
    base_pred, stage_pred = L[zl.argmax(1)], L[post.argmax(1)]
    ba = lambda p: balanced_accuracy_score(y, p)

    def apex_run(base, current, corrected=False, strength=0.10, penalty=0.05):
        out, _ = apex.apply_arrays(base, ev, labels, families, texts, current, strength, corrected, penalty)
        return out.astype(str)

    variants = {
        "Stack allein": base_pred,
        "Stack + GLM-Stufe (ohne Apex)": stage_pred,
        "Stack + Apex (ohne GLM-Stufe)": apex_run(zl, base_pred, corrected=True),
        "0.702 wie öffentlich (Apex auf Stack-Logits)": apex_run(zl, stage_pred),
        "  dito, Typografie hart (penalty 0, wie V6)": apex_run(zl, stage_pred, penalty=0.0),
        "Apex auf Logits nach GLM-Stufe, GLM-Labels behalten": apex_run(post, stage_pred),
        "Apex auf Logits nach GLM-Stufe, alles neu (corrected=True)": apex_run(post, stage_pred, corrected=True),
    }
    for s in (0.2, 0.3):
        variants[f"  Apex nach GLM-Stufe, strength {s}"] = apex_run(post, stage_pred, strength=s)

    ref = variants["0.702 wie öffentlich (Apex auf Stack-Logits)"]
    changed = stage_pred != base_pred
    lost = changed & (ref != stage_pred) & ~np.isin(stage_pred, GLM)
    print(f"Texte: {len(y)}, GLM-Stufe bewertet {len(sel)} ({len(sel) / len(y):.0%}), ändert {changed.sum()} Vorhersagen")
    print(f"  davon nach GLM verschoben: {(changed & np.isin(stage_pred, GLM)).sum()}, "
          f"von GLM weg: {(changed & np.isin(base_pred, GLM) & ~np.isin(stage_pred, GLM)).sum()}")
    print(f"  im öffentlichen Apex wieder überschrieben: {lost.sum()} "
          f"(davon war die GLM-Stufe richtig: {(lost & (stage_pred == y)).sum()}, "
          f"Apex-Ergebnis richtig: {(lost & (ref == y)).sum()})\n")
    print(f"{'Variante':<62} {'bal.acc':>8} {'Δ vs 0.702':>10}")
    for name, p in variants.items():
        print(f"{name:<62} {ba(p):>8.4f} {ba(p) - ba(ref):>+10.4f}")

    best = "Apex auf Logits nach GLM-Stufe, GLM-Labels behalten"
    m, lo, hi, pp = bootstrap(y, ref, variants[best], real.prompt_id)
    print(f"\nBootstrap über Prompts, '{best}' vs 0.702: {m:+.4f} [{lo:+.4f}; {hi:+.4f}] (90 %), P(>0) = {pp:.2f}")
    glm = np.isin(y, GLM)
    for name in ("0.702 wie öffentlich (Apex auf Stack-Logits)", best):
        p = variants[name]
        print(f"  {name[:45]:<45} Recall GLM-5.3 {(p[y == 'glm-5.3'] == 'glm-5.3').mean():.3f}, "
              f"GLM-Flash {(p[y == 'glm-5.3-flash'] == 'glm-5.3-flash').mean():.3f}, "
              f"Nicht-GLM als GLM vorhergesagt {np.isin(p[~glm], GLM).mean():.3f}")


if __name__ == "__main__":
    main()
