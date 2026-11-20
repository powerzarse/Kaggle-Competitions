"""Experiment 16: gepaarte Token-Merkmale (GLM-5.3 gegen Flash auf denselben Tokens) für den V6-Kopf.

Beide GLMs tokenisieren identisch, je Token liegen lp, Entropie, m2 und Rang unter beiden Modellen vor. Die bisherigen
54 Merkmale vergleichen nur Mittelwerte je Modell (d:… = Flash − GLM-5.3). Gepaarte Merkmale nutzen die Verteilung
der Differenz je Token (t-Statistik, Anteil positiv, Quantile, Rangvergleich, nach Entropie getrennt). Sie kosten
zur Laufzeit nichts, die Token-Scores entstehen ohnehin. Am 17.11. auf train allein +0,26; damals fehlten
Token-Scores für den Pool. Jetzt gibt es sie (Experiment 14):
- train: `models/14_glm_256/train512/out/` (Lauf B, alle 2.699 Texte)
- Pool: `models/14_glm_256/pool512/out/` (4.734 Texte, Auswahl wie Experiment 08)

Köpfe (HGB wie V6, Pool-Gewicht 0.5, Schnitt 160 Tokens wie V6), OOF über die Prompt-Folds, alles auf unseren
eigenen Läufen (B → B). In der V6-Kette (Ersatz-Stack → GLM-Stufe 20 % → Apex) zählt nur die GLM-Entscheidung.

Aufruf (aus Name_That_LLM_2026/): .venv/bin/python experiments/16_paired_tokens/evaluate.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))
sys.path.insert(0, str(ROOT / "experiments" / "08_glm_likelihood" / "v1src"))
sys.path.insert(0, str(ROOT / "experiments" / "14_glm_256"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score

import features
import simulate as s9
from evaluate import hgb, matrix
from src.cv import get_folds
from src.data import load_train

TRAIN = ROOT / "models" / "14_glm_256" / "train512" / "out"
POOL = ROOT / "models" / "14_glm_256" / "pool512" / "out"
OUT = ROOT / "models" / "16_paired_tokens"
CUT = 160
POOL_W = 0.5


def load_pairs(run):
    """id -> (Flash-Scores, GLM-5.3-Scores), gleiche Tokens geprüft."""
    out = {}
    for f in sorted(run.glob("c*_flash_scores.npz")):
        g = f.with_name(f.name.replace("_flash_", "_glm53_"))
        if not g.exists():
            continue
        with np.load(f, allow_pickle=True) as a, np.load(g, allow_pickle=True) as b:
            assert (a["ids"] == b["ids"]).all() and (a["offsets"] == b["offsets"]).all()
            assert (a["tokens"] == b["tokens"]).all(), f
            ids, off, sa, sb = a["ids"].astype(str), a["offsets"], a["scores"], b["scores"]
        for k, i in enumerate(ids):
            out[i] = (sa[off[k]:off[k + 1]], sb[off[k]:off[k + 1]])
    return out


def paired(sf, sg):
    lf, lg = sf[:, 0].astype(np.float64), sg[:, 0].astype(np.float64)
    hf, hg = sf[:, 1].astype(np.float64), sg[:, 1].astype(np.float64)
    vf, vg = np.maximum(sf[:, 2] - hf * hf, 1e-9), np.maximum(sg[:, 2] - hg * hg, 1e-9)
    rf, rg = sf[:, 3], sg[:, 3]
    d = lg - lf                      # > 0: GLM-5.3 findet das Token wahrscheinlicher als Flash
    n = len(d)
    q = np.quantile(d, [0.10, 0.25, 0.50, 0.75, 0.90])
    lo = hf <= np.median(hf)
    zf, zg = (lf + hf) / np.sqrt(vf), (lg + hg) / np.sqrt(vg)
    return {
        "p:d_mean": d.mean(), "p:d_std": d.std(), "p:d_t": d.mean() / (d.std() + 1e-6) * np.sqrt(n),
        "p:d_pos": (d > 0).mean(), "p:d_q10": q[0], "p:d_q25": q[1], "p:d_q50": q[2], "p:d_q75": q[3], "p:d_q90": q[4],
        "p:d_absmean": np.abs(d).mean(),
        "p:dz_mean": (zg - zf).mean(), "p:dz_pos": (zg > zf).mean(),
        "p:dent_mean": (hg - hf).mean(),
        "p:drank_mean": (np.log1p(rg) - np.log1p(rf)).mean(),
        "p:rank_g_better": (rg < rf).mean(), "p:rank_f_better": (rf < rg).mean(),
        "p:top1_g_only": ((rg == 0) & (rf > 0)).mean(), "p:top1_f_only": ((rf == 0) & (rg > 0)).mean(),
        "p:corr": np.corrcoef(lf, lg)[0, 1] if n > 2 and lf.std() > 0 and lg.std() > 0 else 1.0,
        "p:d_lowent": d[lo].mean(), "p:d_highent": d[~lo].mean() if (~lo).any() else d.mean(),
    }


def tables(pairs, ids, n_chars, run, ll_model):
    base = {"flash": {}, "glm53": {}}
    pair = {}
    for i in ids:
        sf, sg = pairs[i]
        base["flash"][i] = features.one(sf[:CUT], n_chars[i])
        base["glm53"][i] = features.one(sg[:CUT], n_chars[i])
        pair[i] = paired(sf[:CUT], sg[:CUT])
    # Prüfung der Textlängen: Merkmale bei vollem Schnitt = Merkmale der Pipeline
    for tag, k in (("flash", 0), ("glm53", 1)):
        pipe = pd.concat([pd.read_csv(f, dtype={"id": str}) for f in sorted(run.glob(f"c*_{tag}_features.csv"))])
        pipe = pipe.set_index("id").loc[list(ids)[:300]]
        mine = pd.DataFrame({i: features.one(pairs[i][k], n_chars[i]) for i in pipe.index}).T[pipe.columns]
        err = np.abs(mine.to_numpy(float) - pipe.to_numpy(float)).max()
        assert np.allclose(mine.to_numpy(float), pipe.to_numpy(float), rtol=1e-5, atol=1e-6), (run.name, tag, err)
    tab = {t: pd.DataFrame(base[t]).T.loc[list(ids)] for t in base}
    Xb = matrix(tab, ll_model)
    P = pd.DataFrame(pair).T.loc[list(ids)]
    return Xb, P.to_numpy(np.float64), list(P.columns)


def n_chars_via_csv(df):
    buf = io.StringIO()
    df[["id", "text"]].to_csv(buf, index=False)
    buf.seek(0)
    q = pd.read_csv(buf)
    return dict(zip(q["id"].astype(str), q["text"].astype(str).str.len()))


def oof(X, y, folds, labels, Xp=None, yp=None):
    out = np.zeros((len(y), len(labels)))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        if Xp is None:
            clf = hgb().fit(X[tr], y[tr])
        else:
            clf = hgb().fit(np.vstack([X[tr], Xp]), np.concatenate([y[tr], yp]),
                            sample_weight=np.concatenate([np.ones(tr.sum()), np.full(len(yp), POOL_W)]))
        p = clf.predict_proba(X[va])
        out[va] = np.log(np.clip(p[:, [list(clf.classes_).index(c) for c in labels]], 1e-9, 1))
    return out


def main():
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    real["id"] = real.id.astype(str)
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    folds = get_folds(real)

    tp = load_pairs(TRAIN)
    assert set(tp) == set(real.id)
    Xb, Xpair, pcols = tables(tp, real.id.tolist(), n_chars_via_csv(real), TRAIN, ll_model)
    pp = load_pairs(POOL)
    lab = pd.read_csv(POOL / "labels.csv", dtype={"id": str}).set_index("id").model
    pids = [i for i in lab.index if i in pp]
    pool = pd.read_parquet(ROOT / "data/external/open_pool/ntl_open_pool.parquet", columns=["id", "text"])
    pool = pool[pool.id.astype(str).isin(set(pids))].astype({"id": str})
    Pb, Ppair, _ = tables(pp, pids, n_chars_via_csv(pool), POOL, ll_model)
    yp = lab.loc[pids].to_numpy().astype(str)
    print(f"train {len(y)} Texte, Pool {len(pids)} Texte ({pd.Series(yp).value_counts().min()}–"
          f"{pd.Series(yp).value_counts().max()} je Klasse), {Xb.shape[1]} + {len(pcols)} gepaarte Merkmale "
          f"(Längenprüfung gegen die Pipeline OK)", flush=True)

    Xbp, Pbp = np.hstack([Xb, Xpair]), np.hstack([Pb, Ppair])
    heads = {
        "54 Merkmale, train + Pool (V6-Rezept)": oof(Xb, y, folds, labels, Pb, yp),
        "54 + gepaart, train + Pool": oof(Xbp, y, folds, labels, Pbp, yp),
        "54 Merkmale, nur train": oof(Xb, y, folds, labels),
        "54 + gepaart, nur train": oof(Xbp, y, folds, labels),
        "nur gepaart + n_tok, train + Pool": oof(np.hstack([Xb[:, :1], Xpair]), y, folds, labels,
                                                 np.hstack([Pb[:, :1], Ppair]), yp),
    }
    with np.load(s9.OUT / "oof_cache.npz") as c:
        ev = c["ev"]
    families = res_mod.family_columns(labels)
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    grp = lambda p: np.where(np.isin(p, s9.GLM), p, "rest")

    def chain(ll, cover=0.20):
        post, _ = s9.glm_stage(zl, ll, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=cover)
        final, _ = apex.apply_arrays(zl, ev, labels, families, texts, L[post.argmax(1)], 0.10, False, 0.0)
        return final.astype(str)

    ref_name = "54 Merkmale, train + Pool (V6-Rezept)"
    ref = chain(heads[ref_name])
    base = balanced_accuracy_score(y, ref)
    idx = np.array([labels.index(c) for c in y])
    glm = np.isin(y, s9.GLM)
    rows = []
    for name, ll in heads.items():
        p = L[ll.argmax(1)]
        f = chain(ll)
        _, lo, hi, pg = s9.bootstrap(y, ref, f, real.prompt_id, n=1000)
        rows.append(dict(kopf=name, kopf_bal=balanced_accuracy_score(y, p), nll=-ll[np.arange(len(y)), idx].mean(),
                         kopf_glm3=balanced_accuracy_score(grp(y), grp(p)),
                         nll_glm=-ll[glm, idx[glm]].mean(),
                         kette=balanced_accuracy_score(y, f), d_kette=100 * (balanced_accuracy_score(y, f) - base),
                         d_lo=100 * lo, d_hi=100 * hi, p_gt0=pg, geaendert=int((f != ref).sum())))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"\nΔ in Pp. gegen '{ref_name}' (Kette, 20 %, Bootstrap 90 % über Prompts):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "results.csv", index=False)
    np.savez(OUT / "feats.npz", Xb=Xb, Xpair=Xpair, Pb=Pb, Ppair=Ppair, y=y, yp=yp,
             ids=np.array(real.id.tolist()), pids=np.array(pids), pcols=np.array(pcols))

    # Welche gepaarten Merkmale tragen? Permutation auf dem train+Pool-Kopf wäre teuer; Einzel-Korrelation mit GLM-Label
    gl = np.where(y == "glm-5.3", 1, np.where(y == "glm-5.3-flash", -1, 0))
    m = gl != 0
    imp = pd.Series({c: abs(np.corrcoef(Xpair[m, j], gl[m])[0, 1]) for j, c in enumerate(pcols)}).sort_values(ascending=False)
    print("\n|Korrelation| gepaarter Merkmale mit GLM-5.3 (+1) gegen Flash (−1), nur GLM-Texte in train:")
    print(imp.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
