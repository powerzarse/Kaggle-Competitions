"""Experiment 14: GLM-Stufe mit längerem Schnitt (256 statt 160 Tokens), lokal auf train.csv.

Daten: Token-Scores aller 2.699 train-Texte unter GLM-5.3-Flash und GLM-5.3, erste 512 Tokens
(`kaggle/build_notebook_glm_scores.py --source train` → `models/14_glm_256/train512/out/`). Merkmale für jeden
kürzeren Schnitt L: `features.one(s[:L], len(text))` (kausales Modell, exakt).

1. Prüfungen: Merkmale bei 512 = Merkmale der Pipeline; Merkmale bei 160 gegen V6s train-Merkmale (estopadilla-Lauf).
2. Köpfe OOF (GroupKFold nach prompt_id wie src/cv.py): LR (estopadillas Rezept) bei 160/256/320, HGB nur train
   bei 160/256, V6 (HGB-160 train + Pool, aus Experiment 09) und Log-Mischungen V6 × LR-256.
3. V6-Kette (Ersatz-Stack → GLM-Stufe → Apex, wie Experiment 10): Abdeckung 20 % für alle und „gleiche Zeit“
   (256er-Köpfe bekommen nur so viele Texte, wie in die Token-Zeilen von 20 % bei 160 passen).
   Aussagekräftig ist nur die GLM-Entscheidung (Experiment 09/12: Nicht-GLM-Simulationen vom LB widerlegt).

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python experiments/14_glm_256/evaluate.py
"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))
sys.path.insert(0, str(ROOT / "experiments" / "08_glm_likelihood" / "v1src"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features
import simulate as s9
from src.cv import get_folds
from src.data import load_train

RUN = ROOT / "models" / "14_glm_256" / "train512" / "out"
OUT = ROOT / "models" / "14_glm_256"
HEAD256 = ROOT / "experiments" / "14_glm_256" / "ll_model_L256_public.json"
CUTS = (160, 256, 320, 512)


def load_scores(tag):
    scores = {}
    for f in sorted(RUN.glob(f"c*_{tag}_scores.npz")):
        with np.load(f, allow_pickle=True) as d:
            ids, off, sc = d["ids"].astype(str), d["offsets"], d["scores"]
        for k, i in enumerate(ids):
            scores[i] = sc[off[k]:off[k + 1]]
    return scores


def feature_tables(real):
    # Textlängen wie im Lauf: train.csv -> CSV -> read_csv -> astype(str) (run_pipeline.py)
    buf = io.StringIO()
    real[["id", "text"]].to_csv(buf, index=False)
    buf.seek(0)
    q = pd.read_csv(buf)
    n_chars = dict(zip(q["id"].astype(str), q["text"].astype(str).str.len()))
    tables = {L: {} for L in CUTS}
    for tag in ("flash", "glm53"):
        sc = load_scores(tag)
        assert len(sc) == len(real), (tag, len(sc))
        for L in CUTS:
            tables[L][tag] = pd.DataFrame({i: features.one(sc[i][:L], n_chars[i]) for i in real.id}).T
        pipe = pd.concat([pd.read_csv(f, dtype={"id": str}) for f in sorted(RUN.glob(f"c*_{tag}_features.csv"))]).set_index("id")
        mine = tables[512][tag].loc[pipe.index, pipe.columns].astype(float)
        err = np.abs(mine.to_numpy() - pipe.to_numpy()).max()
        assert np.allclose(mine.to_numpy(), pipe.to_numpy(), rtol=1e-5, atol=1e-6), (tag, err)
        print(f"{tag}: Merkmale bei 512 = Pipeline (max. Abweichung {err:.1e}); "
              f"Texte > 160 Tokens {(tables[512][tag].n_tok > 160).mean():.0%}, > 256 {(tables[512][tag].n_tok > 256).mean():.0%}")
    return tables


def matrix(tab, model):
    G = pd.concat([tab[t].add_prefix(f"{t}:") for t in ("flash", "glm53")], axis=1)
    for c in model["diff_features"]:
        G[f"d:{c}"] = G[f"flash:{c}"] - G[f"glm53:{c}"]
    return G[model["features"]].to_numpy(np.float64)


def oof(make, X, y, folds, labels):
    out = np.zeros((len(y), len(labels)))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        clf = make().fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])
        out[va] = np.log(np.clip(p[:, [list(clf.classes_).index(c) for c in labels]], 1e-9, 1))
    return out


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))


def hgb():
    return HistGradientBoostingClassifier(random_state=0, **s9.HGB)


def main():
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    real["id"] = real.id.astype(str)
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    folds = get_folds(real)
    pub256 = json.loads(HEAD256.read_text())
    assert pub256["features"] == ll_model["features"] and pub256["classes"] == labels

    tables = feature_tables(real)
    X = {Lc: matrix(tables[Lc], ll_model) for Lc in CUTS}
    with np.load(s9.NPZ, allow_pickle=False) as d:
        X_old = d["X"][:len(real)].astype(np.float64)
        assert (d["y"][:len(real)].astype(str) == y).all()
    corr = [np.corrcoef(X[160][:, j], X_old[:, j])[0, 1] for j in range(X_old.shape[1])]
    print(f"160er-Merkmale (neuer Lauf) gegen V6-train-Merkmale: Korrelation je Merkmal min {min(corr):.4f}, "
          f"Median {np.median(corr):.4f}; n_tok gleich {np.mean(X[160][:, 0] == X_old[:, 0]):.1%}")

    # Reproduktion des öffentlichen 256er-Kopfs (auf ganz train angepasst)
    full = lr().fit(X[256], y)
    coef = full[-1].coef_[[list(full[-1].classes_).index(c) for c in labels]]
    print(f"LR-256 auf unseren Merkmalen gegen den öffentlichen Kopf: Korrelation der Koeffizienten "
          f"{np.corrcoef(coef.ravel(), np.asarray(pub256['coef']).ravel())[0, 1]:.3f}\n")

    with np.load(s9.OUT / "oof_cache.npz") as c:   # Experiment 09: V6-Klassifikator OOF + Residual-Evidenz
        ll_v6, ev = c["ll"], c["ev"]
    families = res_mod.family_columns(labels)

    heads = {"V6: HGB-160 train+Pool": (ll_v6, 160)}
    for Lc in (160, 256, 320):
        heads[f"LR-{Lc} (estopadilla)"] = (oof(lr, X[Lc], y, folds, labels), Lc)
    for Lc in (160, 256):
        heads[f"HGB-{Lc} nur train"] = (oof(hgb, X[Lc], y, folds, labels), Lc)
    for a in (0.3, 0.5, 0.7):
        heads[f"Mix {a} V6 + {1 - a:.1f} LR-256"] = (log_softmax(a * ll_v6 + (1 - a) * heads["LR-256 (estopadilla)"][0], 1), 256)
    heads["Mix 0.5 V6 + 0.5 HGB-256"] = (log_softmax(0.5 * ll_v6 + 0.5 * heads["HGB-256 nur train"][0], 1), 256)

    print("Köpfe allein (OOF): Genauigkeit 18 Klassen, bal. acc., NLL, bal. acc. GLM-5.3 / Flash / Rest")
    idx = np.array([labels.index(c) for c in y])
    grp = lambda p: np.where(np.isin(p, s9.GLM), p, "rest")
    for name, (ll, _) in heads.items():
        p = L[ll.argmax(1)]
        print(f"  {name:<30} acc {accuracy_score(y, p):.3f}  bal {balanced_accuracy_score(y, p):.3f}  "
              f"NLL {-ll[np.arange(len(y)), idx].mean():.3f}  GLM3 {balanced_accuracy_score(grp(y), grp(p)):.3f}")

    # Kette
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    mass = np.exp(zl[:, [labels.index(c) for c in s9.GLM]]).sum(1)
    order = np.argsort(-mass, kind="stable")
    rows_at = lambda Lc: np.minimum(real.text.str.len().to_numpy() * ll_model.get("tok_per_char", 0.235) * 1.05, Lc) + 13
    budget = np.cumsum(rows_at(160)[order])[int(0.20 * len(zl)) - 1]
    cover_eq = {Lc: (np.cumsum(rows_at(Lc)[order]) <= budget).sum() / len(zl) for Lc in (160, 256, 320)}
    print(f"\nAbdeckung bei gleicher Zeit (Token-Zeilen von 20 % bei 160): "
          + ", ".join(f"{Lc}: {c:.1%}" for Lc, c in cover_eq.items()))

    def chain(ll, cover, W=1.0):
        post, _ = s9.glm_stage(zl, ll, labels, ll_model["entropy_ref"], v1.match_temperature, W=W, cover=cover)
        final, _ = apex.apply_arrays(zl, ev, labels, families, texts, L[post.argmax(1)], 0.10, False, 0.0)
        return final.astype(str)

    ref = chain(ll_v6, 0.20)
    base = balanced_accuracy_score(y, ref)
    print(f"\nV6-Kette (Ersatz-Stack OOF {balanced_accuracy_score(y, L[zl.argmax(1)]):.4f}), V6 bei 20 %: {base:.4f}")
    rows = []
    for name, (ll, Lc) in heads.items():
        for label, cover in (("20 %", 0.20), ("gleiche Zeit", cover_eq[Lc])):
            final = chain(ll, cover)
            m, lo, hi, pp = s9.bootstrap(y, ref, final, real.prompt_id, n=1000)
            rows.append(dict(kopf=name, abdeckung=label, cover=cover, bal_acc=balanced_accuracy_score(y, final),
                             d_pp=100 * (balanced_accuracy_score(y, final) - base), lo=100 * lo, hi=100 * hi, p_gt0=pp,
                             glm3=balanced_accuracy_score(grp(y), grp(final)),
                             rec_glm53=(final[y == "glm-5.3"] == "glm-5.3").mean(),
                             rec_flash=(final[y == "glm-5.3-flash"] == "glm-5.3-flash").mean(),
                             n_glm=int(np.isin(final, s9.GLM).sum()), diff_v6=int((final != ref).sum())))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "results.csv", index=False)
    np.savez(OUT / "train_feats.npz", **{f"X{Lc}": X[Lc] for Lc in CUTS}, y=y, ids=real.id.to_numpy().astype(str))


if __name__ == "__main__":
    main()
