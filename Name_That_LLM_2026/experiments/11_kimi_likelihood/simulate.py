"""Experiment 11: Kimi-K3-Likelihood als zweite Stufe hinter der V6-Kette, lokal auf train.csv.

Kette wie V6: Stack-Ersatz -> GLM-Stufe (Top x % nach GLM-Masse) -> Apex (Residual + harte Typografie, GLM-Labels
bleiben). Danach neu die Kimi-Stufe, NACH Apex (sonst würde Apex sie überschreiben):
- Auswahl: die Top y % der Texte nach Kimi-Masse des Stacks.
- Auf diesen Texten, sofern sie nicht GLM-gelabelt sind: argmax(refined + W/T' * ll_kimi), mit den
  Typografie-Regeln von Apex als Maske (verbotene Labels kommen nicht in Frage).
- Klassifikator: K1 HGB auf den Kimi-Merkmalen, K2 HGB auf Kimi + GLM-Merkmalen, je 18 Klassen; „gegen den Rest“-
  Fassung (p(Kimi), Rest gleichmäßig auf 17 Klassen): Die Stufe entscheidet dann nur Kimi ja/nein, die Wahl unter
  den übrigen Klassen bleibt bei Apex (Lehre aus V9/V13: der Klassifikator trennt Nicht-Zielklassen schlecht).
Nur train-Texte mit Kimi-Merkmalen gehen ein (Kimi-OOF über die GroupKFold-Folds); die GLM-Kette läuft auf allen.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python experiments/11_kimi_likelihood/simulate.py --feats models/11_kimi_likelihood/kimi_train_feats_L160.parquet
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import simulate as s9
from src.cv import get_folds
from src.data import load_train

OUT = ROOT / "models" / "11_kimi_likelihood"
KIMI = "kimi-k3"
HGB = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=20,
           early_stopping=False)


def oof(X, y, folds, labels, rows):
    """OOF-Log-Wahrscheinlichkeiten (n, 18) auf den Zeilen `rows` (Rest NaN)."""
    out = np.full((len(y), len(labels)), np.nan)
    for k in np.unique(folds[rows]):
        tr = rows & (folds != k)
        va = rows & (folds == k)
        clf = HistGradientBoostingClassifier(random_state=0, **HGB).fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])
        out[va] = np.log(np.clip(p[:, [list(clf.classes_).index(c) for c in labels]], 1e-9, 1))
    return out


def vs_rest(ll, labels):
    k = labels.index(KIMI)
    p = np.exp(ll)
    q = np.repeat((1 - p[:, [k]]) / (len(labels) - 1), len(labels), 1)
    q[:, k] = p[:, k]
    return np.log(np.clip(q, 1e-9, None))


def typo_mask(texts, labels):
    allowed = {"claude-opus-4-6", "claude-opus-5", "glm-5.3", "glm-5.3-flash"}
    rules = [(" — ", np.array([x not in allowed for x in labels])),
             ("’", np.array([x.startswith(("claude-", "glm-")) for x in labels])),
             ("—", np.array([x.startswith("gemini-") for x in labels]))]
    m = np.zeros((len(texts), len(labels)), bool)
    for i, t in enumerate(texts):
        for marker, forb in rules:
            if marker in t:
                m[i] |= forb
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    a = ap.parse_args()
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    folds = get_folds(real)

    kf = pd.read_parquet(a.feats).astype({"id": str}).set_index("id")
    fcols = [c for c in kf.columns if c != "model"]
    have = real.id.astype(str).isin(kf.index).to_numpy()
    XK = np.full((len(real), len(fcols)), np.nan)
    XK[have] = kf.loc[real.id.astype(str)[have], fcols].to_numpy(np.float64)
    with np.load(s9.NPZ, allow_pickle=False) as d:
        XG = d["X"][:len(real)].astype(np.float64)
    with np.load(s9.OUT / "oof_cache.npz") as c:
        ll_glm, ev = c["ll"], c["ev"]
    families = res_mod.family_columns(labels)
    print(f"Kimi-Merkmale für {have.sum()} von {len(real)} train-Texten ({(y[have] == KIMI).sum()} Kimi)")
    yk = y == KIMI
    aucs = sorted(((c, roc_auc_score(yk[have], XK[have, j])) for j, c in enumerate(fcols)),
                  key=lambda r: -abs(r[1] - .5))
    print("AUC Kimi gegen den Rest:", ", ".join(f"{c} {v:.3f}" for c, v in aucs[:6]))

    OUT.mkdir(parents=True, exist_ok=True)
    members = {"K1 Kimi": oof(XK, y, folds, labels, have), "K2 Kimi+GLM": oof(np.hstack([XK, XG]), y, folds, labels, have)}
    for n, ll in list(members.items()):
        members[n + " gegen Rest"] = vs_rest(ll, labels)
    for n, ll in members.items():
        p = ll[have].argmax(1)
        print(f"  {n:<22} allein: bal. acc {balanced_accuracy_score(y[have], L[p]):.4f}, "
              f"AUC Kimi {roc_auc_score(yk[have], ll[have, labels.index(KIMI)]):.3f}")

    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    tp = v1.match_temperature(zl, ll_model["entropy_ref"])
    mask = typo_mask(texts, labels)
    ki = labels.index(KIMI)

    def v6(cover_glm):
        post, _ = s9.glm_stage(zl, ll_glm, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=cover_glm)
        final, refined = apex.apply_arrays(zl, ev, labels, families, texts, L[post.argmax(1)], 0.10, False, 0.0)
        return final.astype(str), np.asarray(refined, np.float64)

    def kimi_stage(final, refined, ll, W, cover_k):
        sel = np.argsort(-zl[:, ki], kind="stable")[:int(cover_k * len(zl))]
        sel = sel[have[sel] & ~np.isin(final[sel], s9.GLM)]
        z = refined[sel] + W / tp * ll[sel]
        z[mask[sel]] = -np.inf
        out = final.copy()
        out[sel] = L[z.argmax(1)]
        return out

    rows = []
    sub = have   # nur Texte mit Kimi-Merkmalen bewerten (sonst verdünnt der Rest den Effekt)
    for cover_glm in (0.20, 0.15, 0.10):
        base, refined = v6(cover_glm)
        rows.append(dict(glm=cover_glm, kimi=0, member="-", W=0, bal_acc=balanced_accuracy_score(y[sub], base[sub]),
                         rec_kimi=(base[yk & sub] == KIMI).mean(), fp_kimi=(base[~yk & sub] == KIMI).mean(), changed=0))
        for n, ll in members.items():
            for W in (0.5, 1.0, 1.5):
                for cover_k in (0.05, 0.08, 0.12, 0.20):
                    f = kimi_stage(base, refined, ll, W, cover_k)
                    rows.append(dict(glm=cover_glm, kimi=cover_k, member=n, W=W,
                                     bal_acc=balanced_accuracy_score(y[sub], f[sub]),
                                     rec_kimi=(f[yk & sub] == KIMI).mean(), fp_kimi=(f[~yk & sub] == KIMI).mean(),
                                     changed=int((f != base).sum())))
    df = pd.DataFrame(rows)
    ref = df[(df.glm == 0.20) & (df.kimi == 0)].bal_acc.iloc[0]
    df["d_v6"] = 100 * (df.bal_acc - ref)
    df.to_csv(OUT / "results.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[df.kimi == 0].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(df.sort_values("d_v6", ascending=False).head(25).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
