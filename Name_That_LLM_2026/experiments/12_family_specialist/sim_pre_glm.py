"""Experiment 12: Spezialist vor der GLM-Stufe in den Stack einrechnen (Umsortieren innerhalb der Familie,
Familienmasse bleibt), danach GLM 20 % und Apex wie V6 (optional zusaetzlich Spezialist-Evidenz in Apex).

Aufruf: .venv/bin/python experiments/12_family_specialist/sim_pre_glm.py [llm_logits.parquet]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp, softmax
from sklearn.metrics import balanced_accuracy_score as bas

import simulate as s9
from src.data import load_train


def rerank(zl, spec, fams, lam):
    z = zl.copy()
    for cols in fams[:3]:
        mass = logsumexp(zl[:, cols], 1, keepdims=True)
        z[:, cols] = mass + log_softmax(zl[:, cols] + lam * np.log(spec[:, cols]), 1)
    return z


def main(llm_path=None):
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    fams = res_mod.family_columns(labels)
    tfidf = np.load(ROOT / "models/12_family_specialist/spec_tfidf_train_final.npy")
    specs = {"TF-IDF": tfidf}
    if llm_path:
        z = pd.read_parquet(llm_path).astype({"id": str}).set_index("id").loc[real.id.astype(str)]
        llm = np.ones_like(tfidf)
        for cols in fams[:3]:
            llm[:, cols] = softmax(z[[labels[i] for i in cols]].to_numpy(np.float64), 1)
        specs["LLM"] = llm
        specs["LLMxTF"] = llm * tfidf
    with np.load(s9.OUT / "oof_cache.npz") as c:
        ll_glm, ev = c["ll"], c["ev"]
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)

    def chain(z0, evid, s):
        post, _ = s9.glm_stage(z0, ll_glm, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=0.20)
        f, _ = apex.apply_arrays(z0, evid, labels, fams, texts, L[post.argmax(1)], s, False, 0.0)
        return f.astype(str)

    ref = chain(zl, ev, 0.10)
    print(f"V6-Kette (Ersatz-Stack): {bas(y, ref):.4f}")
    rows, finals = [], {}
    for name, sp in specs.items():
        for lam in (0.1, 0.2, 0.3, 0.5, 0.75):
            z2 = rerank(zl, sp, fams, lam)
            for apex_mode in ("V6", "spec"):
                evid, s = (ev, 0.10) if apex_mode == "V6" else (ev * sp, 0.25)
                f = chain(z2, evid, s)
                finals[(name, lam, apex_mode)] = f
                rows.append(dict(spec=name, lam=lam, apex=apex_mode, d=100 * (bas(y, f) - bas(y, ref)),
                                 changed=int((f != ref).sum())))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["spec", "lam"], columns="apex", values="d").round(2).to_string())
    for _, r in df.sort_values("d", ascending=False).head(4).iterrows():
        m, lo, hi, pp = s9.bootstrap(y, ref, finals[(r.spec, r.lam, r.apex)], real.prompt_id, n=1000)
        print(f"  {r.spec} lam {r.lam} apex {r.apex}: {100 * m:+.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}] P>0 {pp:.2f} ({r.changed} anders)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
