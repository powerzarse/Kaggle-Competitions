"""V1 GLM-5.3 / GLM-5.3-Flash likelihood stage with PB-115 token-aligned LR+LGB and hybrid calibration."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax

PREFIX = '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|assistant|><think></think>'


def entropy(z):
    return float(-(np.exp(z) * z).sum(1).mean())


def match_temperature(z, h_ref):
    """T' with mean entropy of log_softmax(T' z) equal to h_ref (entropy decreases with T')."""
    lo, hi = 1e-3, 1e3
    if entropy(log_softmax(z * hi, 1)) > h_ref or entropy(log_softmax(z * lo, 1)) < h_ref:
        raise ValueError('entropy_ref out of reach')
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if entropy(log_softmax(z * mid, 1)) > h_ref:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def build_54_matrix(feats, ids, features_54, diff_features):
    blocks = [feats[t].add_prefix(f'{t}:') for t in ('flash', 'glm53')]
    F = pd.concat(blocks, axis=1).reindex(ids)
    for c in diff_features:
        F[f'd:{c}'] = F[f'flash:{c}'] - F[f'glm53:{c}']
    X = F[features_54].to_numpy(np.float64)
    assert np.isfinite(X).all(), 'non-finite 54-dim likelihood features'
    return X


def member_logp(model, feats, ids):
    """feats: {'flash': DataFrame, 'glm53': DataFrame} indexed by id -> (n, 18) log-probs, model['classes'] order."""
    X = build_54_matrix(feats, ids, model['features'], model['diff_features'])
    z = ((X - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coef']).T + np.asarray(model['intercept'])
    return log_softmax(z, 1)


def marginal_logodds(ll_18, labels):
    """Compute 3-way GLM vs non-GLM log-odds update (zero perturbation among the 16 non-GLM classes)."""
    g53 = labels.index('glm-5.3')
    gf = labels.index('glm-5.3-flash')
    non_glm = np.ones(len(labels), dtype=bool)
    non_glm[[g53, gf]] = False
    p18 = np.clip(np.exp(ll_18), 1e-8, 1.0)
    p_other = np.clip(p18[:, non_glm].sum(1), 1e-8, 1.0)
    non_prior = np.log(1.0 / float(non_glm.sum()))
    out = np.zeros_like(ll_18)
    out[:, g53] = np.log(p18[:, g53]) - (np.log(p_other) + non_prior)
    out[:, gf] = np.log(p18[:, gf]) - (np.log(p_other) + non_prior)
    return out


def marginal_logodds_from_p3(p3, labels):
    """Convert direct 3-class [other, glm-5.3, glm-5.3-flash] probabilities into 18-class marginal log-odds."""
    g53 = labels.index('glm-5.3')
    gf = labels.index('glm-5.3-flash')
    p3 = np.clip(np.asarray(p3, dtype=np.float64), 1e-8, 1.0)
    non_prior = np.log(1.0 / float(len(labels) - 2))
    out = np.zeros((len(p3), len(labels)), dtype=np.float64)
    out[:, g53] = np.log(p3[:, 1]) - (np.log(p3[:, 0]) + non_prior)
    out[:, gf] = np.log(p3[:, 2]) - (np.log(p3[:, 0]) + non_prior)
    return out


def fit_and_predict_pb115(pb115_dir, train_csv, feats_te, sel_ids, flash_npz, glm53_npz, labels):
    """Fit PB-115 (54 + 61 token-aligned) 55% LR + 45% 3-seed LGB18 and 50% LR3 + 50% 3-seed LGB3."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import v1_token_features as vtf

    pb_dir = Path(pb115_dir)
    spec = json.loads((pb_dir / 'spec.json').read_text())
    assert spec['format'] == 'ntl-prefixb-115-v1'
    tr = pd.read_csv(train_csv, dtype={'id': str})
    tr_ids = tr['id'].tolist()
    l2i = {l: i for i, l in enumerate(labels)}
    y = tr['model'].map(l2i).to_numpy(dtype=np.int64)
    g53, gf = l2i['glm-5.3'], l2i['glm-5.3-flash']
    y3 = np.where(y == g53, 1, np.where(y == gf, 2, 0))

    tr_feats = {
        t: pd.read_csv(pb_dir / f'{t}_features.csv', dtype={'id': str}).set_index('id').loc[tr_ids]
        for t in ('flash', 'glm53')
    }
    X_tr_54 = build_54_matrix(tr_feats, tr_ids, spec['features_54'], spec['diff_features'])
    p_tr_61 = pd.read_parquet(pb_dir / 'paired_token_features.parquet').loc[tr_ids][spec['features_61']].to_numpy(np.float64)
    X_tr = np.hstack([X_tr_54, p_tr_61])
    assert np.isfinite(X_tr).all() and X_tr.shape == (len(tr_ids), 115)

    X_te_54 = build_54_matrix(feats_te, sel_ids, spec['features_54'], spec['diff_features'])
    p_te_df = vtf.extract_paired_score_features(flash_npz, glm53_npz)
    p_te_61 = p_te_df.loc[sel_ids][spec['features_61']].to_numpy(np.float64)
    X_te = np.hstack([X_te_54, p_te_61])
    assert np.isfinite(X_te).all() and X_te.shape == (len(sel_ids), 115)

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_te_s = sc.transform(X_te)

    lr18 = LogisticRegression(C=0.025, max_iter=1000).fit(X_tr_s, y)
    p18_lr = lr18.predict_proba(X_te_s)
    p18_lgb_list = []
    for sd in (42, 1042, 2042):
        clf = lgb.LGBMClassifier(
            objective='multiclass', num_class=len(labels), n_estimators=220, learning_rate=0.03,
            num_leaves=15, min_child_samples=25, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=5.0, random_state=sd, verbose=-1, n_jobs=4,
        ).fit(X_tr_s, y)
        p18_lgb_list.append(clf.predict_proba(X_te_s))
    p18 = np.clip(0.55 * p18_lr + 0.45 * np.mean(p18_lgb_list, axis=0), 1e-8, 1.0)
    ll_18 = np.log(p18)

    # Derive 3-way marginal log-odds directly from calibrated 18-class model (avoids 16:1 class imbalance of 3-class head)
    ll_m3 = marginal_logodds(ll_18, labels)
    assert np.isfinite(ll_18).all() and np.isfinite(ll_m3).all()
    return ll_18, ll_m3


def compute_hybrid_delta(zl, ll_18, labels, W, h_ref, hybrid_alpha=0.20, marg_w=1.2, marg_href=0.90, ll_m3_override=None):
    """Combine calibrated 18-class GLM logp with zero-non-GLM-noise marginal GLM log-odds."""
    tp18 = match_temperature(zl, h_ref)
    w18 = W / tp18
    if hybrid_alpha <= 0.0:
        return w18 * ll_18, {'tp18': tp18, 'w18': w18, 'tp3': None, 'w3': 0.0, 'hybrid_alpha': 0.0}
    tp3 = match_temperature(zl, marg_href)
    w3 = marg_w / tp3
    ll_m3 = ll_m3_override if ll_m3_override is not None else marginal_logodds(ll_18, labels)
    delta = (1.0 - hybrid_alpha) * w18 * ll_18 + hybrid_alpha * w3 * ll_m3
    return delta, {'tp18': tp18, 'w18': w18, 'tp3': tp3, 'w3': w3, 'hybrid_alpha': hybrid_alpha}


def main():
    ap = argparse.ArgumentParser()
    for k in ('--test-csv', '--final-logits', '--out', '--report', '--bin', '--pipeline', '--flash-segs', '--glm53-segs',
              '--model', '--work', '--outdir'):
        ap.add_argument(k, required=True)
    ap.add_argument('--deadline-mono', type=float, required=True, help='time.monotonic() by which all must be done')
    ap.add_argument('--post-logits', default='', help='optional path to save post-GLM log-probabilities')
    ap.add_argument('--pb115-dir', default='', help='optional dir with full 2699 Prefix-B 115-dim train features')
    ap.add_argument('--train-csv', default='', help='path to competition train.csv (required with --pb115-dir)')
    ap.add_argument('--hybrid-alpha', type=float, default=0.20, help='share of 3-way marginal GLM log-odds in blend')
    ap.add_argument('--w-override', type=float, default=0.0, help='if >0, overrides model["W"] for 18-class term')
    ap.add_argument('--marg-w', type=float, default=1.2, help='weight for 3-way marginal GLM log-odds term')
    ap.add_argument('--marg-href', type=float, default=0.90, help='reference entropy for 3-way marginal term')
    ap.add_argument('--pack', action='store_true')
    ap.add_argument('--ubatch-div', type=int, default=1)
    ap.add_argument('--seq-batch', type=int, default=8)
    ap.add_argument('--max-text-tokens', type=int, default=0)
    ap.add_argument('--fixed-s-flash', type=float, default=870.0)
    ap.add_argument('--row-s-flash', type=float, default=0.0141)
    ap.add_argument('--fixed-s-glm53', type=float, default=1140.0)
    ap.add_argument('--row-s-glm53', type=float, default=0.0231)
    ap.add_argument('--select-classes', default='glm-5.3,glm-5.3-flash')
    ap.add_argument('--max-cover', type=float, default=0.30)
    ap.add_argument('--min-texts', type=int, default=50)
    ap.add_argument('--reserve-s', type=float, default=180.0)
    ap.add_argument('--prefetch', action='store_true')
    ap.add_argument('--priority-order', action='store_true')
    ap.add_argument('--shrink', action='store_true')
    ap.add_argument('--max-changed', type=float, default=0.25)
    a = ap.parse_args()
    assert a.priority_order or not a.shrink, '--shrink keeps a prefix of the input: it needs --priority-order'

    t0 = time.monotonic()
    rep = {'status': 'started', 'deadline_left_s': a.deadline_mono - t0}
    out_report = Path(a.report)

    def save(**kw):
        rep.update(kw)
        out_report.write_text(json.dumps(rep, indent=1))
        print('V1', json.dumps({k: v for k, v in kw.items() if not isinstance(v, dict)}), flush=True)

    def left():
        return a.deadline_mono - time.monotonic() - a.reserve_s

    model = json.loads(Path(a.model).read_text())
    assert model['format'] == 'ntl-v1-ll-model-1'
    a.max_text_tokens = a.max_text_tokens or model['max_text_tokens']
    assert a.max_text_tokens == model['max_text_tokens'], 'the test must be cut like the train features'
    frame = pd.read_csv(a.test_csv, dtype={'id': str})
    frame['text'] = frame.text.fillna('').astype(str)
    ids = frame.id.tolist()
    with np.load(a.final_logits, allow_pickle=False) as f:
        assert f['ids'].astype(str).tolist() == ids, 'final-logits ids differ from the test order'
        labels = f['labels'].astype(str).tolist()
        z = f['logits'].astype(np.float64)
    assert sorted(labels) == sorted(model['classes']) and z.shape == (len(ids), 18) and np.isfinite(z).all()

    zl = log_softmax(z, 1)
    sel_cls = a.select_classes.split(',')
    assert all(c in labels for c in sel_cls), sel_cls
    mass = np.exp(zl[:, [labels.index(c) for c in sel_cls]]).sum(1)
    order = np.argsort(-mass, kind='stable')

    tpc = model.get('tok_per_char', 0.235)
    prefix = model.get('prefix', PREFIX)
    user_template = model.get('user_template', '')
    assert ('{user}' in prefix) == bool(user_template), 'prefix and user_template go together'
    rows_i = np.minimum(frame.text.str.len().to_numpy() * tpc * 1.05, a.max_text_tokens) + model.get('prefix_rows', 13)
    cum = np.cumsum(rows_i[order])
    est = lambda r: (a.fixed_s_flash + a.row_s_flash * r, a.fixed_s_glm53 + a.row_s_glm53 * r)
    fits = 1.1 * (a.fixed_s_flash + a.fixed_s_glm53 + (a.row_s_flash + a.row_s_glm53) * cum) <= left()
    n_sel = int(min(fits.sum(), int(a.max_cover * len(ids))))
    rows = int(cum[n_sel - 1]) if n_sel else 0
    est_f, est_g = est(rows)
    save(status='estimated', n_texts=len(ids), n_selected=n_sel, cover=n_sel / len(ids),
         mass_threshold=float(mass[order[n_sel - 1]]) if n_sel else None, est_rows=rows,
         est_rows_all=int(cum[-1]), est_flash_s=est_f, est_glm53_s=est_g, left_s=left())
    if n_sel < a.min_texts:
        save(status='fallback_s10', reason=f'only {n_sel} texts fit in the time left')
        return
    pick = (lambda n: order[:n]) if a.priority_order else (lambda n: np.sort(order[:n]))
    sel = pick(n_sel)
    sel_ids = [ids[i] for i in sel]

    Path(a.work).mkdir(parents=True, exist_ok=True)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    times = {}
    reserve_flash = 1.05 * (a.fixed_s_glm53 + a.row_s_glm53 * cum[min(a.min_texts, n_sel) - 1])
    for tag, segs, reserve in (('flash', a.flash_segs, reserve_flash), ('glm53', a.glm53_segs, 0.0)):
        if tag == 'glm53':
            m = int(min(n_sel, (1.1 * (a.fixed_s_glm53 + a.row_s_glm53 * cum[:n_sel]) <= left()).sum()))
            if m < a.min_texts:
                save(status='fallback_s10', reason=f'after Flash only {m} texts fit for GLM-5.3', times=times)
                return
            if m < n_sel:
                save(status='glm53_shrunk', n_selected_flash=n_sel, n_selected=m, cover=m / len(ids),
                     mass_threshold=float(mass[order[m - 1]]), left_s=left())
                n_sel = m
                sel = pick(n_sel)
                sel_ids = [ids[i] for i in sel]
        budget = left() - reserve
        if budget <= 0:
            save(status='fallback_s10', reason=f'no time left for {tag}')
            return
        query = Path(a.work) / f'v1_query_{tag}.csv'
        frame.iloc[sel][['id', 'text']].to_csv(query, index=False)
        cmd = [sys.executable, a.pipeline, '--bin', a.bin, '--segs', segs, '--csv', str(query), '--out', a.outdir,
               '--tag', tag, '--work', str(Path(a.work) / 'state'), '--seq-batch', str(a.seq_batch),
               '--state-dtype', 'auto', '--max-text-tokens', str(a.max_text_tokens), '--prefix', prefix,
               '--budget-s', f'{budget:.0f}']
        if user_template:
            cmd += ['--user-template', user_template]
        if a.shrink:
            cmd += ['--shrink-min', str(a.min_texts)]
        if a.pack:
            cmd.append('--pack')
        if a.ubatch_div > 1:
            cmd += ['--ubatch-div', str(a.ubatch_div)]
        if a.prefetch:
            cmd.append('--prefetch')
            if tag == 'flash':
                cmd += ['--prefetch-after', sorted(glob.glob(a.glm53_segs))[0]]
        t = time.monotonic()
        with open(Path(a.outdir) / f'{tag}_pipeline.txt', 'w') as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        times[tag] = time.monotonic() - t
        if code == 3:
            save(status='fallback_s10', reason=f'{tag} time budget exceeded', times=times)
            return
        if code != 0:
            raise RuntimeError(f'V1 {tag} pipeline failed with exit code {code}; see {tag}_pipeline.txt')
        if a.shrink:
            done = set(pd.read_csv(Path(a.outdir) / f'{tag}_features.csv', dtype={'id': str}).id)
            n_done = len(done)
            assert n_done >= a.min_texts and set(sel_ids[:n_done]) == done, f'{tag}: kept texts are not the top'
            if n_done < n_sel:
                shrinks = json.loads((Path(a.outdir) / f'{tag}_timing.json').read_text()).get('shrinks', [])
                save(**{f'{tag}_shrunk': {'from': n_sel, 'to': n_done, 'shrinks': shrinks}},
                     n_selected=n_done, cover=n_done / len(ids), mass_threshold=float(mass[order[n_done - 1]]))
                n_sel = n_done
                sel = pick(n_sel)
                sel_ids = [ids[i] for i in sel]
        save(status=f'{tag}_done', times=times, left_s=left())

    feats = {t: pd.read_csv(Path(a.outdir) / f'{t}_features.csv', dtype={'id': str}).set_index('id')
             for t in ('flash', 'glm53')}
    for t, f in feats.items():
        assert set(sel_ids) <= set(f.index), f'{t}: features missing for some texts'
        feats[t] = f.loc[sel_ids]

    ll_m3_override = None
    member_mode = 'll_model_54'
    if a.pb115_dir and a.train_csv:
        try:
            t_pb = time.monotonic()
            ll, ll_m3_override = fit_and_predict_pb115(
                a.pb115_dir, a.train_csv, feats, sel_ids,
                Path(a.outdir) / 'flash_scores.npz', Path(a.outdir) / 'glm53_scores.npz', labels,
            )
            member_mode = f'pb115_lr_lgb_direct3 ({time.monotonic() - t_pb:.1f}s)'
        except Exception as _e_pb:
            print(f'PB115 head failed, falling back to 54-dim ll_model.json: {_e_pb!r}', flush=True)
            ll = member_logp(model, feats, sel_ids)[:, [model['classes'].index(c) for c in labels]]
    else:
        ll = member_logp(model, feats, sel_ids)[:, [model['classes'].index(c) for c in labels]]

    eff_w = a.w_override if a.w_override > 0 else model['W']
    delta, cal_info = compute_hybrid_delta(
        zl, ll, labels, eff_w, model['entropy_ref'],
        hybrid_alpha=a.hybrid_alpha, marg_w=a.marg_w, marg_href=a.marg_href,
        ll_m3_override=ll_m3_override,
    )
    zl_post = zl.copy()
    zl_post[sel] = zl[sel] + delta
    assert np.isfinite(zl_post).all()

    before = zl.argmax(1)
    after = zl_post.argmax(1)
    changed = float((before != after).mean())
    changed_sel = float((before[sel] != after[sel]).mean())
    agrees = float((ll.argmax(1) == before[sel]).mean())
    if changed > a.max_changed:
        save(status='error_sanity', changed=changed, changed_selected=changed_sel,
             temperature=cal_info['tp18'], weight=cal_info['w18'], member_argmax_agrees_s10=agrees)
        raise RuntimeError(f'V1 blend changes {changed:.1%} of the predictions (> {a.max_changed:.0%}): not submitted')

    if a.post_logits:
        _sel_mask = np.zeros(len(ids), dtype=bool)
        _sel_mask[sel] = True
        np.savez_compressed(a.post_logits, ids=np.asarray(ids, dtype=str),
                            labels=np.asarray(labels, dtype=str), logits=zl_post.astype(np.float64),
                            sel_mask=_sel_mask)

    sub = pd.DataFrame({'id': ids, 'model': np.asarray(labels)[after]})
    assert len(sub) == len(frame) and sub.id.is_unique and sub.model.isin(labels).all()
    sub.to_parquet(a.out, index=False)
    assert pd.read_parquet(a.out).equals(sub)
    count = lambda p: {labels[i]: int(n) for i, n in zip(*np.unique(p, return_counts=True))}
    save(status='applied', member_mode=member_mode, n_blended=len(sel_ids), entropy_s10=entropy(zl),
         temperature=cal_info['tp18'], weight=cal_info['w18'], hybrid_alpha=a.hybrid_alpha,
         changed=changed, changed_selected=changed_sel, member_argmax_agrees_s10=agrees,
         counts_s10=count(before), counts_v1=count(after), times=times, total_s=time.monotonic() - t0)


if __name__ == '__main__':
    main()
