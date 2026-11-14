"""Rank by GLM probability and add two-model likelihood evidence on selected texts."""

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


def member_logp(model, feats, ids):
    """feats: {'flash': DataFrame, 'glm53': DataFrame} indexed by id -> (n, 18) log-probs, model['classes'] order."""
    blocks = [feats[t].add_prefix(f'{t}:') for t in ('flash', 'glm53')]
    F = pd.concat(blocks, axis=1).reindex(ids)
    for c in model['diff_features']:
        F[f'd:{c}'] = F[f'flash:{c}'] - F[f'glm53:{c}']
    X = F[model['features']].to_numpy(np.float64)
    assert np.isfinite(X).all(), 'non-finite likelihood features'
    z = ((X - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coef']).T + np.asarray(model['intercept'])
    return log_softmax(z, 1)


def main():
    ap = argparse.ArgumentParser()
    for k in ('--test-csv', '--final-logits', '--out', '--report', '--bin', '--pipeline', '--flash-segs', '--glm53-segs',
              '--model', '--work', '--outdir'):
        ap.add_argument(k, required=True)
    ap.add_argument('--pack', action='store_true')
    ap.add_argument('--ubatch-div', type=int, default=1)
    ap.add_argument('--seq-batch', type=int, default=8)
    ap.add_argument('--max-text-tokens', type=int, default=0,
                    help='0 = the cut the member was trained with (model max_text_tokens); any other value must match it')
    ap.add_argument('--select-classes', default='glm-5.3,glm-5.3-flash',
                    help='texts are scored in decreasing order of the S10 probability of these classes')
    ap.add_argument('--max-cover', type=float, default=0.20, help='at most this fraction of the test is scored')
    ap.add_argument('--min-texts', type=int, default=50, help='fewer texts than this fit: keep S10')
    ap.add_argument('--prefetch', action='store_true', help='run_pipeline --prefetch (page-cache warm-up of segments)')
    ap.add_argument('--max-changed', type=float, default=0.25,
                    help='sanity guard: the blend changes ~11%% of the predictions on train OOF (W=1); far more means '
                         'something is broken (e.g. features unlike train): fail instead of submitting it')
    a = ap.parse_args()

    t0 = time.monotonic()
    rep = {'status': 'started'}
    out_report = Path(a.report)

    def save(**kw):
        rep.update(kw)
        out_report.write_text(json.dumps(rep, indent=1))
        print('V1', json.dumps({k: v for k, v in kw.items() if not isinstance(v, dict)}), flush=True)

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

    # Bound work before starting, below the measured 160k-token pass.
    tpc = model.get('tok_per_char', 0.235)
    rows_i = np.minimum(frame.text.str.len().to_numpy() * tpc * 1.05, a.max_text_tokens) + 13
    cum = np.cumsum(rows_i[order])
    n_sel = min(int((cum <= 130000).sum()), int(a.max_cover * len(ids)))
    save(status='selected', n_texts=len(ids), n_selected=n_sel,
         estimated_token_rows=int(cum[n_sel-1]) if n_sel else 0)
    if n_sel < a.min_texts:
        save(status='fallback_s10', reason='Insufficient selected texts')
        return
    sel = np.sort(order[:n_sel])
    sel_ids = [ids[i] for i in sel]

    Path(a.work).mkdir(parents=True, exist_ok=True)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    times = {}
    for tag, segs in (('flash', a.flash_segs), ('glm53', a.glm53_segs)):
        query = Path(a.work) / f'v1_query_{tag}.csv'
        frame.iloc[sel][['id', 'text']].to_csv(query, index=False)
        cmd = [sys.executable, a.pipeline, '--bin', a.bin, '--segs', segs, '--csv', str(query), '--out', a.outdir,
               '--tag', tag, '--work', str(Path(a.work) / 'state'), '--seq-batch', str(a.seq_batch),
               '--state-dtype', 'auto', '--max-text-tokens', str(a.max_text_tokens), '--prefix', PREFIX]
        if a.pack:
            cmd.append('--pack')
        if a.ubatch_div > 1:
            cmd += ['--ubatch-div', str(a.ubatch_div)]
        if a.prefetch:
            cmd.append('--prefetch')
            if tag == 'flash':
                cmd += ['--prefetch-after', json.loads(a.glm53_segs)[0]]
        t = time.monotonic()
        with open(Path(a.outdir) / f'{tag}_pipeline.txt', 'w') as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        times[tag] = time.monotonic() - t
        if code != 0:
            raise RuntimeError(f'V1 {tag} pipeline failed with exit code {code}; see {tag}_pipeline.txt')
        save(status=f'{tag}_done', times=times)

    feats = {t: pd.read_csv(Path(a.outdir) / f'{t}_features.csv', dtype={'id': str}).set_index('id')
             for t in ('flash', 'glm53')}
    for t, f in feats.items():
        assert set(sel_ids) <= set(f.index), f'{t}: features missing for some texts'
        feats[t] = f.loc[sel_ids]
    ll = member_logp(model, feats, sel_ids)[:, [model['classes'].index(c) for c in labels]]

    tp = match_temperature(zl, model['entropy_ref'])
    w = model['W'] / tp
    before = zl.argmax(1)
    after = before.copy()
    after[sel] = (zl[sel] + w * ll).argmax(1)
    changed = float((before != after).mean())
    changed_sel = float((before[sel] != after[sel]).mean())
    agrees = float((ll.argmax(1) == before[sel]).mean())
    if changed > a.max_changed:
        save(status='error_sanity', changed=changed, changed_selected=changed_sel, temperature=tp, weight=w,
             member_argmax_agrees_s10=agrees)
        raise RuntimeError(f'V1 blend changes {changed:.1%} of the predictions (> {a.max_changed:.0%}): not submitted')
    sub = pd.DataFrame({'id': ids, 'model': np.asarray(labels)[after]})
    assert len(sub) == len(frame) and sub.id.is_unique and sub.model.isin(labels).all()
    sub.to_parquet(a.out, index=False)
    assert pd.read_parquet(a.out).equals(sub)
    count = lambda p: {labels[i]: int(n) for i, n in zip(*np.unique(p, return_counts=True))}
    save(status='applied', n_blended=len(sel_ids), entropy_s10=entropy(zl), temperature=tp, weight=w, changed=changed,
         changed_selected=changed_sel, member_argmax_agrees_s10=agrees,
         counts_s10=count(before), counts_v1=count(after), times=times, total_s=time.monotonic() - t0)


if __name__ == '__main__':
    main()
