"""Token-aligned differential and high-entropy likelihood features from paired GLM-5.3-Flash / GLM-5.3 scores."""
import numpy as np
import pandas as pd


def _single_extra(s: np.ndarray, prefix: str) -> dict[str, float]:
    lp = s[:, 0].astype(np.float64)
    h = s[:, 1].astype(np.float64)
    m2 = s[:, 2].astype(np.float64)
    rk = s[:, 3].astype(np.float64)
    n = len(lp)
    if n == 0:
        keys = (
            'lp_head32', 'lp_tail32', 'fdg_head32', 'fdg_tail32',
            'lp_hi_ent', 'z_hi_ent', 'top1_hi_ent', 'logrk_hi_ent',
            'lp_lo_ent', 'z_lo_ent', 'z_q10', 'z_q50', 'z_q90',
        )
        return {f'{prefix}:{k}': 0.0 for k in keys}

    var = np.maximum(m2 - h * h, 1e-9)
    sd = np.sqrt(var)
    z = (lp + h) / sd
    k32 = min(32, n)
    hi = h > 1.5
    lo = ~hi
    zq = np.quantile(z, [0.10, 0.50, 0.90])

    def _m(arr, mask, fallback):
        return float(arr[mask].mean()) if mask.any() else float(fallback)

    return {
        f'{prefix}:lp_head32': float(lp[:k32].mean()),
        f'{prefix}:lp_tail32': float(lp[-k32:].mean()),
        f'{prefix}:fdg_head32': float((lp[:k32] + h[:k32]).sum() / np.sqrt(var[:k32].sum())),
        f'{prefix}:fdg_tail32': float((lp[-k32:] + h[-k32:]).sum() / np.sqrt(var[-k32:].sum())),
        f'{prefix}:lp_hi_ent': _m(lp, hi, lp.mean()),
        f'{prefix}:z_hi_ent': _m(z, hi, z.mean()),
        f'{prefix}:top1_hi_ent': _m((rk == 0).astype(np.float64), hi, (rk == 0).mean()),
        f'{prefix}:logrk_hi_ent': _m(np.log1p(rk), hi, np.log1p(rk).mean()),
        f'{prefix}:lp_lo_ent': _m(lp, lo, lp.mean()),
        f'{prefix}:z_lo_ent': _m(z, lo, z.mean()),
        f'{prefix}:z_q10': float(zq[0]),
        f'{prefix}:z_q50': float(zq[1]),
        f'{prefix}:z_q90': float(zq[2]),
    }


def _paired_token_features(sf: np.ndarray, sg: np.ndarray) -> dict[str, float]:
    n = min(len(sf), len(sg))
    if n == 0:
        keys = (
            'dlp_mean', 'dlp_std', 'dlp_tstat', 'dlp_pos', 'dlp_big_pos', 'dlp_big_neg',
            'dlp_q10', 'dlp_q25', 'dlp_q50', 'dlp_q75', 'dlp_q90',
            'dlp_first', 'dlp_second', 'dlp_head32', 'dlp_tail32',
            'excl_t1_f', 'excl_t1_g', 'both_t1', 'excl_t1_diff',
            'excl_t5_f', 'excl_t5_g', 'excl_t5_diff',
            'dlrk_mean', 'dlrk_std', 'dlrk_tstat',
            'dz_mean', 'dz_std', 'dz_tstat', 'dz_pos',
            'dent_mean', 'dent_std', 'dent_pos',
            'dlp_hi_ent', 'dlp_hi_ent_pos', 'dz_hi_ent',
        )
        return {f'p:{k}': 0.0 for k in keys}

    sf, sg = sf[:n], sg[:n]
    lpf, hf, m2f, rkf = sf[:, 0].astype(np.float64), sf[:, 1].astype(np.float64), sf[:, 2].astype(np.float64), sf[:, 3].astype(np.float64)
    lpg, hg, m2g, rkg = sg[:, 0].astype(np.float64), sg[:, 1].astype(np.float64), sg[:, 2].astype(np.float64), sg[:, 3].astype(np.float64)

    zf = (lpf + hf) / np.sqrt(np.maximum(m2f - hf * hf, 1e-9))
    zg = (lpg + hg) / np.sqrt(np.maximum(m2g - hg * hg, 1e-9))

    dlp = lpf - lpg
    dlrk = np.log1p(rkf) - np.log1p(rkg)
    dz = zf - zg
    dent = hf - hg

    sqn = np.sqrt(float(n))
    dq = np.quantile(dlp, [0.10, 0.25, 0.50, 0.75, 0.90])
    half = max(n // 2, 1)
    k32 = min(32, n)
    hi = (0.5 * (hf + hg)) > 1.5

    ex_t1_f = float(((rkf == 0) & (rkg > 0)).mean())
    ex_t1_g = float(((rkg == 0) & (rkf > 0)).mean())
    ex_t5_f = float(((rkf < 5) & (rkg >= 5)).mean())
    ex_t5_g = float(((rkg < 5) & (rkf >= 5)).mean())

    return {
        'p:dlp_mean': float(dlp.mean()),
        'p:dlp_std': float(dlp.std()),
        'p:dlp_tstat': float(dlp.mean() * sqn / (dlp.std() + 1e-6)),
        'p:dlp_pos': float((dlp > 0.0).mean()),
        'p:dlp_big_pos': float((dlp > 1.0).mean()),
        'p:dlp_big_neg': float((dlp < -1.0).mean()),
        'p:dlp_q10': float(dq[0]),
        'p:dlp_q25': float(dq[1]),
        'p:dlp_q50': float(dq[2]),
        'p:dlp_q75': float(dq[3]),
        'p:dlp_q90': float(dq[4]),
        'p:dlp_first': float(dlp[:half].mean()),
        'p:dlp_second': float(dlp[half:].mean()) if n > half else float(dlp.mean()),
        'p:dlp_head32': float(dlp[:k32].mean()),
        'p:dlp_tail32': float(dlp[-k32:].mean()),
        'p:excl_t1_f': ex_t1_f,
        'p:excl_t1_g': ex_t1_g,
        'p:both_t1': float(((rkf == 0) & (rkg == 0)).mean()),
        'p:excl_t1_diff': ex_t1_f - ex_t1_g,
        'p:excl_t5_f': ex_t5_f,
        'p:excl_t5_g': ex_t5_g,
        'p:excl_t5_diff': ex_t5_f - ex_t5_g,
        'p:dlrk_mean': float(dlrk.mean()),
        'p:dlrk_std': float(dlrk.std()),
        'p:dlrk_tstat': float(dlrk.mean() * sqn / (dlrk.std() + 1e-6)),
        'p:dz_mean': float(dz.mean()),
        'p:dz_std': float(dz.std()),
        'p:dz_tstat': float(dz.mean() * sqn / (dz.std() + 1e-6)),
        'p:dz_pos': float((dz > 0.0).mean()),
        'p:dent_mean': float(dent.mean()),
        'p:dent_std': float(dent.std()),
        'p:dent_pos': float((dent > 0.0).mean()),
        'p:dlp_hi_ent': float(dlp[hi].mean()) if hi.any() else float(dlp.mean()),
        'p:dlp_hi_ent_pos': float((dlp[hi] > 0.0).mean()) if hi.any() else float((dlp > 0.0).mean()),
        'p:dz_hi_ent': float(dz[hi].mean()) if hi.any() else float(dz.mean()),
    }


def extract_paired_score_features(flash_npz_path, glm53_npz_path) -> pd.DataFrame:
    """Build DataFrame indexed by string id containing extra single-model and token-aligned paired features."""
    with np.load(flash_npz_path, allow_pickle=True) as fz, np.load(glm53_npz_path, allow_pickle=True) as gz:
        f_ids = fz['ids'].astype(str).tolist()
        g_ids = gz['ids'].astype(str).tolist()
        f_off = fz['offsets']
        g_off = gz['offsets']
        f_sc = fz['scores']
        g_sc = gz['scores']

    g_pos = {tid: idx for idx, tid in enumerate(g_ids)}
    common_ids = [tid for tid in f_ids if tid in g_pos]
    f_pos = {tid: idx for idx, tid in enumerate(f_ids)}

    rows = []
    for tid in common_ids:
        fi = f_pos[tid]
        gi = g_pos[tid]
        sf = f_sc[f_off[fi]:f_off[fi + 1]]
        sg = g_sc[g_off[gi]:g_off[gi + 1]]
        row = {'id': tid}
        row.update(_single_extra(sf, 'flash'))
        row.update(_single_extra(sg, 'glm53'))
        row.update(_paired_token_features(sf, sg))
        rows.append(row)

    df = pd.DataFrame(rows).set_index('id')
    assert np.isfinite(df.to_numpy(np.float64)).all(), 'Non-finite paired token features'
    return df
