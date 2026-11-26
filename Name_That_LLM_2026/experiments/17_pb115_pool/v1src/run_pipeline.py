"""Score texts under a big GLM model that ships as layer-segment GGUFs, with ntl-segrun.

S14 copy (docs/12 §2.15): run_pipeline_s13.py (frozen, the S13 payload) + --shrink-min. Without --shrink-min
it does exactly what run_pipeline_s13.py does.
S13 (docs/12 §2.12): run_pipeline_s12.py + the per-text user-turn prefix of src/segrun/run_pipeline.py
(--prefix with '{user}' and --user-template, functions copied verbatim).

--shrink-min N (S14): the input CSV is in priority order (the caller ranks it). When the projection of the rest
does not fit the budget, instead of stopping, the call keeps only the first m texts (the largest m >= N whose rest
fits, with a 10 % margin) for the remaining segments: the token file is rewritten with those m, and ntl-segrun
reads the first m texts of the state file (it holds the texts in order and may hold more than requested), so the
kept texts get exactly the scores they would have had. It may shrink again after a later segment. Only with one
chunk and without --pack (otherwise it stops as before). If not even N fit, exit code 3 as before.
Segment time model for the projection: t = unit * (phi + (1 - phi) * rows / rows0), phi = --shrink-fixed (share
that does not scale with the rows: loading the weights), unit fitted on the segments done.

Every text goes through segment k before segment k+1 is loaded, so each weight is read once per chunk.
The residual state between segments lives in one file on local disk, rewritten in place.
If the disk cannot hold the state of all texts, the texts are split into chunks (each chunk reads the
model again).

    python run_pipeline.py --bin BIN_DIR --segs 'GLOB' --csv train.csv --out OUT_DIR --tag flash
        [--work /tmp/ntl] [--prefix STR] [--max-text-tokens 320] [--seq-batch 8] [--state-dtype auto]
        [--limit N] [--deadline-s S] [--pack] [--ubatch-div P] [--budget-s S] [--prefetch [--prefetch-after F]]

--pack sorts the texts by token count (longest first) and packs them into padded decodes (see ntl-segrun.cpp);
the scores are put back in the input order. --budget-s is a hard limit for the whole call: every ntl-segrun call
gets the time left as its timeout, and after each segment the time of the rest is projected from the segments
done so far; if it does not fit, the call stops with exit code 3 (BudgetExceeded) and writes nothing.
--prefetch: once ntl-segrun has loaded segment k ("context ready"), a background thread reads segment k+1 once
(to /dev/null), so its pages are in the page cache when it is mapped; during the last segment it reads
--prefetch-after (e.g. the first segment of the next model). The same bytes, only earlier: scores are unchanged.

Writes OUT_DIR/{tag}_scores.npz (per-token logprob/entropy/m2/rank, offsets, ids), {tag}_features.csv
and {tag}_timing.json.
"""
import argparse
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import segio  # noqa: E402

CHAT_PREFIX = '[gMASK]<sop><|user|><|assistant|><think></think>'


def seg_index(path):
    m = re.search(r'seg(\d+)-of-(\d+)', os.path.basename(path))
    assert m, path
    return int(m.group(1)), int(m.group(2))


def gguf_kv_u32(path, key):
    """Read one uint32 metadata value from a GGUF header without the gguf package."""
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        _ver, _nt, nkv = struct.unpack('<IQQ', f.read(20))

        def rstr():
            (n,) = struct.unpack('<Q', f.read(8))
            return f.read(n).decode('utf-8', 'replace')

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(t):
            if t == 8:
                rstr()
            elif t == 9:
                it, n = struct.unpack('<IQ', f.read(12))
                if it in sizes:
                    f.seek(sizes[it] * n, 1)
                else:
                    for _ in range(n):
                        skip(it)
            else:
                f.seek(sizes[t], 1)

        for _ in range(nkv):
            k = rstr()
            (t,) = struct.unpack('<I', f.read(4))
            if k == key:
                assert t == 4, (key, t)
                return struct.unpack('<I', f.read(4))[0]
            skip(t)
    return None


class BudgetExceeded(RuntimeError):
    """The time budget (--budget-s) is not enough: stop and let the caller fall back."""


class EarlyFailure(RuntimeError):
    """ntl-segrun failed before writing any output (model load or context reservation, e.g. out of GPU memory)."""


_PREFETCHED = {}


def prefetch(path):
    """Read a file once in a daemon thread (page cache warm-up for the next mmap); each file at most once."""
    path = os.path.realpath(path)
    if path in _PREFETCHED:
        return
    info = _PREFETCHED[path] = {'s': None}

    def read():
        t0 = time.time()
        with open(path, 'rb', buffering=0) as f:
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
            buf = bytearray(16 << 20)
            while f.readinto(buf):
                pass
        info['s'] = time.time() - t0
    threading.Thread(target=read, daemon=True).start()


def run(cmd, log, timeout=None, on_ready=None):
    """Run one ntl-segrun call; on_ready() is called when it prints "context ready" (the model is loaded)."""
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    expired = threading.Event()

    def kill():
        expired.set()
        p.kill()
    timer = threading.Timer(timeout, kill) if timeout is not None else None
    if timer is not None:
        timer.daemon = True
        timer.start()
    lines = []
    for line in p.stdout:
        lines.append(line)
        if on_ready is not None and 'context ready' in line:
            on_ready()
            on_ready = None
    p.wait()
    if timer is not None:
        timer.cancel()
    out = ''.join(lines)
    if expired.is_set():
        with open(log, 'a') as lf:
            lf.write('$ ' + ' '.join(cmd) + '\n' + out + '\n[killed: timeout]\n')
        raise BudgetExceeded(f'timeout after {timeout:.0f} s: {" ".join(cmd)}')
    with open(log, 'a') as lf:
        lf.write('$ ' + ' '.join(cmd) + '\n' + out)
    dt = time.time() - t0
    if p.returncode != 0:
        print('\n'.join(out.splitlines()[-40:]), flush=True)
        err = EarlyFailure if 'context ready' not in out else RuntimeError
        raise err(f'failed ({p.returncode}): {" ".join(cmd)}')
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--segs', required=True, help='glob of the segment files')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--text-col', default='text')
    ap.add_argument('--id-col', default='id')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--work', default='/tmp/ntl')
    ap.add_argument('--prefix', default=CHAT_PREFIX)
    ap.add_argument('--max-text-tokens', type=int, default=320)
    ap.add_argument('--seq-batch', type=int, default=8)
    ap.add_argument('--state-dtype', default='auto', help='auto | f32 | f16 | bf16')
    ap.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 4))
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--disk-frac', type=float, default=0.8, help='share of the free work disk the state may use')
    ap.add_argument('--deadline-s', type=float, default=0, help='stop before starting a chunk after this many seconds')
    ap.add_argument('--pack', action='store_true', help='sort by length and pack padded decodes (ntl-segrun --pack)')
    ap.add_argument('--ubatch-div', type=int, default=1, help='n_ubatch = n_batch / P (pipeline parallel on 2 GPUs)')
    ap.add_argument('--budget-s', type=float, default=0, help='hard time limit of the whole call (0 = none)')
    ap.add_argument('--prefetch', action='store_true', help='read the next segment while the current one computes')
    ap.add_argument('--prefetch-after', default='', help='with --prefetch: file to read during the last segment')
    ap.add_argument('--shrink-min', type=int, default=0,
                    help='S14: on a budget overrun keep the first m >= N texts instead of stopping (0 = stop, as S13)')
    ap.add_argument('--shrink-fixed', type=float, default=0.5,
                    help='share of a segment time that does not scale with the rows (projection when shrinking)')
    ap.add_argument('--user-template', default='Write a single paragraph of about {n_words} words, with no title or headings.',
                    help="with '{user}' in --prefix: the user turn of each text ({n_words} = its word count)")
    a = ap.parse_args()
    try:
        main_run(a)
    except BudgetExceeded as e:
        print(f'[{a.tag}] BUDGET EXCEEDED: {e}', flush=True)
        sys.exit(3)


def user_prefix_tokens(a, segrun, tok_model, texts, tx, tk, log):
    """A prefix with a user turn that depends on the text: --prefix holds '{user}', replaced for each text by
    --user-template formatted with n_words (the text's word count), e.g. prvsiyan's GLM prefix
    '[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>{user}<|assistant|><think></think>'.
    ntl-segrun tokenize takes one prefix for all texts, so the tokens are put together from two calls (exact: the
    special tokens split the string before BPE, so each plain-text piece is tokenized on its own):
      full = tokenize(texts, prefix with an empty user turn) = H + T + text   (H = up to '{user}', T = the rest)
      user = tokenize(user turns, prefix H)                    = H + U_i
    and each text gets H + U_i + T + text, with n_prefix = len(H + U_i + T). Checked on the first 3 texts
    against a tokenization with their whole prefix; if that check fails (a tokenizer where the pieces are not
    independent), the exact fallback tokenizes each group of texts that share the user turn with its whole prefix."""
    users = [a.user_template.format(n_words=len(t.split())) for t in texts]
    try:
        return _user_prefix_pieces(a, segrun, tok_model, texts, users, tx, tk, log)
    except AssertionError as e:
        # exact fallback: one tokenize call per distinct user turn (texts with the same word count share it)
        print(f'[{a.tag}] user-turn prefix: piecewise tokens rejected ({e}); tokenizing by groups', flush=True)
    groups = {}
    for i, u in enumerate(users):
        groups.setdefault(u, []).append(i)
    seqs = [None] * len(texts)
    for j, (u, idx) in enumerate(groups.items()):
        gx, gk = tk + f'.g{j}.bin', tk + f'.g{j}_tok.bin'
        segio.write_texts(gx, [texts[i] for i in idx])
        run([segrun, 'tokenize', '--model', tok_model, '--texts', gx, '--out', gk, '--prefix', a.prefix.replace('{user}', u),
             '--max-text-tokens', str(a.max_text_tokens)], log)
        for i, sq in zip(idx, segio.read_tokens(gk)):
            seqs[i] = sq
        os.remove(gx); os.remove(gk)
    print(f'[{a.tag}] user-turn prefix: {len(groups)} groups, prefix {min(s[1] for s in seqs)}-{max(s[1] for s in seqs)} '
          f'tokens; e.g. {users[0]!r}', flush=True)
    return seqs


def _user_prefix_pieces(a, segrun, tok_model, texts, users, tx, tk, log):
    head = a.prefix.split('{user}')[0]
    run([segrun, 'tokenize', '--model', tok_model, '--texts', tx, '--out', tk, '--prefix', a.prefix.replace('{user}', ''),
         '--max-text-tokens', str(a.max_text_tokens)], log)
    full = segio.read_tokens(tk)
    ux, uk = tk + '.user_texts.bin', tk + '.user_tok.bin'
    segio.write_texts(ux, users)
    run([segrun, 'tokenize', '--model', tok_model, '--texts', ux, '--out', uk, '--prefix', head,
         '--max-text-tokens', '100000'], log)
    user = segio.read_tokens(uk)
    n_h = user[0][1]
    seqs = []
    for (f_ids, f_pre), (u_ids, u_pre) in zip(full, user):
        assert u_pre == n_h and f_pre >= n_h and list(f_ids[:n_h]) == list(u_ids[:n_h]), 'prefix head differs'
        seqs.append((np.concatenate([u_ids, f_ids[n_h:]]).astype('<i4'), len(u_ids) + f_pre - n_h))
    for i in range(min(3, len(texts))):
        one_x, one_k = tk + f'.check{i}.bin', tk + f'.check{i}_tok.bin'
        segio.write_texts(one_x, [texts[i]])
        run([segrun, 'tokenize', '--model', tok_model, '--texts', one_x, '--out', one_k,
             '--prefix', a.prefix.replace('{user}', users[i]), '--max-text-tokens', str(a.max_text_tokens)], log)
        ids, pre = segio.read_tokens(one_k)[0]
        assert pre == seqs[i][1] and list(ids) == list(seqs[i][0]), f'user-prefix tokens differ for text {i}'
    print(f'[{a.tag}] user-turn prefix: head {n_h} tokens, prefix {min(s[1] for s in seqs)}-{max(s[1] for s in seqs)} '
          f'tokens (pieces checked on {min(3, len(texts))} texts); e.g. {users[0]!r}', flush=True)
    return seqs


def project_rest(a, seg_t, seg_rows, rows0, rows_kept, rest):
    """Projected time of the `rest` segments (the last counted double, as in the S13 projection) if only the first
    m texts are kept, for every m (array, increasing). Per-segment time t = unit * (phi + (1 - phi) * rows / rows0),
    unit = mean over the segments done (each normalised by the rows it ran with)."""
    phi = a.shrink_fixed
    unit = float(np.mean([t / (phi + (1 - phi) * r / rows0) for t, r in zip(seg_t, seg_rows)]))
    return unit * (phi + (1 - phi) * np.cumsum(rows_kept) / rows0) * (rest + (1 if rest > 0 else 0))


def main_run(a):

    t_start = time.time()

    def left():
        return None if not a.budget_s else a.budget_s - (time.time() - t_start)

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    log = os.path.join(a.out, f'{a.tag}_run.log')
    segrun = os.path.join(a.bin, 'ntl-segrun')

    segs = sorted(glob.glob(a.segs), key=lambda p: seg_index(p)[0])
    assert segs, a.segs
    n_seg = seg_index(segs[0])[1]
    assert [seg_index(p)[0] for p in segs] == list(range(n_seg)), segs
    ranges = [(gguf_kv_u32(p, 'ntl.layer_start'), gguf_kv_u32(p, 'ntl.layer_end')) for p in segs]
    print(f'[{a.tag}] {n_seg} segments, layers {ranges}', flush=True)
    for (s0, e0), (s1, _e1) in zip(ranges, ranges[1:]):
        assert e0 == s1, ranges

    df = pd.read_csv(a.csv)
    if a.limit > 0:
        df = df.iloc[:a.limit]
    texts = df[a.text_col].astype(str).tolist()
    tx = os.path.join(a.work, f'{a.tag}_texts.bin')
    tk = os.path.join(a.work, f'{a.tag}_tok.bin')
    segio.write_texts(tx, texts)
    if '{user}' in a.prefix:
        seqs = user_prefix_tokens(a, segrun, segs[0], texts, tx, tk, log)
    else:
        run([segrun, 'tokenize', '--model', segs[0], '--texts', tx, '--out', tk, '--prefix', a.prefix,
             '--max-text-tokens', str(a.max_text_tokens)], log)
        seqs = segio.read_tokens(tk)
    n_rows = np.array([len(ids) for ids, _ in seqs])
    # processing order: longest first when packing (similar lengths share a decode, little padding)
    order = np.argsort(-n_rows, kind='stable') if a.pack else np.arange(len(seqs))
    print(f'[{a.tag}] {len(seqs)} texts, {n_rows.sum()} rows, max {n_rows.max()}, prefix {seqs[0][1]}', flush=True)

    # state width: glm5-next hc*n_embd, glm-dsa n_embd (read from the first segment's metadata)
    n_embd = gguf_kv_u32(segs[0], 'glm5-next.embedding_length')
    if n_embd is not None:
        width = gguf_kv_u32(segs[0], 'glm5-next.hyper_connection.count') * n_embd
    else:
        width = gguf_kv_u32(segs[0], 'glm-dsa.embedding_length')
    free = shutil.disk_usage(a.work).free
    if a.state_dtype == 'auto':
        # exact f32 if the whole state fits in one chunk, else bf16 (half the disk, no overflow risk)
        a.state_dtype = 'f32' if n_rows.sum() * width * 4 <= free * a.disk_frac else 'bf16'
        print(f'[{a.tag}] state dtype auto -> {a.state_dtype}', flush=True)
    el = 2 if a.state_dtype in ('f16', 'bf16') else 4
    max_rows = int(free * a.disk_frac / (width * el))
    chunks, cur, cur_rows = [], [], 0
    for i in order:
        r = n_rows[i]
        if cur and cur_rows + r > max_rows:
            chunks.append(cur)
            cur, cur_rows = [], 0
        cur.append(i)
        cur_rows += r
    chunks.append(cur)
    print(f'[{a.tag}] state width {width}, {el} B/value, {free/2**30:.1f} GiB free on {a.work} -> '
          f'{len(chunks)} chunk(s)', flush=True)

    multi_gpu = False
    try:
        out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout
        multi_gpu = out.count('GPU ') > 1
    except FileNotFoundError:
        pass

    timing = {'tag': a.tag, 'n_texts': len(seqs), 'n_rows': int(n_rows.sum()), 'chunks': [], 'segments': segs,
              'ranges': ranges, 'seq_batch': a.seq_batch, 'state_dtype': a.state_dtype, 'prefix': a.prefix,
              'pack': a.pack, 'ubatch_div': a.ubatch_div, 'budget_s': a.budget_s}
    all_scores = [None] * len(seqs)
    st = os.path.join(a.work, f'{a.tag}_state.bin')
    done = 0
    can_shrink = a.shrink_min > 0 and len(chunks) == 1 and not a.pack
    if can_shrink:
        timing['shrinks'] = []
    for ci, idx in enumerate(chunks):
        if a.deadline_s and time.time() - t_start > a.deadline_s:
            print(f'[{a.tag}] deadline reached before chunk {ci}, stopping', flush=True)
            break
        ctk = os.path.join(a.work, f'{a.tag}_tok_c{ci}.bin')
        segio.write_tokens(ctk, [seqs[i] for i in idx])
        sc = os.path.join(a.work, f'{a.tag}_scores_c{ci}.bin')
        seg_t = []
        rows0 = int(n_rows[idx].sum())
        seg_rows = []   # rows of the chunk when each segment ran (they drop after a shrink)
        for k, p in enumerate(segs):
            while True:
                cmd = [segrun, 'run', '--model', p, '--tokens', ctk, '--threads', str(a.threads),
                       '--seq-batch', str(a.seq_batch), '--state-dtype', a.state_dtype]
                if multi_gpu:
                    cmd.append('--split-mid')
                if a.pack:
                    cmd.append('--pack')
                if a.ubatch_div > 1:
                    cmd += ['--ubatch-div', str(a.ubatch_div)]
                if k > 0:
                    cmd += ['--state-in', st]
                cmd += ['--out', sc if k == n_seg - 1 else st]
                if left() is not None and left() <= 0:
                    raise BudgetExceeded('no time left')
                nxt = (segs[k + 1] if k + 1 < n_seg else segs[0] if ci + 1 < len(chunks) else a.prefetch_after)
                ready = (lambda f=nxt: prefetch(f)) if a.prefetch and nxt else None
                try:
                    dt = run(cmd, log, timeout=left(), on_ready=ready)
                    break
                except EarlyFailure:
                    # nothing written yet (the input state is intact): retry with half the texts per decode
                    if a.seq_batch == 1:
                        raise
                    a.seq_batch //= 2
                    timing['seq_batch_reduced_to'] = a.seq_batch
                    print(f'[{a.tag}] segment {k+1} failed before writing: retrying with seq-batch {a.seq_batch}', flush=True)
            seg_t.append(dt)
            seg_rows.append(int(n_rows[idx].sum()))
            print(f'[{a.tag}] chunk {ci+1}/{len(chunks)} segment {k+1}/{n_seg} {os.path.basename(p)}: {dt:.0f} s '
                  f'(total {time.time()-t_start:.0f} s)', flush=True)
            if a.budget_s:
                # projection: the remaining segments of this chunk at the mean so far (the last one, with the
                # softmax over the vocabulary, counts double), plus the other chunks at this chunk's pace
                rest = n_seg - k - 1
                per = float(np.mean(seg_t))
                proj = per * (rest + (1 if rest > 0 else 0)) + per * n_seg * 1.15 * (len(chunks) - ci - 1)
                if can_shrink:   # one chunk: the same projection, aware of the rows left after a shrink
                    proj_m_all = project_rest(a, seg_t, seg_rows, rows0, n_rows[idx], rest)
                    proj = float(proj_m_all[-1])
                if proj > left():
                    if not can_shrink:
                        raise BudgetExceeded(f'projected {proj:.0f} s for the rest, {left():.0f} s left')
                    m = int((proj_m_all <= 0.9 * left()).sum())   # increasing in m: the count is the largest prefix
                    proj_m = float(proj_m_all[m - 1]) if m else float('inf')
                    if m < a.shrink_min:
                        raise BudgetExceeded(f'projected {proj:.0f} s for the rest, {left():.0f} s left; '
                                             f'not even {a.shrink_min} texts fit')
                    timing['shrinks'].append({'after_segment': k + 1, 'from': len(idx), 'to': m,
                                              'left_s': round(left()), 'proj_all_s': round(proj),
                                              'proj_kept_s': round(proj_m)})
                    print(f'[{a.tag}] SHRINK after segment {k+1}/{n_seg}: {len(idx)} -> {m} texts '
                          f'(rest projected {proj:.0f} s with all, {proj_m:.0f} s with {m}; {left():.0f} s left)',
                          flush=True)
                    idx = idx[:m]
                    segio.write_tokens(ctk, [seqs[i] for i in idx])
        for i, s in zip(idx, segio.read_scores(sc)):
            all_scores[i] = s
        done += len(idx)
        timing['chunks'].append({'n_texts': len(idx), 'seg_seconds': seg_t})
        for f in (ctk, sc, st):
            if os.path.exists(f):
                os.remove(f)

    ok = [i for i, s in enumerate(all_scores) if s is not None]
    off = np.zeros(len(ok) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(all_scores[i]) for i in ok])
    np.savez_compressed(os.path.join(a.out, f'{a.tag}_scores.npz'),
                        ids=df[a.id_col].astype(str).values[ok],
                        scores=np.concatenate([all_scores[i] for i in ok]).astype(np.float32),
                        offsets=off,
                        n_prefix=np.array([seqs[i][1] for i in ok]),
                        tokens=np.concatenate([seqs[i][0][seqs[i][1]:] for i in ok]).astype(np.int32))
    feats = features.text_features([all_scores[i] for i in ok], [texts[i] for i in ok])
    feats.insert(0, a.id_col, df[a.id_col].astype(str).values[ok])
    feats.to_csv(os.path.join(a.out, f'{a.tag}_features.csv'), index=False)
    timing['total_s'] = time.time() - t_start
    if a.prefetch:
        timing['prefetch_s'] = {os.path.basename(k): v['s'] for k, v in _PREFETCHED.items()}
    timing['n_done'] = len(ok)
    with open(os.path.join(a.out, f'{a.tag}_timing.json'), 'w') as f:
        json.dump(timing, f, indent=1)
    print(f'[{a.tag}] done: {len(ok)}/{len(seqs)} texts in {timing["total_s"]:.0f} s', flush=True)


if __name__ == '__main__':
    main()
