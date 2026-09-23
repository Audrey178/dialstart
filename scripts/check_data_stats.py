#!/usr/bin/env python3
"""Sanity-check a segmentation dataset before training on it.

Reads the same .txt folders that data_preprocess.py / eval_utils.py read
(one utterance per line, "================" after the last utterance of a
segment) and reports, per split:

  - diversity: sessions, segments, distinct segments (by exact text),
    distinct utterances, how often each distinct segment is reused, and
    segments repeated inside a single session
  - segment length (utterances) distribution and boundaries per session
  - segment-opener first-word distribution, and its JS divergence from train
  - overlap with train: shared distinct segments and shared utterances
  - model-free baselines scored exactly like eval_utils.evaluate_dataset:
      even    - oracle boundary count, boundaries spaced evenly
      stride  - no oracle, a boundary every round(train mean seg length)
      random  - oracle boundary count, random gaps (mean over --seeds)

If a baseline that only looks at positions (even / stride) gets a low Pk, the
data is predictable from position alone and the model will learn that
instead of topic shifts.

Exits 1 when any FAIL check trips, so it can gate training:
    python scripts/check_data_stats.py && bash scripts/train_vn.sh
"""

import os
import re
import sys
import json
import math
import random
import argparse
import statistics
from collections import Counter

import segeval

MARKER = '================'


def read_session(path):
    """Return (utterances, seg_r) parsed the same way as eval_utils."""
    text, seg_r, tmp = [], [], 0
    for line in open(path, encoding='utf-8'):
        if MARKER not in line.strip():
            text.append(line.strip())
            tmp += 1
        else:
            seg_r.append(tmp)
            tmp = 0
    seg_r.append(tmp)
    return text, seg_r


def segments_of(text, seg_r):
    out, start = [], 0
    for n in seg_r:
        if n > 0:
            out.append(tuple(text[start:start + n]))
        start += n
    return out


def load_split(folder, max_docs=None):
    files = sorted(f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    if max_docs is not None:
        files = files[:max_docs]
    return [read_session(os.path.join(folder, f)) for f in files]


def norm(s):
    return re.sub(r'\s+', ' ', s).strip().lower()


def opener_word(seg):
    words = re.findall(r'\w+', seg[0].lower())
    return words[0] if words else ''


def labels_to_masses(labels):
    # Same construction as seg_p in eval_utils.evaluate_dataset.
    seg_p, tmp = [], 0
    for fake in labels:
        tmp += 1
        if fake == 1:
            seg_p.append(tmp)
            tmp = 0
    seg_p.append(tmp)
    return seg_p


def score(boundaries, n_utts, seg_r):
    labels = [0] * n_utts
    for i in boundaries:
        labels[i] = 1
    seg_p = labels_to_masses(labels)
    return float(segeval.pk(seg_p, seg_r)), float(segeval.window_diff(seg_p, seg_r))


def baselines(sessions, stride, seeds):
    acc = {'even': [0, 0], 'stride': [0, 0], 'random': [0, 0]}
    c = 0
    for text, seg_r in sessions:
        n = len(text)
        if n < 2:
            continue
        gaps = n - 1  # boundary after utterance i, i in [0, n-2]
        k = min(len(seg_r) - 1, gaps)

        even = sorted({round((j + 1) * n / (k + 1)) - 1 for j in range(k)})
        even = [min(max(i, 0), gaps - 1) for i in even]
        for m, (pk, wd) in (('even', score(even, n, seg_r)),
                            ('stride', score(list(range(stride - 1, gaps, stride)), n, seg_r))):
            acc[m][0] += pk
            acc[m][1] += wd

        pk_r = wd_r = 0
        for s in range(seeds):
            rnd = random.Random(s * 100003 + c)
            pk, wd = score(rnd.sample(range(gaps), k), n, seg_r)
            pk_r += pk
            wd_r += wd
        acc['random'][0] += pk_r / seeds
        acc['random'][1] += wd_r / seeds
        c += 1
    return {m: (v[0] / c, v[1] / c) for m, v in acc.items()} if c else {}


def js_divergence(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    js = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = (a + b) / 2
        if a:
            js += 0.5 * a * math.log2(a / m)
        if b:
            js += 0.5 * b * math.log2(b / m)
    return js


def describe(name, sessions):
    segs = [s for text, seg_r in sessions for s in segments_of(text, seg_r)]
    seg_counts = Counter(segs)
    lens = [len(s) for s in segs]
    bounds = [len(seg_r) - 1 for _, seg_r in sessions]
    in_session_repeats = sum(len(ss) - len(set(ss)) for ss in
                             (segments_of(t, r) for t, r in sessions))
    utts = {norm(u) for text, _ in sessions for u in text}
    return {
        'name': name,
        'sessions': len(sessions),
        'segments': len(segs),
        'distinct_segments': len(seg_counts),
        'distinct_utterances': len(utts),
        'mean_uses_per_segment': len(segs) / max(1, len(seg_counts)),
        'max_uses_per_segment': max(seg_counts.values(), default=0),
        'in_session_repeats': in_session_repeats,
        'seg_len_mean': statistics.mean(lens) if lens else 0,
        'seg_len_std': statistics.pstdev(lens) if lens else 0,
        'seg_len_min': min(lens, default=0),
        'seg_len_max': max(lens, default=0),
        'seg_len_hist': dict(sorted(Counter(lens).items())),
        'boundaries_mean': statistics.mean(bounds) if bounds else 0,
        'boundaries_hist': dict(sorted(Counter(bounds).items())),
        'openers': Counter(opener_word(s) for s in segs),
        '_segs': set(seg_counts),
        '_utts': utts,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataroot', default='./data')
    p.add_argument('--train', default='vn_synth_train', help='Train folder under --dataroot')
    p.add_argument('--eval', nargs='*', default=['vn_synth_val', 'vn_synth_test'],
                   help='Eval folders under --dataroot (missing ones are skipped)')
    p.add_argument('--max_docs', type=int, default=None, help='Only read the first N sessions of each split')
    p.add_argument('--seeds', type=int, default=20, help='Random-baseline repeats per session')
    p.add_argument('--top_openers', type=int, default=8)
    p.add_argument('--min_train_segments', type=int, default=500)
    p.add_argument('--min_eval_segments', type=int, default=40)
    p.add_argument('--max_uses', type=int, default=5, help='Warn when a distinct train segment is reused more than this')
    p.add_argument('--min_position_pk', type=float, default=0.2,
                   help='FAIL when the even/stride baseline Pk is below this')
    p.add_argument('--max_opener_js', type=float, default=0.3)
    p.add_argument('--json', help='Also write the report to this path')
    args = p.parse_args()

    names = [args.train] + [e for e in args.eval]
    stats, checks = {}, []

    def check(level, ok, msg):
        checks.append((('OK  ' if ok else level), msg))

    for name in names:
        folder = os.path.join(args.dataroot, name)
        if not os.path.isdir(folder):
            if name == args.train:
                sys.exit(f'train folder not found: {folder} (run scripts/convert_sessions_to_txt.py)')
            print(f'[skip] {folder} not found')
            continue
        stats[name] = describe(name, load_split(folder, args.max_docs))

    train = stats[args.train]
    stride = max(1, round(train['seg_len_mean']))
    for name, s in stats.items():
        s['baselines'] = baselines(load_split(os.path.join(args.dataroot, name), args.max_docs),
                                   stride, args.seeds)

    for name, s in stats.items():
        print(f'\n===== {name} =====')
        print(f"sessions {s['sessions']} | segments {s['segments']} | distinct segments {s['distinct_segments']}"
              f" | distinct utterances {s['distinct_utterances']}")
        print(f"uses per distinct segment: mean {s['mean_uses_per_segment']:.1f}, max {s['max_uses_per_segment']}"
              f" | segments repeated inside a session: {s['in_session_repeats']}")
        print(f"segment length: mean {s['seg_len_mean']:.2f} std {s['seg_len_std']:.2f}"
              f" [{s['seg_len_min']}, {s['seg_len_max']}]  hist {s['seg_len_hist']}")
        print(f"boundaries/session: mean {s['boundaries_mean']:.2f}  hist {s['boundaries_hist']}")
        total = sum(s['openers'].values()) or 1
        top = ', '.join(f'{w or "<none>"} {c / total:.0%}' for w, c in s['openers'].most_common(args.top_openers))
        print(f'segment openers: {top}')
        if name != args.train:
            s['opener_js_vs_train'] = js_divergence(s['openers'], train['openers'])
            s['shared_segments_with_train'] = len(s['_segs'] & train['_segs'])
            s['shared_utterances_with_train'] = len(s['_utts'] & train['_utts'])
            print(f"vs train: opener JS {s['opener_js_vs_train']:.3f} | shared distinct segments "
                  f"{s['shared_segments_with_train']} | shared utterances {s['shared_utterances_with_train']}"
                  f" ({s['shared_utterances_with_train'] / max(1, s['distinct_utterances']):.1%})")
        print('baselines (pk / wd):  ' + '  '.join(
            f"{m}{'@' + str(stride) if m == 'stride' else ''} {pk:.3f} / {wd:.3f}"
            for m, (pk, wd) in s['baselines'].items()))

    check('FAIL', train['seg_len_std'] > 0, f"train segment length std {train['seg_len_std']:.2f} > 0")
    check('WARN', train['distinct_segments'] >= args.min_train_segments,
          f"train distinct segments {train['distinct_segments']} >= {args.min_train_segments}")
    check('WARN', train['max_uses_per_segment'] <= args.max_uses,
          f"train max uses per distinct segment {train['max_uses_per_segment']} <= {args.max_uses}")
    check('WARN', train['in_session_repeats'] == 0,
          f"train segments repeated inside a session: {train['in_session_repeats']}")
    for name, s in stats.items():
        for m in ('even', 'stride'):
            if m in s['baselines']:
                pk = s['baselines'][m][0]
                check('FAIL', pk >= args.min_position_pk,
                      f'{name} {m} baseline pk {pk:.3f} >= {args.min_position_pk} (not guessable from position)')
        if name == args.train:
            continue
        check('WARN', s['distinct_segments'] >= args.min_eval_segments,
              f"{name} distinct segments {s['distinct_segments']} >= {args.min_eval_segments}")
        check('FAIL', s['shared_segments_with_train'] == 0,
              f"{name} shares {s['shared_segments_with_train']} distinct segments with train")
        check('WARN', s['opener_js_vs_train'] <= args.max_opener_js,
              f"{name} opener JS vs train {s['opener_js_vs_train']:.3f} <= {args.max_opener_js}")
    names_present = list(stats)
    for i, a in enumerate(names_present[1:], 1):
        for b in names_present[i + 1:]:
            shared = len(stats[a]['_segs'] & stats[b]['_segs'])
            check('FAIL', shared == 0, f'{a} and {b} share {shared} distinct segments')

    print('\n===== checks =====')
    for level, msg in checks:
        print(f'[{level}] {msg}')

    if args.json:
        out = {n: {k: (dict(v.most_common()) if isinstance(v, Counter) else v)
                   for k, v in s.items() if not k.startswith('_')} for n, s in stats.items()}
        out['checks'] = [{'level': l.strip(), 'msg': m} for l, m in checks]
        json.dump(out, open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'report written to {args.json}')

    n_fail = sum(l == 'FAIL' for l, _ in checks)
    n_warn = sum(l == 'WARN' for l, _ in checks)
    print(f'\n{n_fail} FAIL, {n_warn} WARN')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
