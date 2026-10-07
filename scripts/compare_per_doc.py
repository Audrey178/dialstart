"""Paired per-meeting comparison of segmentation runs.

Reads metric/<model>/<split>_per_doc.json (written by test.py) for a baseline
and one or more candidates, matches documents by file name and reports, for
each candidate minus baseline: mean difference, a paired bootstrap 95% CI,
win/tie/loss counts and a Wilcoxon signed-rank p-value. Lower Pk/WD is better,
so a negative difference means the candidate is better.

    python scripts/compare_per_doc.py --split test --baseline model_dialstart_vi_phobert \
        model_dialstart_vi_phobert_gold_margin model_dialstart_vi_phobert_gold_supcon
"""
import json
import argparse
import numpy as np


def load(root, model, split):
    docs = json.load(open(f'{root}/metric/{model}/{split}_per_doc.json'))
    return {d['file']: d for d in docs}


def paired_bootstrap_ci(diff, n_boot, seed):
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]), float((means >= 0).mean())


def wilcoxon_p(diff):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float('nan')
    if np.all(diff == 0):
        return 1.0
    return float(wilcoxon(diff).pvalue)


def main(args):
    base = load(args.root, args.baseline, args.split)
    for cand_name in args.candidates:
        cand = load(args.root, cand_name, args.split)
        files = sorted(set(base) & set(cand))
        if len(files) != len(base) or len(files) != len(cand):
            print(f'warning: {len(base)} baseline / {len(cand)} candidate docs, comparing {len(files)} shared')
        print(f'\n=== {cand_name}  vs  {args.baseline}  ({args.split}, {len(files)} docs)')
        for metric in ['pk', 'wd']:
            b = np.array([base[f][metric] for f in files])
            c = np.array([cand[f][metric] for f in files])
            diff = c - b
            (lo, hi), p_not_better = paired_bootstrap_ci(diff, args.n_boot, args.seed)
            wins, ties, losses = int((diff < -1e-9).sum()), int((np.abs(diff) <= 1e-9).sum()), int((diff > 1e-9).sum())
            print(f'{metric}: base {b.mean()*100:.2f}  cand {c.mean()*100:.2f}  diff {diff.mean()*100:+.2f} '
                  f'[95% CI {lo*100:+.2f}, {hi*100:+.2f}]  P(boot diff>=0)={p_not_better:.3f}  '
                  f'win/tie/loss {wins}/{ties}/{losses}  wilcoxon p={wilcoxon_p(diff):.3f}')
        if args.show_docs:
            print(f'{"file":<28}{"utt":>6}{"seg":>5}{"base pk":>9}{"cand pk":>9}{"diff":>8}')
            for f in sorted(files, key=lambda f: cand[f]['pk'] - base[f]['pk']):
                d = cand[f]['pk'] - base[f]['pk']
                print(f'{f:<28}{base[f]["n_utt"]:>6}{base[f]["n_seg"]:>5}'
                      f'{base[f]["pk"]*100:>9.2f}{cand[f]["pk"]*100:>9.2f}{d*100:>+8.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('candidates', nargs='+', help='model dirs under metric/ to compare against --baseline')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--root', default='.')
    parser.add_argument('--n_boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_docs', action='store_true', help='print the per-meeting Pk table')
    main(parser.parse_args())
