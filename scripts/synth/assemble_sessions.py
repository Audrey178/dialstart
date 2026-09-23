#!/usr/bin/env python3
"""Step B2: assemble D_i into multi-topic sessions, one split at a time.

Differences from the skill's assemble_sessions.py (the cause of the
memorisation seen in vn_synth):
  - a D_i never appears twice in one session
  - each D_i is used at most --max_uses (train) / --eval_max_uses times in
    its split, so the session count follows the corpus size instead of a
    fixed 2500
  - --same_domain_prob of adjacent pairs share a domain but come from
    different source docs (hard boundaries); all other pairs change domain
  - --min_segs..--max_segs segments per session
  - D_i only mix with D_i of the same split (set by make_gen_jobs.py)

Writes <out_dir>/<split>/session_<id>.json plus the .txt files that
data_preprocess.py / eval_utils.py read, in <txt_root>/<prefix>_<split>/.

    python scripts/synth/assemble_sessions.py --dialogues_dir data/vn_synth_v2/single_topic_dialogues \
        --out_dir data/vn_synth_v2/sessions --txt_root data --prefix vn2
"""

import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from convert_sessions_to_txt import session_to_lines  # noqa: E402


def pick(rng, candidates, uses, max_uses):
    # Favour the least-used D_i so the budget is spread evenly.
    weights = [max_uses - uses[d['dialogue_id']] for d in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def build_session(pool, uses, max_uses, n_segs, same_domain_prob, rng):
    avail = [d for d in pool if uses[d['dialogue_id']] < max_uses]
    if not avail:
        return None
    chosen = [pick(rng, avail, uses, max_uses)]
    while len(chosen) < n_segs:
        last = chosen[-1]
        taken = {d['dialogue_id'] for d in chosen}
        free = [d for d in avail if d['dialogue_id'] not in taken]
        same = [d for d in free if d['domain'] == last['domain'] and d['source_doc_id'] != last['source_doc_id']]
        other = [d for d in free if d['domain'] != last['domain']]
        first, second = (same, other) if rng.random() < same_domain_prob else (other, same)
        cands = first or second
        if not cands:
            break
        chosen.append(pick(rng, cands, uses, max_uses))
    return chosen


def assemble_split(split, pool, args, rng):
    max_uses = args.max_uses if split == 'train' else args.eval_max_uses
    uses = Counter()
    sessions = []
    while True:
        segs = build_session(pool, uses, max_uses, rng.randint(args.min_segs, args.max_segs),
                             args.same_domain_prob, rng)
        if not segs or len(segs) < args.min_segs:
            break
        for d in segs:
            uses[d['dialogue_id']] += 1
        utterances, boundaries = [], []
        for d in segs:
            utterances.extend(d['utterances'])
            boundaries.append(len(utterances) - 1)
        sessions.append({
            'session_id': f'{split}_{len(sessions):04d}',
            'split': split,
            'num_sub_dialogues': len(segs),
            'num_turns': len(utterances),
            'utterances': utterances,
            'gold_boundaries': boundaries[:-1],
            'domain_sequence': [d['domain'] for d in segs],
            'source_doc_ids': [d['source_doc_id'] for d in segs],
            'source_dialogue_ids': [d['dialogue_id'] for d in segs],
        })
    return sessions, uses


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dialogues_dir', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--txt_root', type=Path, default=Path('data'))
    p.add_argument('--prefix', default='vn2', help='txt folders are <txt_root>/<prefix>_<split>')
    p.add_argument('--max_uses', type=int, default=5)
    p.add_argument('--eval_max_uses', type=int, default=3)
    p.add_argument('--min_segs', type=int, default=3)
    p.add_argument('--max_segs', type=int, default=8)
    p.add_argument('--same_domain_prob', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    pools = {}
    for f in sorted(args.dialogues_dir.rglob('*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        pools.setdefault(d['split'], []).append(d)

    manifest = []
    for split in sorted(pools):
        sessions, uses = assemble_split(split, pools[split], args, rng)
        sess_dir = args.out_dir / split
        txt_dir = args.txt_root / f'{args.prefix}_{split}'
        for d in (sess_dir, txt_dir):
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob('*'):
                old.unlink()
        for s in sessions:
            (sess_dir / f'session_{s["session_id"]}.json').write_text(
                json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')
            (txt_dir / f'dialogue_{s["session_id"]}.txt').write_text(
                '\n'.join(session_to_lines(s)) + '\n', encoding='utf-8')
            manifest.append({k: v for k, v in s.items() if k != 'utterances'})

        pairs = [(a, b) for s in sessions for a, b in zip(s['domain_sequence'], s['domain_sequence'][1:])]
        same = sum(a == b for a, b in pairs)
        seg_hist = Counter(s['num_sub_dialogues'] for s in sessions)
        unused = sum(1 for d in pools[split] if uses[d['dialogue_id']] == 0)
        print(f'[{split}] {len(pools[split])} D_i -> {len(sessions)} sessions in {txt_dir} | '
              f'segments/session {dict(sorted(seg_hist.items()))} | same-domain adjacent '
              f'{same}/{len(pairs)} ({same / max(1, len(pairs)):.0%}) | unused D_i {unused}')

    (args.out_dir / 'session_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
