#!/usr/bin/env python3
"""Validate raw LLM replies for make_gen_jobs.py jobs and write accepted D_i.

Replies come as JSONL lines {"job_id": ..., "output": "<raw reply text>"}
(--outputs, repeatable), or as <job_id>.json / <job_id>.txt files in
--outputs_dir. A reply is accepted when it parses to {"utterances": [...]},
speakers alternate starting with the citizen, the turn count is within
--turn_tolerance of what the job asked for (LLMs drift toward 8), there is
no markdown / numbering, and no earlier accepted D_i has the same text.

Accepted D_i go to <out_dir>/<domain>/<dialogue_id>.json and keep the job's
split, so assemble_sessions.py never mixes splits. Rejected jobs are written
to <out_dir>/../jobs_retry.jsonl for another round.

    python scripts/synth/ingest_dialogues.py --jobs data/vn_synth_v2/jobs.jsonl \
        --outputs data/vn_synth_v2/outputs.jsonl --out_dir data/vn_synth_v2/single_topic_dialogues
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter

MARKDOWN = re.compile(r'(\*\*|^#+\s|^\s*[-*•]\s|^\s*\d+[.)]\s)', re.M)


def parse_reply(text):
    text = text.strip()
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < 0:
        raise ValueError('no_json')
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        raise ValueError('bad_json')
    utts = obj.get('utterances') if isinstance(obj, dict) else None
    if not isinstance(utts, list) or not utts:
        raise ValueError('no_utterances')
    return utts


def validate(utts, job, tolerance):
    for i, u in enumerate(utts):
        if not isinstance(u, dict) or not str(u.get('text', '')).strip():
            return 'empty_turn'
        if u.get('speaker') != ('citizen' if i % 2 == 0 else 'officer'):
            return 'speaker_order'
        if MARKDOWN.search(u['text']):
            return 'markdown'
    if abs(len(utts) - job['num_turns']) > tolerance:
        return 'turn_count'
    return None


def load_replies(args):
    replies = {}
    for path in args.outputs or []:
        for line in open(path, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                replies[r['job_id']] = r['output']
    if args.outputs_dir:
        for f in sorted(args.outputs_dir.iterdir()):
            if f.suffix in ('.json', '.txt'):
                replies[f.stem] = f.read_text(encoding='utf-8')
    return replies


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jobs', type=Path, required=True)
    p.add_argument('--outputs', type=Path, action='append')
    p.add_argument('--outputs_dir', type=Path)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--turn_tolerance', type=int, default=1)
    args = p.parse_args()

    jobs = [json.loads(l) for l in open(args.jobs, encoding='utf-8') if l.strip()]
    replies = load_replies(args)

    seen = set()
    for f in args.out_dir.rglob('*.json') if args.out_dir.exists() else []:
        seen.add(tuple(u['text'].strip() for u in json.loads(f.read_text(encoding='utf-8'))['utterances']))

    reasons, drift, retry, accepted = Counter(), Counter(), [], Counter()
    for job in jobs:
        dialogue_id = f'd_{job["job_id"]}'
        out_path = args.out_dir / job['domain'] / f'{dialogue_id}.json'
        if out_path.exists():
            reasons['already_ingested'] += 1
            continue
        if job['job_id'] not in replies:
            reasons['missing_reply'] += 1
            retry.append(job)
            continue
        try:
            utts = parse_reply(replies[job['job_id']])
            reason = validate(utts, job, args.turn_tolerance)
        except ValueError as e:
            utts, reason = None, str(e)
        if utts is not None:
            drift[len(utts) - job['num_turns']] += 1
        if reason is None:
            key = tuple(u['text'].strip() for u in utts)
            reason = 'duplicate' if key in seen else None
            seen.add(key)
        if reason:
            reasons[reason] += 1
            retry.append(job)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            'dialogue_id': dialogue_id,
            'domain': job['domain'],
            'source_doc_id': job['source_doc_id'],
            'split': job['split'],
            'target_turns': job['num_turns'],
            'persona': job['persona'],
            'opener': job['opener'],
            'utterances': [{'speaker': u['speaker'], 'text': u['text'].strip()} for u in utts],
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        accepted[job['split']] += 1
        reasons['accepted'] += 1

    retry_path = args.out_dir.parent / 'jobs_retry.jsonl'
    with open(retry_path, 'w', encoding='utf-8') as f:
        for job in retry:
            f.write(json.dumps(job, ensure_ascii=False) + '\n')

    print(f'{len(jobs)} jobs, {len(replies)} replies')
    print('outcome: ' + ', '.join(f'{k} {v}' for k, v in reasons.most_common()))
    print('accepted by split: ' + ', '.join(f'{k} {v}' for k, v in sorted(accepted.items())))
    print('turns got - asked: ' + ', '.join(f'{k:+d}: {v}' for k, v in sorted(drift.items())))
    print(f'{len(retry)} jobs to retry -> {retry_path}')


if __name__ == '__main__':
    main()
