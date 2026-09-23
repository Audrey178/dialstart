#!/usr/bin/env python3
"""Step B1/B3: turn source docs into randomized single-topic dialogue (D_i) jobs.

Each job is one LLM call that writes one D_i. Every job draws its own
turn count, fact subset, persona / forms of address, opening style and
closing, so segments do not all share one length or one opener (the old
data had every D_i at exactly 8 turns and most openers were "Chào ...",
which a position-only baseline segments perfectly).

Splits are decided per source doc, so no source doc feeds two splits:
  - --ood_domains: whole domains kept out of train; their docs alternate
    between val_ood and test_ood
  - every other domain with >= --min_docs_for_holdout docs gives one doc
    to val_in or test_in (alternating across domains), the rest go to train

Output: one JSON object per line with job_id, split, domain, source_doc_id,
the sampled settings, and the full prompt. Run the prompts with any LLM and
feed the raw replies to ingest_dialogues.py.

    python scripts/synth/make_gen_jobs.py --source_dir data/vn_synth_v2/source_docs \
        --out data/vn_synth_v2/jobs.jsonl --dialogues_per_doc 8
"""

import json
import random
import argparse
from pathlib import Path

DEFAULT_OOD_DOMAINS = ['y_te_cong', 'giao_duc', 'phong_chay_chua_chay', 'so_huu_tri_tue']

# (who the citizen is, citizen self-reference, citizen calls officer,
#  officer self-reference, officer calls citizen)
PERSONAS = [
    ('sinh viên khoảng 20 tuổi', 'em', '{o}', '{o}', 'em'),
    ('người đi làm khoảng 30 tuổi', 'em', '{o}', '{o}', 'em'),
    ('người trung niên', 'tôi', '{o}', 'tôi', '{c}'),
    ('cụ già ngoài 70 tuổi', 'bác', 'cháu', 'cháu', 'bác'),
    ('người trẻ, cán bộ lớn tuổi hơn nhiều', 'con', '{elder}', '{elder}', 'con'),
    ('chủ một cơ sở kinh doanh nhỏ', 'tôi', '{o}', 'em', '{c}'),
    ('người đi hỏi thay cho bố mẹ', 'em', '{o}', '{o}', 'em'),
    ('công nhân từ tỉnh khác lên', 'cháu', '{elder}', '{elder}', 'cháu'),
]

OPENERS = [
    ('greet', 0.20, 'Câu đầu tiên chào ngắn gọn rồi hỏi luôn.'),
    ('direct', 0.25, 'Câu đầu tiên hỏi thẳng vào việc, KHÔNG chào, KHÔNG bắt đầu bằng "Chào".'),
    ('situation', 0.25, 'Câu đầu tiên kể ngắn hoàn cảnh của mình trước (ví dụ vừa chuyển nhà, sắp đi làm, '
                        'giấy tờ bị mất...) rồi mới hỏi. Không chào.'),
    ('follow_on', 0.15, 'Câu đầu tiên mở như thể vừa hỏi xong một việc khác và chuyển sang việc này, '
                        'ví dụ "À, còn ...", "Tiện đây cho hỏi thêm ...", "Thế còn chuyện ...". Không chào.'),
    ('paperwork', 0.15, 'Câu đầu tiên nhắc tới giấy tờ đang cầm hoặc việc vừa xảy ra '
                        '(ví dụ "Tôi có tờ giấy này ...", "Hôm qua tôi nộp hồ sơ ..."). Không chào.'),
]

# Odd turn counts end on the citizen, even ones on the officer. Thanks / goodbye
# before every boundary would be as strong a cue as "Chào" after it, so keep it rare.
CITIZEN_CLOSINGS = [
    (0.35, 'Lượt cuối là công dân cảm ơn ngắn gọn.'),
    (0.35, 'Lượt cuối là công dân nhắc lại ngắn một ý vừa nghe để xác nhận, KHÔNG cảm ơn.'),
    (0.30, 'Lượt cuối là công dân nói ngắn mình sẽ làm gì tiếp theo, KHÔNG cảm ơn.'),
]
OFFICER_CLOSINGS = [
    (0.15, 'Lượt cuối của cán bộ trả lời xong thì dặn thêm một câu ngắn hoặc chào.'),
    (0.85, 'Kết thúc ngay sau câu trả lời cuối của cán bộ, KHÔNG cảm ơn, KHÔNG chào tạm biệt.'),
]

PROMPT = """Bạn sinh dữ liệu hội thoại tiếng Việt để huấn luyện mô hình phân đoạn chủ đề hội thoại.

Viết MỘT đoạn hội thoại tự nhiên giữa:
- Công dân: {persona}. Công dân tự xưng "{c_self}", gọi cán bộ là "{c_call}".
- Cán bộ {office}: xưng "{o_self}", gọi công dân là "{o_call}".

Chủ đề duy nhất: {title} (lĩnh vực: {domain}).

Thông tin nền. CHỈ dùng các thông tin dưới đây, không bịa thêm số liệu, điều kiện hay giấy tờ khác:
{facts}

Yêu cầu:
1. ĐÚNG {num_turns} lượt nói, tức mảng "utterances" có đúng {num_turns} phần tử. Lượt 1 là công dân, sau đó cán bộ và công dân xen kẽ.
2. {opener}
3. {closing}
4. Công dân hỏi dần từng ý, mỗi câu hỏi chỉ một phần thông tin; không cần hỏi hết thông tin nền.
5. Cán bộ {answer_style}{clarify}
6. Văn nói tự nhiên, không văn phong công văn, không liệt kê điều khoản, không markdown, không đánh số.
7. Chỉ nói về chủ đề trên, không chuyển sang chủ đề khác.

Chỉ trả về JSON, không thêm chữ nào khác:
{{"utterances": [{{"speaker": "citizen", "text": "..."}}, {{"speaker": "officer", "text": "..."}}]}}"""


def weighted(rng, items, weight_idx):
    r, acc = rng.random() * sum(i[weight_idx] for i in items), 0.0
    for item in items:
        acc += item[weight_idx]
        if r <= acc:
            return item
    return items[-1]


def format_value(v):
    if isinstance(v, list):
        return '; '.join(format_value(x) for x in v)
    if isinstance(v, dict):
        return '; '.join(f'{k}: {format_value(x)}' for k, x in v.items())
    return str(v)


def assign_splits(docs, ood_domains, min_docs_for_holdout, rng):
    by_domain = {}
    for d in docs:
        by_domain.setdefault(d['domain'], []).append(d)
    split_of, flip = {}, 0
    for domain in sorted(by_domain):
        ds = sorted(by_domain[domain], key=lambda d: d['doc_id'])
        if domain in ood_domains:
            for i, d in enumerate(ds):
                split_of[d['doc_id']] = 'val_ood' if i % 2 == 0 else 'test_ood'
            continue
        if len(ds) >= min_docs_for_holdout:
            held = rng.choice(ds)
            split_of[held['doc_id']] = 'val_in' if flip % 2 == 0 else 'test_in'
            flip += 1
        for d in ds:
            split_of.setdefault(d['doc_id'], 'train')
    return split_of, by_domain


def make_job(doc, split, idx, args, rng):
    num_turns = rng.randint(args.min_turns, args.max_turns)
    keys = list(doc['key_facts'])
    rng.shuffle(keys)
    n_questions = max(1, num_turns // 2)
    k = max(1, min(len(keys), n_questions + rng.randint(-1, 1)))
    facts = '\n'.join(f'- {key}: {format_value(doc["key_facts"][key])}' for key in keys[:k])

    persona = rng.choice(PERSONAS)
    o = rng.choice(['anh', 'chị'])
    c = rng.choice(['anh', 'chị'])
    elder = rng.choice(['cô', 'chú'])
    fill = lambda s: s.format(o=o, c=c, elder=elder)
    opener = weighted(rng, OPENERS, 1)
    closing = weighted(rng, CITIZEN_CLOSINGS if num_turns % 2 == 1 else OFFICER_CLOSINGS, 0)
    answer_style = rng.choice(['trả lời ngắn, thường một câu.', 'trả lời đủ ý, có khi hai ba câu.'])
    clarify = (' Có một lần cán bộ hỏi lại công dân một chi tiết về hoàn cảnh của họ trước khi trả lời.'
               if num_turns >= 6 and rng.random() < 0.3 else '')

    prompt = PROMPT.format(
        persona=persona[0], c_self=fill(persona[1]), c_call=fill(persona[2]),
        o_self=fill(persona[3]), o_call=fill(persona[4]),
        office=doc.get('agency', 'phụ trách thủ tục'), title=doc.get('title', doc['doc_id']),
        domain=doc['domain'], facts=facts, num_turns=num_turns, opener=opener[2],
        closing=closing[1], answer_style=answer_style, clarify=clarify)
    return {
        'job_id': f'{doc["doc_id"]}_{idx:02d}',
        'split': split,
        'domain': doc['domain'],
        'source_doc_id': doc['doc_id'],
        'num_turns': num_turns,
        'facts_used': keys[:k],
        'persona': persona[0],
        'opener': opener[0],
        'prompt': prompt,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source_dir', type=Path, required=True, help='<domain>/<doc_id>.json source docs')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dialogues_per_doc', type=int, default=8)
    p.add_argument('--min_turns', type=int, default=3)
    p.add_argument('--max_turns', type=int, default=12)
    p.add_argument('--ood_domains', nargs='*', default=DEFAULT_OOD_DOMAINS)
    p.add_argument('--min_docs_for_holdout', type=int, default=3,
                   help='An in-domain doc is held out for val_in/test_in only if its domain has this many docs')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    docs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(args.source_dir.rglob('*.json'))]
    split_of, by_domain = assign_splits(docs, set(args.ood_domains), args.min_docs_for_holdout, rng)

    jobs = [make_job(d, split_of[d['doc_id']], i, args, rng)
            for d in sorted(docs, key=lambda d: d['doc_id']) for i in range(args.dialogues_per_doc)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + '\n')

    counts = {}
    for d in docs:
        counts.setdefault(split_of[d['doc_id']], [0, 0])[0] += 1
    for j in jobs:
        counts[j['split']][1] += 1
    print(f'{len(docs)} source docs in {len(by_domain)} domains -> {len(jobs)} jobs written to {args.out}')
    for split in ['train', 'val_in', 'test_in', 'val_ood', 'test_ood']:
        n_docs, n_jobs = counts.get(split, [0, 0])
        print(f'  {split:9s} {n_docs:4d} docs {n_jobs:5d} jobs')
    thin = [dom for dom, ds in by_domain.items()
            if dom not in args.ood_domains and len(ds) < args.min_docs_for_holdout]
    if thin:
        print(f'  {len(thin)} train domains have < {args.min_docs_for_holdout} docs and give nothing '
              f'to val_in/test_in: {", ".join(sorted(thin))}')
    if 'val_in' not in counts or 'val_ood' not in counts:
        print('  WARNING: val_in or val_ood is empty, collect more source docs first')


if __name__ == '__main__':
    main()
