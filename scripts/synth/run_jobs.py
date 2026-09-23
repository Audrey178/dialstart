#!/usr/bin/env python3
"""Step 2b: run make_gen_jobs.py prompts through an LLM, append raw replies to JSONL.

Each output line is {"job_id", "output", "model", "stop_reason", "usage"},
which ingest_dialogues.py reads. Jobs already in --out are skipped, so an
interrupted run resumes where it stopped; pass jobs_retry.jsonl as --jobs
(with a new --out, or the same one) to redo the ones ingest rejected.

Backends
  anthropic  Claude via the Anthropic SDK (ANTHROPIC_API_KEY or `ant auth login`).
             Default: concurrent requests with server-side refusal fallback.
             --batch: Message Batches API, 50% cheaper, usually done within
             an hour; no fallback there, refused jobs just come back as retries.
  openai     Any OpenAI-compatible endpoint (--base_url), e.g. a vLLM server
             on the H200:  vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 1
             then --backend openai --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-72B-Instruct

    python scripts/synth/run_jobs.py --jobs data/vn_synth_v2/jobs.jsonl \
        --out data/vn_synth_v2/outputs.jsonl --backend anthropic --batch --limit 20
"""

import sys
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_jobs(args):
    jobs = [json.loads(l) for l in open(args.jobs, encoding='utf-8') if l.strip()]
    done = set()
    if args.out.exists():
        done = {json.loads(l)['job_id'] for l in open(args.out, encoding='utf-8') if l.strip()}
    todo = [j for j in jobs if j['job_id'] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f'{len(jobs)} jobs, {len(done)} already in {args.out}, running {len(todo)}')
    return todo


class Writer:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, 'a', encoding='utf-8')
        self.lock = threading.Lock()
        self.ok = self.failed = 0

    def write(self, job_id, text, model, stop_reason, usage):
        with self.lock:
            self.f.write(json.dumps({'job_id': job_id, 'output': text, 'model': model,
                                     'stop_reason': stop_reason, 'usage': usage}, ensure_ascii=False) + '\n')
            self.f.flush()
            self.ok += 1

    def fail(self, job_id, why):
        with self.lock:
            self.failed += 1
            print(f'  [{job_id}] {why}', file=sys.stderr)


def text_of(content):
    return ''.join(b.text for b in content if b.type == 'text')


# ---------------------------------------------------------------- anthropic

def anthropic_params(job, args):
    return {
        'model': args.model,
        'max_tokens': args.max_tokens,
        'thinking': {'type': 'adaptive'},
        'output_config': {'effort': args.effort},
        'messages': [{'role': 'user', 'content': job['prompt']}],
    }


def run_anthropic_concurrent(jobs, args, writer):
    import anthropic
    client = anthropic.Anthropic(max_retries=6)

    def one(job):
        try:
            msg = client.beta.messages.create(
                **anthropic_params(job, args),
                betas=['server-side-fallback-2026-07-01'],
                fallbacks='default',
            )
        except anthropic.BadRequestError as e:
            return writer.fail(job['job_id'], f'bad request: {e.message}')
        except anthropic.APIStatusError as e:
            return writer.fail(job['job_id'], f'API error {e.status_code}: {e.message}')
        except anthropic.APIConnectionError as e:
            return writer.fail(job['job_id'], f'connection error: {e}')
        if msg.stop_reason == 'refusal':
            return writer.fail(job['job_id'], 'refused by every model in the fallback chain')
        if msg.stop_reason == 'max_tokens':
            return writer.fail(job['job_id'], 'hit max_tokens')
        writer.write(job['job_id'], text_of(msg.content), msg.model, msg.stop_reason, msg.usage.to_dict())

    with ThreadPoolExecutor(args.concurrency) as pool:
        for i, _ in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done ({writer.ok} ok, {writer.failed} failed)')


def run_anthropic_batch(jobs, args, writer):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()

    state = args.out.with_suffix('.batch_id')
    if args.batch_id:
        batch_id = args.batch_id
    else:
        batch = client.messages.batches.create(requests=[
            Request(custom_id=j['job_id'], params=MessageCreateParamsNonStreaming(**anthropic_params(j, args)))
            for j in jobs])
        batch_id = batch.id
        state.write_text(batch_id)
        print(f'batch {batch_id} submitted ({len(jobs)} requests); id saved to {state}. '
              f'If this process stops, resume with --batch_id {batch_id}')

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f'  {batch.processing_status}: processing {c.processing}, succeeded {c.succeeded}, '
              f'errored {c.errored}, expired {c.expired}')
        if batch.processing_status == 'ended':
            break
        time.sleep(args.poll)

    for r in client.messages.batches.results(batch_id):
        if r.result.type != 'succeeded':
            writer.fail(r.custom_id, f'batch result {r.result.type}')
            continue
        msg = r.result.message
        if msg.stop_reason in ('refusal', 'max_tokens'):
            writer.fail(r.custom_id, f'stop_reason {msg.stop_reason}')
            continue
        writer.write(r.custom_id, text_of(msg.content), msg.model, msg.stop_reason, msg.usage.to_dict())


# ------------------------------------------------------------------- openai

def run_openai(jobs, args, writer):
    import openai
    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key or 'EMPTY', max_retries=6)

    def one(job):
        try:
            r = client.chat.completions.create(
                model=args.model, max_tokens=args.max_tokens, temperature=args.temperature,
                messages=[{'role': 'user', 'content': job['prompt']}])
        except openai.APIStatusError as e:
            return writer.fail(job['job_id'], f'API error {e.status_code}: {e.message}')
        except openai.APIConnectionError as e:
            return writer.fail(job['job_id'], f'connection error: {e}')
        choice = r.choices[0]
        if choice.finish_reason == 'length':
            return writer.fail(job['job_id'], 'hit max_tokens')
        writer.write(job['job_id'], choice.message.content or '', r.model, choice.finish_reason,
                     r.usage.model_dump() if r.usage else None)

    with ThreadPoolExecutor(args.concurrency) as pool:
        for i, _ in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done ({writer.ok} ok, {writer.failed} failed)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--jobs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', choices=['anthropic', 'openai'], default='anthropic')
    p.add_argument('--model', default=None, help='Default: claude-opus-5 for anthropic; required for openai')
    p.add_argument('--limit', type=int, default=None, help='Only run the first N pending jobs (pilot)')
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--max_tokens', type=int, default=16000)
    p.add_argument('--effort', default='low', choices=['low', 'medium', 'high', 'xhigh', 'max'],
                   help='anthropic only: thinking depth / token spend')
    p.add_argument('--batch', action='store_true', help='anthropic only: use the Message Batches API (50%% off)')
    p.add_argument('--batch_id', help='anthropic --batch: resume polling an already submitted batch')
    p.add_argument('--poll', type=int, default=60, help='Seconds between batch status checks')
    p.add_argument('--base_url', help='openai only, e.g. http://localhost:8000/v1')
    p.add_argument('--api_key', help='openai only; defaults to OPENAI_API_KEY, or EMPTY for a local vLLM')
    p.add_argument('--temperature', type=float, default=0.9, help='openai only')
    args = p.parse_args()

    if args.backend == 'anthropic':
        args.model = args.model or 'claude-opus-5'
    elif not args.model:
        p.error('--model is required with --backend openai')
    if args.backend == 'openai' and args.api_key is None:
        import os
        args.api_key = os.environ.get('OPENAI_API_KEY')

    jobs = [] if args.batch_id else load_jobs(args)
    if not jobs and not args.batch_id:
        return
    writer = Writer(args.out)
    if args.backend == 'openai':
        run_openai(jobs, args, writer)
    elif args.batch or args.batch_id:
        run_anthropic_batch(jobs, args, writer)
    else:
        run_anthropic_concurrent(jobs, args, writer)
    print(f'{writer.ok} replies appended to {args.out}, {writer.failed} failed '
          f'(failed jobs stay pending: rerun the same command, or ingest and use jobs_retry.jsonl)')


if __name__ == '__main__':
    main()
