# vn_synth v2 pipeline

Regenerates the synthetic Vietnamese segmentation data so that it cannot be
solved by memorising a handful of segments or by counting positions. Every
step after 2 is deterministic and free; steps 1 and 2 need an LLM.

| Step | What | Tool |
|---|---|---|
| 1 | Collect 5-6 source docs per domain into `data/vn_synth_v2/source_docs/<domain>/<doc_id>.json` (same schema as `data/vn_synth/source_docs`) | web search + LLM |
| 2a | Build one randomized prompt per D_i, split by source doc | `make_gen_jobs.py` |
| 2b | Run every `prompt` in `jobs.jsonl` through an LLM, save `{"job_id", "output"}` lines to `outputs.jsonl` | any LLM |
| 2c | Validate replies, write D_i; failed jobs go to `jobs_retry.jsonl` (rerun 2b on it, then 2c again) | `ingest_dialogues.py` |
| 3 | Assemble sessions per split and write `data/vn2_<split>/*.txt` | `assemble_sessions.py` |
| 4 | Gate: must print `0 FAIL` before preprocessing / training | `../check_data_stats.py` |

```bash
V=data/vn_synth_v2
python scripts/synth/make_gen_jobs.py --source_dir $V/source_docs --out $V/jobs.jsonl
# ... run the LLM on $V/jobs.jsonl -> $V/outputs.jsonl ...
python scripts/synth/ingest_dialogues.py --jobs $V/jobs.jsonl --outputs $V/outputs.jsonl \
    --out_dir $V/single_topic_dialogues
python scripts/synth/assemble_sessions.py --dialogues_dir $V/single_topic_dialogues \
    --out_dir $V/sessions --txt_root data --prefix vn2
python scripts/check_data_stats.py --train vn2_train \
    --eval vn2_val_in vn2_test_in vn2_val_ood vn2_test_ood
```

Splits: `y_te_cong`, `giao_duc`, `phong_chay_chua_chay`, `so_huu_tri_tue` are
out-of-domain (`val_ood` / `test_ood`). Every other domain with at least 3
docs gives one doc to `val_in` or `test_in`, the rest go to `train`. No
source doc feeds two splits.
