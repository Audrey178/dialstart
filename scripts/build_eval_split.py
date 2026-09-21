#!/usr/bin/env python3
"""
Lắp ráp val/test session cho vn_synth từ 2 domain giữ riêng cho eval
(y_te_cong, giao_duc), tách biệt hoàn toàn khỏi 2500 session train
(within_limit) hiện có — không share domain nào với train.

Mỗi domain có 8 D_i: 4 D_i đầu (0001-0004) -> val pool, 4 D_i sau
(0005-0008) -> test pool, để val và test cũng không share D_i với nhau.

Sau khi lắp ráp dư (oversample), subsample theo bucket token-length
(quartile đo từ 2500 session train) để val/test có phân phối độ dài
khớp với train.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_SCRIPTS = REPO_ROOT / ".claude/skills/vn-meeting-dialogue-synth/scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from assemble_sessions import (  # noqa: E402
    assemble_session_to_token_target,
    group_by_domain,
    estimate_token_count_utterances,
)

SINGLE_TOPIC_DIR = REPO_ROOT / "data/vn_synth/single_topic_dialogues"
TRAIN_MANIFEST = REPO_ROOT / "data/vn_synth/sessions/session_manifest.json"
OUTPUT_ROOT = REPO_ROOT / "data/vn_synth/sessions"

EVAL_DOMAINS = ["y_te_cong", "giao_duc"]
VAL_DIALOGUE_SUFFIXES = {"0001", "0002", "0003", "0004"}
TEST_DIALOGUE_SUFFIXES = {"0005", "0006", "0007", "0008"}

TARGET_SIZE = 250
OVERSAMPLE_SIZE = 600
SUB_DIALOGUE_RANGE = (3, 5)


def load_pool_dialogues(dialogue_suffixes: set[str]) -> list[dict]:
    """
    Đọc các D_i thuộc EVAL_DOMAINS mà dialogue_id có suffix nằm trong
    dialogue_suffixes (vd {"0001",...,"0004"} cho val pool).

    Đầu ra: danh sách dict D_i, có thêm "_token_count".
    """
    dialogues = []
    for domain in EVAL_DOMAINS:
        domain_dir = SINGLE_TOPIC_DIR / domain
        for json_path in sorted(domain_dir.glob("*.json")):
            d = json.loads(json_path.read_text(encoding="utf-8"))
            suffix = d["dialogue_id"].split("_")[-1]
            if suffix in dialogue_suffixes:
                d["_token_count"] = estimate_token_count_utterances(d["utterances"])
                dialogues.append(d)
    return dialogues


def load_train_token_quartiles() -> list[int]:
    """
    Đọc estimated_tokens của 2500 session train (bucket within_limit) từ
    manifest hiện có, trả về 3 điểm chia quartile [q25, q50, q75].
    """
    manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    tokens = sorted(
        m["estimated_tokens"] for m in manifest if m["bucket"] == "within_limit"
    )
    n = len(tokens)
    return [tokens[int(p * n)] for p in (0.25, 0.50, 0.75)]


def bucket_of(token_count: int, quartiles: list[int]) -> int:
    """Gán session vào 1 trong 4 bucket theo quartile của train (0..3)."""
    q25, q50, q75 = quartiles
    if token_count <= q25:
        return 0
    if token_count <= q50:
        return 1
    if token_count <= q75:
        return 2
    return 3


def oversample_sessions(
    pool_dialogues: list[dict], split_name: str, seed: int
) -> list[dict]:
    """Lắp ráp OVERSAMPLE_SIZE session ứng viên từ pool_dialogues."""
    by_domain = group_by_domain(pool_dialogues)
    rng = random.Random(seed)
    sessions = []
    for i in range(OVERSAMPLE_SIZE):
        session_id = f"{split_name}_candidate_{i:04d}"
        session = assemble_session_to_token_target(by_domain, session_id, None, rng)
        sessions.append(session)
    return sessions


def stratified_subsample(
    candidates: list[dict], quartiles: list[int], target_size: int, seed: int
) -> list[dict]:
    """
    Chọn target_size session từ candidates sao cho tỷ lệ mỗi bucket
    token-length bằng nhau (~target_size/4 mỗi bucket), khớp tỷ lệ đều
    của chính train theo định nghĩa quartile.
    """
    rng = random.Random(seed)
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for s in candidates:
        by_bucket[bucket_of(s["estimated_tokens"], quartiles)].append(s)

    per_bucket_target = target_size // 4
    selected = []
    for bucket_id in range(4):
        available = by_bucket[bucket_id]
        rng.shuffle(available)
        take = min(per_bucket_target, len(available))
        if take < per_bucket_target:
            print(
                f"CẢNH BÁO: bucket {bucket_id} chỉ có {len(available)} ứng viên, "
                f"cần {per_bucket_target} — thiếu {per_bucket_target - take}."
            )
        selected.extend(available[:take])
    return selected


def build_split(split_name: str, dialogue_suffixes: set[str], quartiles: list[int], seed: int) -> list[dict]:
    pool = load_pool_dialogues(dialogue_suffixes)
    print(f"[{split_name}] pool D_i: {len(pool)} ({[d['dialogue_id'] for d in pool]})")
    candidates = oversample_sessions(pool, split_name, seed)
    selected = stratified_subsample(candidates, quartiles, TARGET_SIZE, seed + 1)
    for idx, s in enumerate(selected):
        s["session_id"] = f"{split_name}_{idx:04d}"
        s["bucket"] = split_name
    return selected


def write_sessions(sessions: list[dict], split_name: str) -> None:
    out_dir = OUTPUT_ROOT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in sessions:
        out_path = out_dir / f"session_{s['session_id']}.json"
        out_path.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{split_name}] đã ghi {len(sessions)} session vào {out_dir}")


def main() -> None:
    quartiles = load_train_token_quartiles()
    print(f"Quartile token train (within_limit): {quartiles}")

    val_sessions = build_split("val", VAL_DIALOGUE_SUFFIXES, quartiles, seed=123)
    test_sessions = build_split("test", TEST_DIALOGUE_SUFFIXES, quartiles, seed=456)

    write_sessions(val_sessions, "val")
    write_sessions(test_sessions, "test")

    manifest_entries = []
    for split_name, sessions in [("val", val_sessions), ("test", test_sessions)]:
        for s in sessions:
            manifest_entries.append(
                {
                    "session_id": s["session_id"],
                    "bucket": split_name,
                    "num_sub_dialogues": s["num_sub_dialogues"],
                    "num_turns": s["num_turns"],
                    "num_boundaries": len(s["gold_boundaries"]),
                    "estimated_tokens": s["estimated_tokens"],
                    "reused_dialogue_count": s["reused_dialogue_count"],
                }
            )
    manifest_path = OUTPUT_ROOT / "eval_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã ghi manifest eval split: {manifest_path}")


if __name__ == "__main__":
    main()
