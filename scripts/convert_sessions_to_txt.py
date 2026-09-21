#!/usr/bin/env python3
"""
Convert session JSON (vn_synth format: utterances + gold_boundaries) sang
định dạng .txt mà data_preprocess.py đang đọc cho dialseg711/doc2dial:
mỗi dòng 1 utterance (bỏ nhãn speaker), ranh giới đánh dấu bằng 1 dòng
"================".

Mỗi split (train/val/test) -> 1 thư mục data/vn_synth_<split>/, mỗi
session -> 1 file .txt.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_ROOT = REPO_ROOT / "data/vn_synth/sessions"
BOUNDARY_MARKER = "================"

SPLIT_TO_SESSION_DIR = {
    "train": SESSIONS_ROOT / "within_limit",
    "val": SESSIONS_ROOT / "val",
    "test": SESSIONS_ROOT / "test",
}


def session_to_lines(session: dict) -> list[str]:
    """Chuyển 1 session JSON thành danh sách dòng .txt (utterance + marker)."""
    boundary_set = set(session["gold_boundaries"])
    lines = []
    for idx, utt in enumerate(session["utterances"]):
        lines.append(utt["text"])
        if idx in boundary_set:
            lines.append(BOUNDARY_MARKER)
    return lines


def convert_split(split_name: str, session_dir: Path) -> int:
    out_dir = REPO_ROOT / "data" / f"vn_synth_{split_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for json_path in sorted(session_dir.glob("*.json")):
        session = json.loads(json_path.read_text(encoding="utf-8"))
        lines = session_to_lines(session)
        out_path = out_dir / f"dialogue_{session['session_id']}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        count += 1
    return count


def main() -> None:
    for split_name, session_dir in SPLIT_TO_SESSION_DIR.items():
        n = convert_split(split_name, session_dir)
        print(f"[{split_name}] {n} file .txt -> data/vn_synth_{split_name}/")


if __name__ == "__main__":
    main()
