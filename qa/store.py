"""
qa/store.py
Append and read answers.jsonl for a session.
"""
import json
import os

from qa import paths as qa_paths


def append_answer(session_id: str, record: dict):
    """Append a single answer record to answers.jsonl (atomic single write + flush)."""
    if not qa_paths.session_exists(session_id):
        return  # Session was deleted
    qa_d = qa_paths.qa_dir(session_id)
    os.makedirs(qa_d, exist_ok=True)
    path = qa_paths.answers_jsonl_path(session_id)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def load_answers(session_id: str) -> list[dict]:
    """Load all answers from answers.jsonl, skipping malformed lines."""
    path = qa_paths.answers_jsonl_path(session_id)
    if not os.path.exists(path):
        return []
    answers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                answers.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return answers
