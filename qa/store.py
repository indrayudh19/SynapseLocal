"""
qa/store.py
Append and read answers.jsonl for a session.
"""
import json
import os
import re

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


_CITE_RE = re.compile(r'\s*\[[a-zA-Z0-9_.]+\]')


def display_text(record: dict) -> str:
    """
    Return the single text to display in the UI for this record.
    Priority:
    1. Not-found message if status is not_found
    2. polished.text
    3. Older records: consolidated lead + points joined as plain sentences (citations stripped)
    4. Older records: claim texts joined as a paragraph (citations stripped)
    """
    if record.get("status") == "not_found":
        return "The documents do not appear to contain an answer to this question."

    polished = record.get("polished")
    if isinstance(polished, dict) and polished.get("text"):
        return polished["text"].strip()

    if isinstance(record.get("answer"), str) and record.get("answer").strip():
        return record["answer"].strip()

    consolidated = record.get("consolidated")
    if isinstance(consolidated, dict):
        lead = consolidated.get("lead", {})
        lead_text = lead.get("text", "") if isinstance(lead, dict) else ""
        points = consolidated.get("points", [])
        pts_text = [p.get("text", "") for p in points if isinstance(p, dict) and p.get("text")]
        combined = ([lead_text] if lead_text else []) + pts_text
        text = " ".join(t.strip() for t in combined if t.strip())
        cleaned = _CITE_RE.sub("", text)
        return re.sub(r'\s+', ' ', cleaned).strip()

    claims = record.get("claims", [])
    if claims:
        texts = [c.get("text", "").strip() for c in claims if isinstance(c, dict) and c.get("text")]
        if texts:
            cleaned = _CITE_RE.sub("", " ".join(texts))
            return re.sub(r'\s+', ' ', cleaned).strip()

    return "The documents do not appear to contain an answer to this question."
