"""
qa/understand.py
Q1: Understand — analyse the user's question with Qwen 2.5 3B via Ollama.
Runs in the Streamlit process (HTTP only, no torch).
"""
import json
import os
import re

import ollama

from qa.config import QA_MODEL, NUM_CTX_UNDERSTAND, NUM_PREDICT_UNDERSTAND
from qa.schema import Understanding
from qa.text_utils import content_tokens
from qa import paths as qa_paths


_SYSTEM_PROMPT = (
    "You analyze a user's question about a document collection so a search engine "
    "can find the answer.\n"
    "Output JSON only. Do NOT answer the question.\n"
    "Fields:\n"
    "- intent: definition | fact | list | comparison | procedure | explanation | numeric | yes_no | summary | other\n"
    "- answer_type: span (short phrase) | list | paragraph | number | boolean\n"
    "- focus: the main thing asked about, at most 6 words\n"
    "- key_terms: up to 6 exact terms or phrases likely to appear in the answer text; "
    "keep acronyms and add the expansion if you know it\n"
    "- queries: up to 3 standalone rewrites of the question for semantic search "
    "(different wording, no pronouns)\n"
    "- hypothetical_answer: one sentence (max 40 words) written the way a textbook "
    "would state the answer\n"
    "- sub_questions: only for comparison or multi-part questions, otherwise []\n"
    '- doc_hint: "slides", "notes", a file name fragment, or null; set only if the '
    "question says where to look"
)

_FEW_SHOT_USER = (
    'Question: How are B-tree indexes different from hash indexes in my notes?'
)
_FEW_SHOT_ASSISTANT = json.dumps({
    "intent": "comparison",
    "answer_type": "paragraph",
    "focus": "B-tree vs hash index",
    "key_terms": ["B-tree index", "hash index", "range query", "equality lookup"],
    "queries": [
        "differences between B-tree and hash indexes",
        "when to use a hash index instead of a B-tree",
        "B-tree range queries versus hash index equality lookups",
    ],
    "hypothetical_answer": (
        "B-tree indexes keep keys ordered and support range queries, "
        "while hash indexes support only equality lookups but are faster for them."
    ),
    "sub_questions": ["How do B-tree indexes work?", "How do hash indexes work?"],
    "doc_hint": "notes",
}, ensure_ascii=False)


def _cap(s: str, maxlen: int) -> str:
    return s[:maxlen].strip() if s else ""


def _resolve_doc_hint(hint: str | None, chunk_files: set[str]) -> set[str]:
    """Resolve doc_hint to a set of source_file values."""
    if not hint:
        return set()
    hint_lower = hint.lower().strip()

    # Pattern-based resolution
    slides_re = re.compile(r'slides?|presentation|ppt', re.IGNORECASE)
    notes_re = re.compile(r'notes?|pdf|paper', re.IGNORECASE)

    matched = set()
    if slides_re.search(hint_lower):
        for f in chunk_files:
            if f.lower().endswith('.pptx'):
                matched.add(f)
    elif notes_re.search(hint_lower):
        for f in chunk_files:
            if f.lower().endswith('.pdf'):
                matched.add(f)

    if matched:
        return matched

    # Fuzzy match against file names
    for f in chunk_files:
        if hint_lower in f.lower():
            matched.add(f)

    return matched


def run(session_id: str, run_id: str, question: str,
        chunk_files: set[str] | None = None) -> dict:
    """
    Q1: Understand the question.
    Returns the understanding dict (also written to understanding.json).
    """
    try:
        response = ollama.chat(
            model=QA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _FEW_SHOT_USER},
                {"role": "assistant", "content": _FEW_SHOT_ASSISTANT},
                {"role": "user", "content": f"Question: {question}"},
            ],
            format=Understanding.model_json_schema(),
            options={
                "temperature": 0,
                "seed": 42,
                "num_ctx": NUM_CTX_UNDERSTAND,
                "num_predict": NUM_PREDICT_UNDERSTAND,
            },
            keep_alive="0",
        )

        raw_text = response["message"]["content"]
        parsed = None

        # Try parsing
        for attempt in range(2):
            try:
                data = json.loads(raw_text)
                parsed = Understanding(**data)
                break
            except Exception:
                if attempt == 0:
                    # Retry: ask the model again
                    try:
                        response2 = ollama.chat(
                            model=QA_MODEL,
                            messages=[
                                {"role": "system", "content": _SYSTEM_PROMPT},
                                {"role": "user", "content": f"Question: {question}"},
                            ],
                            format=Understanding.model_json_schema(),
                            options={
                                "temperature": 0,
                                "seed": 42,
                                "num_ctx": NUM_CTX_UNDERSTAND,
                                "num_predict": NUM_PREDICT_UNDERSTAND,
                            },
                            keep_alive="0",
                        )
                        raw_text = response2["message"]["content"]
                    except Exception:
                        break

        # Fallback if parsing failed
        if parsed is None:
            fallback_terms = content_tokens(question)[:6]
            parsed = Understanding(
                intent="other",
                answer_type="paragraph",
                focus=question[:40],
                key_terms=fallback_terms if fallback_terms else [question[:60]],
                queries=[],
                hypothetical_answer="",
                sub_questions=[],
                doc_hint=None,
            )

        # Deterministic cleanup
        result = parsed.model_dump()

        # Strip and deduplicate key_terms, cap lengths
        seen_terms = set()
        clean_terms = []
        for t in result["key_terms"]:
            t = _cap(t, 60)
            if t and t.lower() not in seen_terms:
                seen_terms.add(t.lower())
                clean_terms.append(t)
        result["key_terms"] = clean_terms[:6]

        # If key_terms empty after cleanup, fall back
        if not result["key_terms"]:
            result["key_terms"] = content_tokens(question)[:6] or [question[:60]]

        # Cap and deduplicate queries
        seen_q = set()
        clean_queries = []
        for q in result["queries"]:
            q = _cap(q, 160)
            if q and q.lower() not in seen_q:
                seen_q.add(q.lower())
                clean_queries.append(q)
        result["queries"] = clean_queries[:3]

        # Cap hypothetical_answer
        result["hypothetical_answer"] = _cap(result["hypothetical_answer"], 300)
        result["focus"] = _cap(result["focus"], 50)

        # Prepend original question to query list
        result["queries"] = [question] + result["queries"]

        # Resolve doc_hint to source files
        hinted_files: set[str] = set()
        if chunk_files and result.get("doc_hint"):
            hinted_files = _resolve_doc_hint(result["doc_hint"], chunk_files)
        result["_hinted_files"] = list(hinted_files)

        # Write output
        out_path = qa_paths.understanding_path(session_id, run_id)
        if not qa_paths.session_exists(session_id):
            raise FileNotFoundError(f"Session {session_id} no longer exists")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    finally:
        # Unload the model
        try:
            ollama.generate(model=QA_MODEL, prompt="", keep_alive=0)
        except Exception:
            pass
