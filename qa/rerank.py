"""
qa/rerank.py
Q3: LLM Semantic Filter — Qwen determines which retrieved passages are
actually relevant to the question, removes irrelevant material, and builds
evidence windows.
No cross-encoder or sentence-transformers needed.
"""
import json
import os

import ollama

from qa.config import (
    QA_MODEL, NUM_CTX_FILTER, NUM_PREDICT_FILTER,
    LLM_FILTER_MAX_PASSAGES,
    K_PASSAGES, K_PASSAGES_COMPARE,
    PARENT_MAX_CHARS, WINDOW_MAX_CHARS,
)
from qa import paths as qa_paths
from qa.text_utils import split_sentences


def _load_parent_text(session_id: str, parent_id: str, chunks: list[dict]) -> str:
    """
    Reconstruct parent text by concatenating children with the same parent_id in order.
    """
    children = [c for c in chunks if c.get("parent_id") == parent_id]
    # Children should already be in order from chunks.jsonl
    texts = [c.get("text", "") for c in children]
    combined = "\n".join(texts)
    return combined[:PARENT_MAX_CHARS]


def _llm_filter_passages(question: str, candidates: list[dict]) -> list[dict]:
    """
    Use Qwen to rate each candidate passage as relevant or irrelevant.
    Returns candidates annotated with a 'relevant' boolean.
    Each passage is evaluated individually for reliability.
    """
    rated = []
    for c in candidates:
        heading = c.get("heading", "")
        text = c.get("text", "")
        doc_text = (heading + " " + text).strip() if heading else text
        # Truncate very long passages to keep context manageable
        doc_text = doc_text[:800]

        prompt = (
            'Decide whether the PASSAGE is relevant to the QUESTION.\n'
            'Answer {"relevant": true} if the passage contains information that '
            'helps answer the question, even partially.\n'
            'Answer {"relevant": false} if the passage is completely unrelated.\n'
            f'QUESTION: {question}\n'
            f'PASSAGE:\n{doc_text}'
        )

        try:
            response = ollama.chat(
                model=QA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format={
                    "type": "object",
                    "properties": {"relevant": {"type": "boolean"}},
                    "required": ["relevant"],
                },
                options={
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": NUM_CTX_FILTER,
                    "num_predict": NUM_PREDICT_FILTER,
                },
                keep_alive="60s",
            )
            verdict = json.loads(response["message"]["content"])
            is_relevant = verdict.get("relevant", True)
        except Exception:
            # On error, keep the passage (fail open)
            is_relevant = True

        c_copy = dict(c)
        c_copy["llm_relevant"] = is_relevant
        rated.append(c_copy)

    return rated


def _position_scored_window(sentences: list[str], max_chars: int) -> list[dict]:
    """
    Build a sentence window using position-based heuristic scoring.
    Earlier sentences score higher (they tend to contain topic sentences).
    Returns a list of sentence dicts with sid and text.
    """
    if not sentences:
        return []

    # Score: earlier sentences get higher scores
    n = len(sentences)
    sent_entries = []
    for i, s in enumerate(sentences):
        score = max(0.0, 1.0 - (i / max(n, 1)) * 0.5)
        sent_entries.append({
            "sid": f"0.{i+1}",  # placeholder sid, will be re-assigned later
            "text": s,
            "score": round(score, 4),
        })

    # Greedy window: find the best contiguous block within max_chars
    best_score = -float("inf")
    best_start, best_end = 0, 0

    for start in range(n):
        total_chars = 0
        total_score = 0.0
        for end in range(start, n):
            total_chars += len(sent_entries[end]["text"])
            sc = sent_entries[end]["score"]
            if sc > 0:
                total_score += sc
            if total_chars > max_chars:
                break
            if total_score > best_score:
                best_score = total_score
                best_start = start
                best_end = end

    return sent_entries[best_start:best_end + 1]


def run(session_id: str, run_id: str) -> dict:
    """
    Q3: LLM Semantic Filter + evidence window construction.
    Reads candidates.json and understanding.json, writes reranked.json.
    Uses Qwen to filter irrelevant passages instead of a cross-encoder.
    """
    # Load inputs
    c_path = qa_paths.candidates_path(session_id, run_id)
    u_path = qa_paths.understanding_path(session_id, run_id)

    with open(c_path, "r", encoding="utf-8") as f:
        cand_data = json.load(f)
    with open(u_path, "r", encoding="utf-8") as f:
        understanding = json.load(f)

    candidates = cand_data.get("candidates", [])
    if not candidates:
        result = {"status": "not_found", "closest": []}
        out_path = qa_paths.reranked_path(session_id, run_id)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    question = understanding.get("queries", [""])[0] if understanding.get("queries") else ""
    intent = understanding.get("intent", "other")

    # ── LLM semantic filter ─────────────────────────────────
    rated = _llm_filter_passages(question, candidates)

    # Separate relevant and irrelevant
    relevant = [c for c in rated if c.get("llm_relevant", True)]
    irrelevant = [c for c in rated if not c.get("llm_relevant", True)]

    # Answerability gate: if no passages are relevant, return not_found
    if not relevant:
        result = {
            "status": "not_found",
            "closest": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "source_file": c.get("source_file"),
                    "heading": c.get("heading"),
                    "text": c.get("text", "")[:300],
                    "retrieval_score": c.get("retrieval_score", 0),
                }
                for c in candidates[:3]
            ],
        }
        out_path = qa_paths.reranked_path(session_id, run_id)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # ── Select passages (parent-deduped) ────────────────────
    k_pass = K_PASSAGES_COMPARE if intent == "comparison" else K_PASSAGES
    seen_parents = set()
    selected = []
    for c in relevant:
        pid = c.get("parent_id", c.get("chunk_id"))
        if pid in seen_parents:
            continue
        seen_parents.add(pid)
        selected.append(c)
        if len(selected) >= min(k_pass, LLM_FILTER_MAX_PASSAGES):
            break

    # Per-document coverage: include best candidate from each source file
    if len(selected) > 0:
        source_files_in_selected = {c.get("source_file") for c in selected}
        all_source_files = {c.get("source_file") for c in relevant}

        if len(all_source_files) >= 2:
            for c in relevant:
                sf = c.get("source_file")
                pid = c.get("parent_id", c.get("chunk_id"))
                if sf not in source_files_in_selected and pid not in seen_parents:
                    selected.append(c)
                    seen_parents.add(pid)
                    source_files_in_selected.add(sf)

    # Cap to max
    selected = selected[:LLM_FILTER_MAX_PASSAGES]

    # ── Load all chunks for parent text reconstruction ──────
    all_chunks = []
    chunks_path = qa_paths.chunks_jsonl_path(session_id)
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))

    # ── Build passage windows ───────────────────────────────
    passages = []
    for rank, c in enumerate(selected, 1):
        parent_id = c.get("parent_id", "")

        # Get parent text
        parent_text = _load_parent_text(session_id, parent_id, all_chunks)
        if not parent_text:
            parent_text = c.get("text", "")

        # Split into sentences
        sents = split_sentences(parent_text)
        if not sents:
            sents = [parent_text[:WINDOW_MAX_CHARS]]

        # Build window with position-based scoring
        window_entries = _position_scored_window(sents, WINDOW_MAX_CHARS)

        # Re-assign sentence IDs with passage rank
        for i, entry in enumerate(window_entries):
            entry["sid"] = f"{rank}.{i+1}"

        if not window_entries:
            # Fallback: first few sentences
            window_entries = []
            for i, s in enumerate(sents[:3]):
                window_entries.append({
                    "sid": f"{rank}.{i+1}",
                    "text": s,
                    "score": 0.0,
                })

        passage = {
            "pid": rank,
            "chunk_id": c.get("chunk_id", ""),
            "parent_id": parent_id,
            "source_file": c.get("source_file", ""),
            "source_type": c.get("source_type", ""),
            "page_or_slide": c.get("page_or_slide", 0),
            "heading": c.get("heading", ""),
            "rerank": round(c.get("retrieval_score", 0), 4),
            "rrf": round(c.get("retrieval_score", 0), 6),
            "window": window_entries,
        }
        passages.append(passage)

    result = {"status": "ok", "passages": passages}
    out_path = qa_paths.reranked_path(session_id, run_id)
    if not qa_paths.session_exists(session_id):
        raise FileNotFoundError(f"Session {session_id} no longer exists")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
