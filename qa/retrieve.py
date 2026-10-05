"""
qa/retrieve.py
Q2: Raw Retrieve — keyword/phrase search + BM25 scoring across chunks.jsonl.
No embedding models, no FAISS. Pure Python text matching + lightweight BM25.
"""
import json
import os
import re

from qa.config import (
    RAW_RETRIEVE_TOP_K, BM25_TOPK,
    DOC_HINT_BOOST, PHRASE_MATCH_BONUS, HEADING_MATCH_BONUS,
)
from qa import paths as qa_paths
from qa.text_utils import content_tokens
from qa.bm25 import get_or_build as get_or_build_bm25


def _load_chunks(session_id: str) -> list[dict]:
    """Load chunks.jsonl into a list of dicts."""
    chunks = []
    with open(qa_paths.chunks_jsonl_path(session_id), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _phrase_match_score(text_lower: str, phrases: list[str]) -> float:
    """
    Count how many key-term phrases appear as substrings in the text.
    Returns PHRASE_MATCH_BONUS per matched phrase.
    """
    score = 0.0
    for phrase in phrases:
        if phrase in text_lower:
            score += PHRASE_MATCH_BONUS
    return score


def _heading_match_score(heading_lower: str, phrases: list[str]) -> float:
    """
    Bonus for headings that contain one or more key terms.
    """
    score = 0.0
    for phrase in phrases:
        if phrase in heading_lower:
            score += HEADING_MATCH_BONUS
    return score


def _token_overlap_score(chunk_tokens: set[str], query_tokens: set[str]) -> float:
    """
    Jaccard-like overlap between chunk content tokens and query tokens.
    Returns a 0-1 score scaled to a useful range.
    """
    if not chunk_tokens or not query_tokens:
        return 0.0
    intersection = len(chunk_tokens & query_tokens)
    if intersection == 0:
        return 0.0
    # Weighted by how many query terms were covered
    return intersection / len(query_tokens)


def run(session_id: str, run_id: str) -> dict:
    """
    Q2: Raw Retrieve — search chunks using keyword matching + BM25.
    Reads understanding.json, writes candidates.json.
    No embedding models or FAISS needed.
    """
    # Load understanding
    u_path = qa_paths.understanding_path(session_id, run_id)
    with open(u_path, "r", encoding="utf-8") as f:
        understanding = json.load(f)

    # Load chunks
    chunks = _load_chunks(session_id)
    if not chunks:
        result = {"status": "error", "message": "No chunks found in session."}
        out_path = qa_paths.candidates_path(session_id, run_id)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f)
        return result

    # ── Build search material from understanding ────────────
    key_terms = understanding.get("key_terms", [])
    queries = understanding.get("queries", [])
    hyp = understanding.get("hypothetical_answer", "")
    focus = understanding.get("focus", "")
    sub_questions = understanding.get("sub_questions", [])

    # Lowercase key-term phrases for substring matching
    key_phrases_lower = [t.lower().strip() for t in key_terms if t.strip()]

    # Build a broad set of query tokens from all text sources
    all_query_text = " ".join(queries + key_terms + sub_questions)
    if hyp:
        all_query_text += " " + hyp
    if focus:
        all_query_text += " " + focus
    query_tokens = set(content_tokens(all_query_text))

    # ── BM25 search ─────────────────────────────────────────
    chunk_texts = [c.get("text", "") for c in chunks]
    chunk_headings = [c.get("heading", "") for c in chunks]
    bm25_index = get_or_build_bm25(session_id, chunk_texts, chunk_headings)

    # Run BM25 with multiple queries, collect per-chunk best score
    bm25_scores: dict[int, float] = {}

    question = queries[0] if queries else ""
    bm25_terms_query = " ".join(key_terms) if key_terms else question

    for bm25_query in [question, bm25_terms_query]:
        if not bm25_query:
            continue
        hits = bm25_index.query(bm25_query, BM25_TOPK)
        for idx, score in hits:
            if idx not in bm25_scores or score > bm25_scores[idx]:
                bm25_scores[idx] = score

    # Sub-question BM25
    for sq in sub_questions:
        if sq:
            hits = bm25_index.query(sq, BM25_TOPK)
            for idx, score in hits:
                if idx not in bm25_scores or score > bm25_scores[idx]:
                    bm25_scores[idx] = score

    # ── Composite scoring for every chunk ───────────────────
    # Resolve hinted file indices
    hinted_files = set(understanding.get("_hinted_files", []))

    scored: list[tuple[int, float]] = []
    for i, c in enumerate(chunks):
        text_lower = c.get("text", "").lower()
        heading_lower = c.get("heading", "").lower()
        chunk_toks = set(content_tokens(c.get("text", "")))

        # 1. BM25 component (normalized to ~0-1 range)
        bm25_sc = bm25_scores.get(i, 0.0)

        # 2. Exact phrase match bonus
        phrase_sc = _phrase_match_score(text_lower, key_phrases_lower)

        # 3. Heading match bonus
        heading_sc = _heading_match_score(heading_lower, key_phrases_lower)

        # 4. Token overlap with query
        overlap_sc = _token_overlap_score(chunk_toks, query_tokens)

        # Composite score
        total = bm25_sc + phrase_sc + heading_sc + overlap_sc

        # 5. Doc hint boost
        if hinted_files and c.get("source_file") in hinted_files:
            total *= DOC_HINT_BOOST

        if total > 0:
            scored.append((i, total))

    # Sort by score descending, take top K
    scored.sort(key=lambda x: -x[1])
    scored = scored[:RAW_RETRIEVE_TOP_K]

    if not scored:
        # Nothing matched — still pass empty candidates forward
        result = {"status": "ok", "candidates": []}
        out_path = qa_paths.candidates_path(session_id, run_id)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # ── Build output candidates ─────────────────────────────
    candidates = []
    for chunk_idx, score in scored:
        c = chunks[chunk_idx]
        candidates.append({
            "chunk_id": c.get("id", ""),
            "parent_id": c.get("parent_id", ""),
            "source_file": c.get("source_file", ""),
            "source_type": c.get("source_type", ""),
            "page_or_slide": c.get("page_or_slide", 0),
            "heading": c.get("heading", ""),
            "text": c.get("text", ""),
            "retrieval_score": round(score, 6),
        })

    result = {"status": "ok", "candidates": candidates}
    out_path = qa_paths.candidates_path(session_id, run_id)
    if not qa_paths.session_exists(session_id):
        raise FileNotFoundError(f"Session {session_id} no longer exists")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
