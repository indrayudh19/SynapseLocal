"""
qa/retrieve.py
Q2: Retrieve — dense + BM25 + RRF fusion.
Runs inside the worker subprocess (torch imports are local to functions).
"""
import gc
import json
import os
import pickle

import numpy as np

from qa.config import (
    DENSE_TOPK, BM25_TOPK, FUSED_CAND, SUBQ_TOPK,
    RRF_K, LIST_WEIGHTS, DOC_HINT_BOOST,
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


def _dense_search(query_texts: list[str], session_id: str,
                  top_k: int) -> dict[str, list[tuple[int, float]]]:
    """
    Embed queries and search the FAISS index.
    Returns {list_name: [(chunk_index, score), ...]}.
    The embedding model is loaded, used, and freed within this function.
    """
    import faiss

    # Load FAISS index
    index_path = qa_paths.faiss_index_path(session_id)
    index = faiss.read_index(index_path)

    # Determine embedding model — the session uses all-MiniLM-L6-v2 (sentence-transformers)
    # No prefix is needed for this model (neither document nor query prefix).
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        query_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Free model immediately
    del model
    gc.collect()

    faiss.normalize_L2(embeddings)

    results: dict[str, list[tuple[int, float]]] = {}
    k = min(top_k, index.ntotal)
    if k <= 0:
        return results

    for i, q_emb in enumerate(embeddings):
        q = q_emb.reshape(1, -1)
        scores, indices = index.search(q, k)
        hits = [(int(idx), float(sc)) for sc, idx in zip(scores[0], indices[0]) if idx >= 0]
        results[f"dense_{i}"] = hits

    del index
    gc.collect()
    return results


def _rrf_fuse(ranked_lists: dict[str, list[tuple[int, float]]],
              weights: dict[str, float],
              hinted_indices: set[int],
              cap: int) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion over weighted lists.
    score(d) = sum_w w / (RRF_K + rank).
    Multiply by DOC_HINT_BOOST for hinted documents.
    """
    scores: dict[int, float] = {}
    for list_name, hits in ranked_lists.items():
        w = weights.get(list_name, 1.0)
        for rank, (idx, _score) in enumerate(hits):
            scores[idx] = scores.get(idx, 0.0) + w / (RRF_K + rank + 1)

    # Apply doc hint boost
    if hinted_indices:
        for idx in hinted_indices:
            if idx in scores:
                scores[idx] *= DOC_HINT_BOOST

    # Sort and cap
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_items[:cap]


def run(session_id: str, run_id: str) -> dict:
    """
    Q2: Retrieve candidates using dense + BM25 + RRF.
    Reads understanding.json, writes candidates.json.
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

    # Build dense query texts
    queries = understanding.get("queries", [])
    hyp = understanding.get("hypothetical_answer", "")
    dense_queries = list(queries)  # already has original question prepended
    if hyp:
        dense_queries.append(hyp)

    # Build list name mapping for dense queries
    dense_list_names = {}
    for i, q in enumerate(dense_queries):
        if i == 0:
            dense_list_names[f"dense_{i}"] = "dense_original"
        elif i <= len(queries) - 1:
            dense_list_names[f"dense_{i}"] = "dense_rewrite"
        else:
            dense_list_names[f"dense_{i}"] = "dense_hyde"

    # Dense search
    dense_results_raw = _dense_search(dense_queries, session_id, DENSE_TOPK)

    # Remap to named lists
    dense_results = {}
    for raw_name, hits in dense_results_raw.items():
        mapped_name = dense_list_names.get(raw_name, raw_name)
        # If multiple lists map to the same name, merge them
        if mapped_name in dense_results:
            # Merge: keep best score per index
            existing = {idx: sc for idx, sc in dense_results[mapped_name]}
            for idx, sc in hits:
                if idx not in existing or sc > existing[idx]:
                    existing[idx] = sc
            dense_results[mapped_name] = sorted(existing.items(), key=lambda x: -x[1])
        else:
            dense_results[mapped_name] = hits

    # BM25 search
    chunk_texts = [c.get("text", "") for c in chunks]
    chunk_headings = [c.get("heading", "") for c in chunks]
    bm25_index = get_or_build_bm25(session_id, chunk_texts, chunk_headings)

    key_terms = understanding.get("key_terms", [])
    question = queries[0] if queries else ""

    bm25_terms_query = " ".join(key_terms) if key_terms else question
    bm25_results = {
        "bm25_terms": bm25_index.query(bm25_terms_query, BM25_TOPK),
        "bm25_question": bm25_index.query(question, BM25_TOPK),
    }

    # Resolve hinted file indices
    hinted_files = set(understanding.get("_hinted_files", []))
    hinted_indices = set()
    if hinted_files:
        for i, c in enumerate(chunks):
            if c.get("source_file") in hinted_files:
                hinted_indices.add(i)

    # Merge all lists
    all_lists = {**dense_results, **bm25_results}

    # Main fusion
    fused = _rrf_fuse(all_lists, LIST_WEIGHTS, hinted_indices, FUSED_CAND)

    # Sub-questions (for comparison / multi-part)
    sub_questions = understanding.get("sub_questions", [])
    if sub_questions:
        for sq in sub_questions:
            sq_dense = _dense_search([sq], session_id, DENSE_TOPK)
            sq_bm25 = {"bm25_question": bm25_index.query(sq, BM25_TOPK)}
            sq_lists = {**{f"dense_original": sq_dense.get("dense_0", [])}, **sq_bm25}
            sq_fused = _rrf_fuse(sq_lists, LIST_WEIGHTS, hinted_indices, SUBQ_TOPK)
            # Union with main
            existing_indices = {idx for idx, _ in fused}
            for idx, score in sq_fused:
                if idx not in existing_indices:
                    fused.append((idx, score))
                    existing_indices.add(idx)

        # Cap total when sub-questions present
        fused = sorted(fused, key=lambda x: -x[1])[:30]

    # Build output candidates
    candidates = []
    for chunk_idx, rrf_score in fused:
        c = chunks[chunk_idx]
        # Build per-list rank info
        ranks = {}
        for list_name, hits in all_lists.items():
            for rank, (idx, _) in enumerate(hits):
                if idx == chunk_idx:
                    ranks[list_name] = rank + 1
                    break

        candidates.append({
            "chunk_id": c.get("id", ""),
            "parent_id": c.get("parent_id", ""),
            "source_file": c.get("source_file", ""),
            "source_type": c.get("source_type", ""),
            "page_or_slide": c.get("page_or_slide", 0),
            "heading": c.get("heading", ""),
            "text": c.get("text", ""),
            "rrf": round(rrf_score, 6),
            "ranks": ranks,
        })

    result = {"status": "ok", "candidates": candidates}
    out_path = qa_paths.candidates_path(session_id, run_id)
    if not qa_paths.session_exists(session_id):
        raise FileNotFoundError(f"Session {session_id} no longer exists")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
