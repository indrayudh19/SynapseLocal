"""
qa/rerank.py
Q3: Rerank and evidence windows.
Runs inside the worker subprocess (torch / cross-encoder imports are local).
"""
import json
import os

from qa.config import (
    RERANKER, RERANK_MAXLEN, RERANK_BATCH,
    K_PASSAGES, K_PASSAGES_COMPARE,
    PARENT_MAX_CHARS, WINDOW_MAX_CHARS,
    NO_ANSWER_SCORE, COVERAGE_MARGIN,
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


def _best_window(sentences: list[dict], max_chars: int) -> list[dict]:
    """
    Choose the contiguous window of sentences (at most max_chars total)
    that maximizes the sum of positive sentence scores.
    """
    if not sentences:
        return []

    n = len(sentences)
    best_score = -float("inf")
    best_start, best_end = 0, 0

    for start in range(n):
        total_chars = 0
        total_score = 0.0
        for end in range(start, n):
            total_chars += len(sentences[end]["text"])
            sc = sentences[end].get("score", 0.0)
            if sc > 0:
                total_score += sc
            if total_chars > max_chars:
                break
            if total_score > best_score:
                best_score = total_score
                best_start = start
                best_end = end

    return sentences[best_start:best_end + 1]


def run(session_id: str, run_id: str) -> dict:
    """
    Q3: Rerank candidates and build evidence windows.
    Reads candidates.json and understanding.json, writes reranked.json.
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
    sub_questions = understanding.get("sub_questions", [])

    # Try loading cross-encoder
    cross_encoder = None
    rerank_skipped = False
    try:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(RERANKER, max_length=RERANK_MAXLEN)
    except Exception:
        rerank_skipped = True

    if cross_encoder is not None:
        # Score each candidate: (question, heading + " " + text)
        pairs = []
        for c in candidates:
            heading = c.get("heading", "")
            text = c.get("text", "")
            doc_text = (heading + " " + text).strip() if heading else text
            pairs.append((question, doc_text))

        scores = cross_encoder.predict(
            pairs, batch_size=RERANK_BATCH, show_progress_bar=False
        )

        # Sub-question scoring: take max
        if sub_questions:
            for sq in sub_questions:
                sq_pairs = [(sq, p[1]) for p in pairs]
                sq_scores = cross_encoder.predict(
                    sq_pairs, batch_size=RERANK_BATCH, show_progress_bar=False
                )
                scores = [max(s, sq_s) for s, sq_s in zip(scores, sq_scores)]

        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        # Sort by rerank score descending
        candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

        # Answerability gate
        best_score = candidates[0].get("rerank_score", 0) if candidates else 0
        if best_score < NO_ANSWER_SCORE:
            result = {
                "status": "not_found",
                "closest": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "source_file": c.get("source_file"),
                        "heading": c.get("heading"),
                        "text": c.get("text", "")[:300],
                        "rerank": c.get("rerank_score", 0),
                    }
                    for c in candidates[:3]
                ],
            }
            out_path = qa_paths.reranked_path(session_id, run_id)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return result
    else:
        # No cross-encoder: keep fused order, no gate
        for c in candidates:
            c["rerank_score"] = c.get("rrf", 0)

    # Select passages
    k_pass = K_PASSAGES_COMPARE if intent == "comparison" else K_PASSAGES
    seen_parents = set()
    selected = []
    for c in candidates:
        pid = c.get("parent_id", c.get("chunk_id"))
        if pid in seen_parents:
            continue
        seen_parents.add(pid)
        selected.append(c)
        if len(selected) >= k_pass:
            break

    # Per-document coverage: if candidates from 2+ source files exist,
    # include the best candidate of each file within COVERAGE_MARGIN of best
    if len(selected) > 0:
        best_rerank = selected[0].get("rerank_score", 0)
        source_files_in_selected = {c.get("source_file") for c in selected}
        all_source_files = {c.get("source_file") for c in candidates}

        if len(all_source_files) >= 2:
            for c in candidates:
                sf = c.get("source_file")
                pid = c.get("parent_id", c.get("chunk_id"))
                if sf not in source_files_in_selected and pid not in seen_parents:
                    if c.get("rerank_score", 0) >= best_rerank - COVERAGE_MARGIN:
                        selected.append(c)
                        seen_parents.add(pid)
                        source_files_in_selected.add(sf)

    # Load all chunks for parent text reconstruction
    all_chunks = []
    chunks_path = qa_paths.chunks_jsonl_path(session_id)
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))

    # Build passage windows
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

        # Score sentences with cross-encoder if available
        sent_entries = []
        if cross_encoder is not None and not rerank_skipped:
            sent_pairs = [(question, s) for s in sents]
            sent_scores = cross_encoder.predict(
                sent_pairs, batch_size=RERANK_BATCH, show_progress_bar=False
            )
            for i, (s, sc) in enumerate(zip(sents, sent_scores)):
                sent_entries.append({
                    "sid": f"{rank}.{i+1}",
                    "text": s,
                    "score": round(float(sc), 4),
                })
        else:
            # No cross-encoder: use first sentences as window
            for i, s in enumerate(sents):
                sent_entries.append({
                    "sid": f"{rank}.{i+1}",
                    "text": s,
                    "score": 0.0,
                })

        # Select best window
        window = _best_window(sent_entries, WINDOW_MAX_CHARS)
        if not window:
            window = sent_entries[:3]  # fallback

        passage = {
            "pid": rank,
            "chunk_id": c.get("chunk_id", ""),
            "parent_id": parent_id,
            "source_file": c.get("source_file", ""),
            "source_type": c.get("source_type", ""),
            "page_or_slide": c.get("page_or_slide", 0),
            "heading": c.get("heading", ""),
            "rerank": round(c.get("rerank_score", 0), 4) if not rerank_skipped else "skipped",
            "rrf": round(c.get("rrf", 0), 6),
            "window": window,
        }
        passages.append(passage)

    # Free cross-encoder
    if cross_encoder is not None:
        del cross_encoder
        import gc
        gc.collect()

    result = {"status": "ok", "passages": passages}
    out_path = qa_paths.reranked_path(session_id, run_id)
    if not qa_paths.session_exists(session_id):
        raise FileNotFoundError(f"Session {session_id} no longer exists")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
