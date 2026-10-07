"""
qa/orchestrator.py
Sequences the six QA stages and manages locks, run directories, and error handling.
Q2 (raw retrieve) and Q3 (LLM semantic filter) now run in-process — no subprocess workers.
"""
import json
import os
import shutil
import time
from datetime import datetime

import ollama

from qa.config import QA_MODEL, PIPELINE_VERSION, QA_DEBUG, WORKER_TIMEOUT_S
from qa import paths as qa_paths
from qa import locks
from qa import store


def _unload_all_models():
    """Best-effort: unload any Ollama models still resident."""
    try:
        running = ollama.ps()
        for m in running.get("models", []):
            try:
                ollama.generate(model=m["name"], prompt="", keep_alive=0)
            except Exception:
                pass
    except Exception:
        pass




def ask(session_id: str, question: str, on_status=None) -> dict:
    """
    Answer a question for the given session.

    Args:
        session_id: Target session
        question: User question (trimmed to 500 chars)
        on_status: Optional callback(label: str) for UI updates

    Returns:
        Final answer record dict, or error dict.
    """
    def _status(label: str):
        if on_status:
            on_status(label)

    # ── Preconditions ────────────────────────────────────────
    question = question.strip()[:500]
    if not question:
        return {"status": "error", "message": "Question is empty."}

    # Check chunks exist (no FAISS index needed for QA)
    if not os.path.exists(qa_paths.chunks_jsonl_path(session_id)):
        return {"status": "error", "message": "No chunks found. Run Embed & Store first."}

    # Check representation lock (read-only)
    if locks.is_representation_running(session_id):
        return {
            "status": "error",
            "message": "A concept map stage is running. Try again when it finishes."
        }

    # ── Acquire locks ────────────────────────────────────────
    if not locks.acquire_machine_lock():
        return {"status": "error", "message": "Another QA run is in progress. Please wait."}

    run_id = None
    try:
        if not locks.acquire_session_lock(session_id):
            return {"status": "error", "message": "Another question is being answered for this session."}

        try:
            # Create run directory
            run_id = datetime.now().strftime("r_%Y%m%d_%H%M%S")
            qa_paths.ensure_run_dir(session_id, run_id)

            timings: dict[str, float] = {}

            # Collect chunk files for doc_hint resolution
            chunk_files: set[str] = set()
            try:
                with open(qa_paths.chunks_jsonl_path(session_id), "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            c = json.loads(line)
                            sf = c.get("source_file", "")
                            if sf:
                                chunk_files.add(sf)
            except Exception:
                pass

            # ── Q1: Understand (in process) ──────────────────
            _status("Understanding the question")
            t0 = time.time()
            from qa.understand import run as understand_run
            understanding = understand_run(session_id, run_id, question, chunk_files)
            timings["understand_s"] = round(time.time() - t0, 1)

            # ── Q2: Raw Retrieve (in process) ────────────────
            _status("Searching the documents")
            t0 = time.time()
            from qa.retrieve import run as retrieve_run
            retrieve_run(session_id, run_id)
            timings["retrieve_s"] = round(time.time() - t0, 1)

            # ── Q3: LLM Semantic Filter (in process) ─────────
            _status("Filtering passages")
            t0 = time.time()
            from qa.rerank import run as rerank_run
            rerank_run(session_id, run_id)
            timings["rerank_s"] = round(time.time() - t0, 1)

            # Check if Q3 wrote not_found
            reranked_file = qa_paths.reranked_path(session_id, run_id)
            with open(reranked_file, "r", encoding="utf-8") as f:
                reranked_data = json.load(f)

            if reranked_data.get("status") == "not_found":
                record = _build_record(
                    run_id, question, understanding, reranked_data,
                    {"status": "not_found", "text": "The documents do not appear to contain an answer to this question."},
                    timings, chunk_files,
                )
                store.append_answer(session_id, record)
                _cleanup_run(session_id, run_id)
                return record

            # ── Q4: Answer (Pass 7: LLM Answer Structuring) ──
            _status("Writing the answer")
            t0 = time.time()
            from qa.answer import run as answer_run
            answer_data = answer_run(session_id, run_id, question)
            timings["answer_s"] = round(time.time() - t0, 1)

            # ── Build final record ───────────────────────────
            record = _build_record(
                run_id, question, understanding, reranked_data,
                answer_data, timings, chunk_files,
            )
            store.append_answer(session_id, record)
            _cleanup_run(session_id, run_id)
            return record

        except Exception as e:
            _cleanup_run(session_id, run_id)
            return {"status": "error", "message": f"QA failed: {str(e)}"}

        finally:
            try:
                ollama.generate(model=QA_MODEL, prompt="", keep_alive=0)
            except Exception:
                pass
            locks.release_session_lock(session_id)

    finally:
        locks.release_machine_lock()


def _build_record(run_id: str, question: str, understanding: dict,
                  reranked: dict, answer_data: dict,
                  timings: dict, chunk_files: set[str]) -> dict:
    """Build the final answer record for answers.jsonl."""
    ts = datetime.now().isoformat(timespec="seconds")
    answer_id = "a_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build evidence from passages
    evidence = []
    for p in reranked.get("passages", []):
        evidence.append({
            "pid": p.get("pid"),
            "chunk_id": p.get("chunk_id", ""),
            "source_file": p.get("source_file", ""),
            "page_or_slide": p.get("page_or_slide", 0),
            "heading": p.get("heading", ""),
            "window": p.get("window", []),
            "rerank": p.get("rerank", 0),
            "rrf": p.get("rrf", 0),
        })

    answer_text = answer_data.get("text", "")

    rec = {
        "id": answer_id,
        "ts": ts,
        "question": question,
        "status": answer_data.get("status", "answered"),
        "partial": answer_data.get("partial", False),
        "understanding": {
            "intent": understanding.get("intent", "other"),
            "answer_type": understanding.get("answer_type", "paragraph"),
            "key_terms": understanding.get("key_terms", []),
            "queries": understanding.get("queries", []),
            "sub_questions": understanding.get("sub_questions", []),
        },
        "answer": answer_text,
        "claims": answer_data.get("claims", []),
        "evidence": evidence,
        "dropped_claims": answer_data.get("dropped_claims", []),
        "timings": timings,
        "versions": {
            "qa_model": QA_MODEL,
            "retrieval": "raw_text+bm25",
            "filter": "qwen_llm_semantic",
            "pipeline": PIPELINE_VERSION,
        },
        "polished": {
            "text": answer_text,
            "fallback": False,
            "words": len(answer_text.split()),
            "answer_s": timings.get("answer_s", 0),
        },
    }
    return rec


def _cleanup_run(session_id: str, run_id: str):
    """Delete the run directory unless QA_DEBUG is True."""
    if QA_DEBUG:
        return
    if run_id is None:
        return
    r_dir = qa_paths.run_dir(session_id, run_id)
    if os.path.isdir(r_dir):
        try:
            shutil.rmtree(r_dir)
        except OSError:
            pass
    # Also clean up empty runs/ directory
    runs_parent = os.path.dirname(r_dir)
    if os.path.isdir(runs_parent) and not os.listdir(runs_parent):
        try:
            os.rmdir(runs_parent)
        except OSError:
            pass
