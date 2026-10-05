"""
qa/orchestrator.py
Sequences the five QA stages and manages locks, run directories, and error handling.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import ollama

from qa.config import QA_MODEL, PIPELINE_VERSION, QA_DEBUG, WORKER_TIMEOUT_S, RERANKER
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


def _run_worker(stage: str, session_id: str, run_id: str) -> dict:
    """
    Run a QA worker subprocess for retrieve or rerank.
    Returns the parsed JSON status line from stdout.
    """
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    result = subprocess.run(
        [sys.executable, "-m", "qa.worker", stage, session_id, run_id],
        timeout=WORKER_TIMEOUT_S,
        capture_output=True,
        text=True,
        env=env,
        cwd=qa_paths.PROJECT_ROOT,
    )

    if result.returncode != 0:
        # Try reading error from stdout or error.json
        error_msg = f"Worker {stage} failed (exit {result.returncode})"
        if result.stdout.strip():
            try:
                status = json.loads(result.stdout.strip().split("\n")[-1])
                error_msg = status.get("message", error_msg)
            except Exception:
                pass
        err_file = qa_paths.error_path(session_id, run_id)
        if os.path.exists(err_file):
            try:
                with open(err_file, "r", encoding="utf-8") as f:
                    err_data = json.load(f)
                error_msg = err_data.get("error", error_msg)
            except Exception:
                pass
        raise RuntimeError(error_msg)

    # Parse status from stdout
    if result.stdout.strip():
        try:
            return json.loads(result.stdout.strip().split("\n")[-1])
        except Exception:
            pass
    return {"status": "ok"}


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

    # Check index exists
    if not os.path.exists(qa_paths.faiss_index_path(session_id)):
        return {"status": "error", "message": "No index found. Run Embed & Store first."}

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

            # Best-effort unload after Q1
            _unload_all_models()

            # ── Q2: Retrieve (worker subprocess) ─────────────
            _status("Searching the documents")
            t0 = time.time()
            _run_worker("retrieve", session_id, run_id)
            timings["retrieve_s"] = round(time.time() - t0, 1)

            # ── Q3: Rerank (worker subprocess) ───────────────
            _status("Ranking passages")
            t0 = time.time()
            _run_worker("rerank", session_id, run_id)
            timings["rerank_s"] = round(time.time() - t0, 1)

            # Check if Q3 wrote not_found
            reranked_file = qa_paths.reranked_path(session_id, run_id)
            with open(reranked_file, "r", encoding="utf-8") as f:
                reranked_data = json.load(f)

            if reranked_data.get("status") == "not_found":
                # Skip Q4, Q5, Q6
                record = _build_record(
                    run_id, question, understanding, reranked_data,
                    {"status": "not_found", "claims": [], "partial": False,
                     "closest": reranked_data.get("closest", []), "dropped_claims": []},
                    None, timings, chunk_files,
                )
                store.append_answer(session_id, record)
                _cleanup_run(session_id, run_id)
                return record

            # ── Q4: Answer (in process) ──────────────────────
            _status("Writing the answer")
            t0 = time.time()
            from qa.answer import run as answer_run
            answer_data = answer_run(session_id, run_id, question)
            timings["answer_s"] = round(time.time() - t0, 1)

            # ── Q5: Verify (in process) ──────────────────────
            _status("Checking the answer")
            t0 = time.time()
            from qa.verify import run as verify_run
            verified = verify_run(session_id, run_id)
            timings["verify_s"] = round(time.time() - t0, 1)

            # ── Q6: Polish (in process) ──────────────────────
            _status("Writing the final answer")
            t0 = time.time()
            from qa.polish import run as polish_run
            polished = polish_run(session_id, run_id)
            if polished and "polish_s" in polished:
                timings["polish_s"] = polished["polish_s"]
            else:
                timings["polish_s"] = round(time.time() - t0, 1)

            if polished is None:
                record = _build_record(
                    run_id, question, understanding, reranked_data,
                    {"status": "not_found", "claims": [], "partial": False,
                     "closest": reranked_data.get("closest", []), "dropped_claims": []},
                    None, timings, chunk_files,
                )
                store.append_answer(session_id, record)
                _cleanup_run(session_id, run_id)
                return record

            # ── Build final record ───────────────────────────
            record = _build_record(
                run_id, question, understanding, reranked_data,
                verified, polished, timings, chunk_files,
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
                  reranked: dict, verified: dict, polished: dict | None,
                  timings: dict, chunk_files: set[str]) -> dict:
    """Build the final answer record for answers.jsonl."""
    ts = datetime.now().isoformat(timespec="seconds")
    answer_id = "a_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine embed model from session (always all-MiniLM-L6-v2 in this codebase)
    embed_model = "all-MiniLM-L6-v2"

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

    rec = {
        "id": answer_id,
        "ts": ts,
        "question": question,
        "status": verified.get("status", "answered"),
        "partial": verified.get("partial", False),
        "understanding": {
            "intent": understanding.get("intent", "other"),
            "answer_type": understanding.get("answer_type", "paragraph"),
            "key_terms": understanding.get("key_terms", []),
            "queries": understanding.get("queries", []),
            "sub_questions": understanding.get("sub_questions", []),
        },
        "claims": verified.get("claims", []),
        "evidence": evidence,
        "dropped_claims": verified.get("dropped_claims", []),
        "timings": timings,
        "versions": {
            "qa_model": QA_MODEL,
            "embed_model": embed_model,
            "reranker": RERANKER,
            "pipeline": PIPELINE_VERSION,
        },
    }
    if polished is not None:
        rec["polished"] = polished
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
