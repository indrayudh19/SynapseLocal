"""
qa/paths.py
Session-scoped path helpers for the QA pipeline.
All paths resolve under data/sessions/<session_id>/qa/.
Every writer must check that the session directory exists before writing.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "sessions")


def session_dir(session_id: str) -> str:
    """Absolute path to data/sessions/<session_id>/."""
    return os.path.join(DATA_ROOT, session_id)


def qa_dir(session_id: str) -> str:
    """Absolute path to data/sessions/<session_id>/qa/."""
    return os.path.join(DATA_ROOT, session_id, "qa")


def run_dir(session_id: str, run_id: str) -> str:
    """Absolute path to data/sessions/<session_id>/qa/runs/<run_id>/."""
    return os.path.join(qa_dir(session_id), "runs", run_id)


def ensure_run_dir(session_id: str, run_id: str) -> str:
    """Create and return the run directory. Aborts if the session dir is gone."""
    s_dir = session_dir(session_id)
    if not os.path.isdir(s_dir):
        raise FileNotFoundError(f"Session directory does not exist: {s_dir}")
    r_dir = run_dir(session_id, run_id)
    os.makedirs(r_dir, exist_ok=True)
    return r_dir


def session_exists(session_id: str) -> bool:
    """Check whether the session directory exists on disk."""
    return os.path.isdir(session_dir(session_id))


# ── Run-scoped stage files ──────────────────────────────────

def understanding_path(session_id: str, run_id: str) -> str:
    return os.path.join(run_dir(session_id, run_id), "understanding.json")

def candidates_path(session_id: str, run_id: str) -> str:
    return os.path.join(run_dir(session_id, run_id), "candidates.json")

def reranked_path(session_id: str, run_id: str) -> str:
    return os.path.join(run_dir(session_id, run_id), "reranked.json")

def answer_path(session_id: str, run_id: str) -> str:
    return os.path.join(run_dir(session_id, run_id), "answer.json")

def verified_path(session_id: str, run_id: str) -> str:
    return os.path.join(run_dir(session_id, run_id), "verified.json")

def consolidated_path(session_id: str, run_id: str) -> str:
    return os.path.join(run_dir(session_id, run_id), "consolidated.json")

def error_path(session_id: str, run_id: str) -> str:
    return os.path.join(run_dir(session_id, run_id), "error.json")


# ── Session-scoped QA files ────────────────────────────────

def answers_jsonl_path(session_id: str) -> str:
    return os.path.join(qa_dir(session_id), "answers.jsonl")


# ── BM25 cache ──────────────────────────────────────────────

def bm25_index_path(session_id: str) -> str:
    return os.path.join(qa_dir(session_id), "bm25_index.npz")

def bm25_vocab_path(session_id: str) -> str:
    return os.path.join(qa_dir(session_id), "bm25_vocab.json")

def bm25_hash_path(session_id: str) -> str:
    return os.path.join(qa_dir(session_id), "bm25.hash")


# ── Locks ────────────────────────────────────────────────────

def session_lock_path(session_id: str) -> str:
    """Per-session QA lock: qa/.lock"""
    return os.path.join(qa_dir(session_id), ".lock")

def machine_lock_path() -> str:
    """Global QA lock: data/.qa_global.lock"""
    return os.path.join(PROJECT_ROOT, "data", ".qa_global.lock")


# ── Representation lock (read-only check) ────────────────────

def representation_lock_path(session_id: str) -> str:
    """Path to the per-session .running lock used by representation stages."""
    return os.path.join(session_dir(session_id), ".running")


# ── Existing index / chunk store paths ───────────────────────

def faiss_index_path(session_id: str) -> str:
    """Path to the session's FAISS index built by Embed & Store."""
    return os.path.join(session_dir(session_id), "faiss.index")

def chunk_metadata_path(session_id: str) -> str:
    """Path to chunk_metadata.pkl (list of ChunkMeta dataclass instances)."""
    return os.path.join(session_dir(session_id), "chunk_metadata.pkl")

def parent_map_path(session_id: str) -> str:
    """Path to parent_map.pkl (parent_key -> [child_chunk_ids])."""
    return os.path.join(session_dir(session_id), "parent_map.pkl")

def chunks_jsonl_path(session_id: str) -> str:
    """Path to chunks.jsonl (one JSON line per chunk with id, parent_id, text, etc.)."""
    return os.path.join(session_dir(session_id), "chunks.jsonl")

def embeddings_npy_path(session_id: str) -> str:
    """Path to embeddings.npy (shape: n_chunks × dim)."""
    return os.path.join(session_dir(session_id), "embeddings.npy")
