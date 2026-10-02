"""
representation/paths.py
Session-scoped path helpers and lock manager for representation services.
Strict isolation: every path helper requires session_id, no global paths.
"""
import os
from contextlib import contextmanager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def session_dir(session_id: str) -> str:
    """Return the absolute path to the session directory."""
    path = os.path.join(PROJECT_ROOT, "data", "sessions", session_id)
    os.makedirs(path, exist_ok=True)
    return path


def uploads_dir(session_id: str) -> str:
    """Return the absolute path to the session's uploads directory."""
    path = os.path.join(session_dir(session_id), "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def representation_dir(session_id: str) -> str:
    """Return the absolute path to data/sessions/<session_id>/representation/."""
    path = os.path.join(session_dir(session_id), "representation")
    os.makedirs(path, exist_ok=True)
    return path


def lock_path(session_id: str) -> str:
    """Path to the session .running lock file."""
    return os.path.join(session_dir(session_id), ".running")


def embeddings_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/embeddings.npy"""
    return os.path.join(session_dir(session_id), "embeddings.npy")


def faiss_index_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/faiss.index"""
    return os.path.join(session_dir(session_id), "faiss.index")


def chunks_jsonl_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/chunks.jsonl"""
    return os.path.join(session_dir(session_id), "chunks.jsonl")


def embed_hash_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/embed.hash"""
    return os.path.join(session_dir(session_id), "embed.hash")





def clusters_json_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/clusters.json"""
    return os.path.join(representation_dir(session_id), "clusters.json")


def cluster_map_png_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/cluster_map.png"""
    return os.path.join(representation_dir(session_id), "cluster_map.png")


def cluster_hash_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/cluster.hash"""
    return os.path.join(representation_dir(session_id), "cluster.hash")


@contextmanager
def session_lock(session_id: str):
    """
    Context manager to enforce single-service execution per session.
    Creates a .running file upon entry and guarantees removal in finally.
    Raises RuntimeError if a service is already running.
    """
    l_path = lock_path(session_id)
    if os.path.exists(l_path):
        raise RuntimeError("Another service is currently running for this session. Please wait until it finishes.")
    
    with open(l_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    
    try:
        yield
    finally:
        if os.path.exists(l_path):
            try:
                os.remove(l_path)
            except OSError:
                pass


def concepts_raw_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/concepts_raw.jsonl"""
    return os.path.join(representation_dir(session_id), "concepts_raw.jsonl")


def extract_log_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/extract_log.json"""
    return os.path.join(representation_dir(session_id), "extract_log.json")


def extract_hash_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/extract.hash"""
    return os.path.join(representation_dir(session_id), "extract.hash")


def concept_graph_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/concept_graph.json"""
    return os.path.join(representation_dir(session_id), "concept_graph.json")


def concept_graph_hash_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/concept_graph.hash"""
    return os.path.join(representation_dir(session_id), "concept_graph.hash")


def concept_clusters_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/concept_clusters.json"""
    return os.path.join(representation_dir(session_id), "concept_clusters.json")


def concept_clusters_hash_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/concept_clusters.hash"""
    return os.path.join(representation_dir(session_id), "concept_clusters.hash")


def concept_map_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/concept_map.png"""
    return os.path.join(representation_dir(session_id), "concept_map.png")


def concept_vectors_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/concept_vectors.npy"""
    return os.path.join(representation_dir(session_id), "concept_vectors.npy")


def concept_map_html_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/concept_map.html"""
    return os.path.join(representation_dir(session_id), "concept_map.html")


def concept_map_hash_path(session_id: str) -> str:
    """Path to data/sessions/<session_id>/representation/concept_map.hash"""
    return os.path.join(representation_dir(session_id), "concept_map.hash")

