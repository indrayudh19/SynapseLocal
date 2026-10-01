"""
backend/session_manager.py
Phase 2: Session Isolation & State Management

Every chat session gets a unique UUID and fully isolated local storage:
  - FAISS index files + metadata
  - Graph objects
  - Uploaded documents

Deletion completely purges the session directory.
"""
import os
import uuid
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions")
SESSIONS_INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions_index.json")


def _load_sessions_index() -> Dict:
    """Load the global sessions index from disk."""
    if os.path.exists(SESSIONS_INDEX):
        with open(SESSIONS_INDEX, "r") as f:
            return json.load(f)
    return {}


def _save_sessions_index(index: Dict):
    """Persist the global sessions index."""
    os.makedirs(os.path.dirname(SESSIONS_INDEX), exist_ok=True)
    with open(SESSIONS_INDEX, "w") as f:
        json.dump(index, f, indent=2, default=str)


def create_session(name: Optional[str] = None) -> str:
    """
    Create a new isolated session with a unique UUID.
    Returns the session_id.
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(DATA_ROOT, session_id)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, "uploads"), exist_ok=True)
    
    index = _load_sessions_index()
    index[session_id] = {
        "name": name or f"Session {session_id[:6]}",
        "created_at": datetime.now().isoformat(),
        "document_count": 0,
        "chunk_count": 0,
        "is_processed": False
    }
    _save_sessions_index(index)
    
    return session_id


def delete_session(session_id: str):
    """Recursively delete all data for a session."""
    session_dir = os.path.join(DATA_ROOT, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    
    index = _load_sessions_index()
    index.pop(session_id, None)
    _save_sessions_index(index)


def list_sessions() -> Dict:
    """List all sessions with their metadata."""
    return _load_sessions_index()


def update_session_meta(session_id: str, **kwargs):
    """Update metadata fields for a session."""
    index = _load_sessions_index()
    if session_id in index:
        index[session_id].update(kwargs)
        _save_sessions_index(index)


def get_session_dir(session_id: str) -> str:
    """Get the absolute path to a session's data directory."""
    return os.path.join(DATA_ROOT, session_id)


def save_uploaded_files(session_id: str, files: list) -> List[str]:
    """
    Save uploaded Streamlit UploadedFile objects to the session directory.
    Returns list of saved filenames.
    """
    upload_dir = os.path.join(DATA_ROOT, session_id, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    saved = []
    for f in files:
        path = os.path.join(upload_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved.append(f.name)
    
    return saved
