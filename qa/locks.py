"""
qa/locks.py
Session lock + machine lock for the QA pipeline.
Atomic create (os.open with O_EXCL), pid+timestamp, stale detection, always released in finally.
"""
import os
import time
import json

from qa.config import STALE_LOCK_S
from qa import paths as qa_paths


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given pid is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _is_stale(lock_file: str) -> bool:
    """Return True if the lock is stale (pid dead or too old)."""
    try:
        with open(lock_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        pid = data.get("pid", 0)
        ts = data.get("ts", 0)
        if not _is_pid_alive(pid):
            return True
        if time.time() - ts > STALE_LOCK_S:
            return True
    except Exception:
        return True
    return False


def _acquire(lock_file: str) -> bool:
    """
    Try to atomically create the lock file.
    Returns True on success, False if the lock is held (and not stale).
    Removes stale locks before retrying once.
    """
    os.makedirs(os.path.dirname(lock_file), exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                content = json.dumps({"pid": os.getpid(), "ts": time.time()})
                os.write(fd, content.encode("utf-8"))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            if attempt == 0 and _is_stale(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
                continue
            return False
    return False


def _release(lock_file: str):
    """Remove the lock file if it exists."""
    try:
        os.remove(lock_file)
    except OSError:
        pass


# ── Public API ──────────────────────────────────────────────

def acquire_session_lock(session_id: str) -> bool:
    """Acquire the per-session QA lock. Returns True on success."""
    return _acquire(qa_paths.session_lock_path(session_id))


def release_session_lock(session_id: str):
    _release(qa_paths.session_lock_path(session_id))


def acquire_machine_lock() -> bool:
    """Acquire the global QA machine lock. Returns True on success."""
    return _acquire(qa_paths.machine_lock_path())


def release_machine_lock():
    _release(qa_paths.machine_lock_path())


def is_representation_running(session_id: str) -> bool:
    """
    Read-only check: is a representation stage currently running for this session?
    Checks the .running lock file. Returns False if stale.
    """
    lock = qa_paths.representation_lock_path(session_id)
    if not os.path.exists(lock):
        return False
    return not _is_stale(lock)
