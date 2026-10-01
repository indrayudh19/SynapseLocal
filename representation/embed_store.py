"""
representation/embed_store.py
Stage 1 Service: Ingests documents, performs semantic chunking, computes
L2-normalized embeddings in batches, and persists artifacts to disk.

Strict service contract:
- Single entrypoint: run(session_id)
- Heavy libraries imported ONLY inside run()
- Embedding model loaded only inside run() and released with del and gc.collect()
- Enforces .running lock file (removed in finally)
- Outputs: embeddings.npy, faiss.index, chunks.jsonl, embed.hash
- Never imports or calls other services
"""
import os
from representation.paths import (
    session_dir,
    uploads_dir,
    embeddings_path,
    faiss_index_path,
    chunks_jsonl_path,
    embed_hash_path,
    lock_path,
)


def _compute_files_hash(u_dir: str) -> str:
    """Compute sha256 hash of all files in uploads directory."""
    import hashlib
    hasher = hashlib.sha256()
    file_names = sorted([f for f in os.listdir(u_dir) if os.path.isfile(os.path.join(u_dir, f))])
    if not file_names:
        return ""
    for fname in file_names:
        fpath = os.path.join(u_dir, fname)
        hasher.update(fname.encode("utf-8"))
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()


def run(session_id: str) -> dict:
    """
    Run Stage 1: Document Ingest -> Chunking -> Embedding -> Vector Store.
    
    Args:
        session_id: Target session identifier
        
    Returns:
        Small stats dict
    """
    # 1. Enforce lock
    s_dir = session_dir(session_id)
    l_path = lock_path(session_id)
    if os.path.exists(l_path):
        return {
            "status": "error",
            "message": f"Another service is currently running for session {session_id}."
        }
    
    # Create lock file
    with open(l_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    
    try:
        # Lazy imports of heavy libraries
        import gc
        import json
        import pickle
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        from backend.parsers import parse_document
        from backend.chunker import semantic_chunk
        from backend.session_manager import update_session_meta
        from backend.vector_store import ChunkMeta

        u_dir = uploads_dir(session_id)
        file_names = sorted([f for f in os.listdir(u_dir) if os.path.isfile(os.path.join(u_dir, f))])
        if not file_names:
            return {
                "status": "error",
                "message": "No uploaded documents found. Please upload documents first."
            }

        # Cache check by content hash
        current_hash = _compute_files_hash(u_dir)
        h_path = embed_hash_path(session_id)
        e_path = embeddings_path(session_id)
        f_path = faiss_index_path(session_id)
        c_path = chunks_jsonl_path(session_id)

        if (
            os.path.exists(h_path)
            and os.path.exists(e_path)
            and os.path.exists(f_path)
            and os.path.exists(c_path)
        ):
            with open(h_path, "r", encoding="utf-8") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                # Count chunks from existing file
                with open(c_path, "r", encoding="utf-8") as f:
                    chunk_count = sum(1 for _ in f)
                embs = np.load(e_path, mmap_mode="r")
                dim = embs.shape[1]
                del embs
                gc.collect()
                return {
                    "status": "ok",
                    "chunks_count": chunk_count,
                    "embedding_dim": dim,
                    "files_processed": len(file_names),
                    "cached": True,
                }

        # Step A: Parse documents
        all_sections = []
        for fname in file_names:
            fpath = os.path.join(u_dir, fname)
            with open(fpath, "rb") as f:
                content = f.read()
            sections = parse_document(content, fname)
            all_sections.extend(sections)

        if not all_sections:
            return {
                "status": "error",
                "message": "No text could be extracted from the uploaded files."
            }

        # Step B: Load embedding model (single load inside run)
        model = SentenceTransformer("all-MiniLM-L6-v2")

        def _embed_fn(texts):
            return model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # Step C: Semantic chunking
        chunks = semantic_chunk(
            sections=all_sections,
            embed_fn=_embed_fn,
            k_threshold=1.0,
            session_id=session_id,
        )

        if not chunks:
            del model
            gc.collect()
            return {
                "status": "error",
                "message": "Chunking produced no output from the parsed text."
            }

        # Step D: Embed all child chunks in batches
        chunk_texts = [c.text for c in chunks]
        embeddings = model.encode(
            chunk_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        # CRITICAL: Free embedding model immediately
        del model
        gc.collect()

        # Step E: Exact inner-product FAISS index (IndexFlatIP)
        n_chunks, d = embeddings.shape
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        # Step F: Persist artifacts to disk
        np.save(e_path, embeddings)
        faiss.write_index(index, f_path)

        # Also write vector_store.index and chunk_metadata.pkl for backward compatibility with chat tab
        compat_index_path = os.path.join(s_dir, "vector_store.index")
        faiss.write_index(index, compat_index_path)

        chunk_meta_list = []
        parent_map = {}
        with open(c_path, "w", encoding="utf-8") as f:
            for c in chunks:
                # Determine source type (PDF, PPTX, TXT, MD)
                ext = os.path.splitext(c.source_file)[1].lstrip(".").upper() or "DOC"
                parent_id = f"{c.source_file}::{c.heading}::{c.page_or_slide}"
                row = {
                    "id": c.chunk_id,
                    "parent_id": parent_id,
                    "source_file": c.source_file,
                    "source_type": ext,
                    "page_or_slide": c.page_or_slide,
                    "heading": c.heading,
                    "text": c.text,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

                # Compat meta
                chunk_meta_list.append(ChunkMeta(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source_file=c.source_file,
                    heading=c.heading,
                    page_or_slide=c.page_or_slide,
                    parent_text=c.parent_text,
                ))
                parent_map.setdefault(parent_id, []).append(c.chunk_id)

        # Save compat metadata
        with open(os.path.join(s_dir, "chunk_metadata.pkl"), "wb") as f:
            pickle.dump(chunk_meta_list, f)
        with open(os.path.join(s_dir, "parent_map.pkl"), "wb") as f:
            pickle.dump(parent_map, f)

        # Write hash file
        with open(h_path, "w", encoding="utf-8") as f:
            f.write(current_hash)

        # Update session metadata
        update_session_meta(
            session_id,
            document_count=len(file_names),
            chunk_count=n_chunks,
            is_processed=True,
        )

        stats = {
            "status": "ok",
            "chunks_count": n_chunks,
            "embedding_dim": d,
            "files_processed": len(file_names),
            "cached": False,
        }

        # Free in-memory objects
        del index
        del embeddings
        del chunks
        del all_sections
        del chunk_meta_list
        del parent_map
        gc.collect()

        return stats

    finally:
        # Guarantee lock removal
        if os.path.exists(l_path):
            try:
                os.remove(l_path)
            except OSError:
                pass
