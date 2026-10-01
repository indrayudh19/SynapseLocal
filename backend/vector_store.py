"""
backend/vector_store.py
Module 1.3: Parent-Child Hierarchical Indexing & Local Vector Store (FAISS-CPU)

Architecture:
  - Child Chunks (Micro): Individual semantic chunks embedded into FAISS IndexFlatIP
    for high-speed inner product (cosine) similarity search.
  - Parent Chunks (Macro): Full section text blocks mapped to their child IDs
    via a local metadata dictionary (pickle-backed).

All data is scoped to a session directory: data/sessions/{session_id}/
"""
import os
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import faiss


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions")


@dataclass
class ChunkMeta:
    """Metadata for a stored chunk."""
    chunk_id: str
    text: str
    source_file: str
    heading: str
    page_or_slide: int
    parent_text: str


class LocalVectorStore:
    """
    FAISS-backed local vector store with parent-child hierarchical indexing.
    
    Uses IndexFlatIP (inner product = cosine similarity on L2-normalized vectors).
    All state is persisted to disk under data/sessions/{session_id}/.
    """

    def __init__(self, session_id: str, embedding_dim: int = 384):
        self.session_id = session_id
        self.embedding_dim = embedding_dim
        
        # Session-scoped directory
        self.session_dir = os.path.join(DATA_ROOT, session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # File paths
        self._index_path = os.path.join(self.session_dir, "vector_store.index")
        self._meta_path = os.path.join(self.session_dir, "chunk_metadata.pkl")
        self._parent_map_path = os.path.join(self.session_dir, "parent_map.pkl")
        
        # In-memory state
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_metadata: List[ChunkMeta] = []
        self.parent_map: Dict[str, List[str]] = {}  # parent_key -> [child_chunk_ids]
        
        # Load existing state or create fresh
        self._load_or_create()
    
    def _load_or_create(self):
        """Load persisted index and metadata from disk, or create new."""
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self.index = faiss.read_index(self._index_path)
            with open(self._meta_path, "rb") as f:
                self.chunk_metadata = pickle.load(f)
            if os.path.exists(self._parent_map_path):
                with open(self._parent_map_path, "rb") as f:
                    self.parent_map = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_metadata = []
            self.parent_map = {}
    
    def _save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.chunk_metadata, f)
        with open(self._parent_map_path, "wb") as f:
            pickle.dump(self.parent_map, f)
    
    def add_chunks(self, chunks: list, embeddings: np.ndarray):
        """
        Add semantic chunks with their embeddings to the store.
        
        Args:
            chunks: List of SemanticChunk objects from chunker.py
            embeddings: np.ndarray of shape (n_chunks, embedding_dim), L2-normalized
        """
        if len(chunks) == 0:
            return
        
        # Ensure float32 and L2-normalize for cosine similarity via inner product
        vectors = embeddings.astype(np.float32)
        faiss.normalize_L2(vectors)
        
        # Add to FAISS index
        self.index.add(vectors)
        
        # Store metadata
        for chunk in chunks:
            meta = ChunkMeta(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_file=chunk.source_file,
                heading=chunk.heading,
                page_or_slide=chunk.page_or_slide,
                parent_text=chunk.parent_text
            )
            self.chunk_metadata.append(meta)
            
            # Build parent -> child mapping
            parent_key = f"{chunk.source_file}::{chunk.heading}::{chunk.page_or_slide}"
            if parent_key not in self.parent_map:
                self.parent_map[parent_key] = []
            self.parent_map[parent_key].append(chunk.chunk_id)
        
        self._save()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[ChunkMeta, float]]:
        """
        Search for the top_k most similar chunks to the query embedding.
        
        Args:
            query_embedding: np.ndarray of shape (embedding_dim,)
            top_k: Number of results to return
        
        Returns:
            List of (ChunkMeta, similarity_score) tuples, sorted by descending score
        """
        if self.index.ntotal == 0:
            return []
        
        # Prepare query vector
        q = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_metadata):
                continue
            results.append((self.chunk_metadata[idx], float(score)))
        
        return results
    
    def get_parent_context(self, chunk_id: str) -> Optional[str]:
        """Retrieve the parent (macro) text for a given child chunk ID."""
        for meta in self.chunk_metadata:
            if meta.chunk_id == chunk_id:
                return meta.parent_text
        return None
    
    def get_sibling_chunks(self, chunk_id: str) -> List[ChunkMeta]:
        """Get all chunks sharing the same parent section."""
        # Find the parent key for this chunk
        target_meta = None
        for meta in self.chunk_metadata:
            if meta.chunk_id == chunk_id:
                target_meta = meta
                break
        if target_meta is None:
            return []
        
        parent_key = f"{target_meta.source_file}::{target_meta.heading}::{target_meta.page_or_slide}"
        sibling_ids = set(self.parent_map.get(parent_key, []))
        
        return [m for m in self.chunk_metadata if m.chunk_id in sibling_ids]
    
    @property
    def total_chunks(self) -> int:
        return self.index.ntotal if self.index else 0


def delete_session_store(session_id: str):
    """Recursively delete all data for a session."""
    import shutil
    session_dir = os.path.join(DATA_ROOT, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
