"""
backend/pipeline.py
End-to-End Ingestion & Retrieval Pipeline

Orchestrates the full flow:
  Ingest: parse -> chunk -> embed -> index -> graph build
  Query:  embed query -> FAISS search -> graph neighborhood -> LLM answer
"""
import numpy as np
from typing import List, Dict, Tuple, Optional

from backend.parsers import parse_document, ParsedSection
from backend.chunker import semantic_chunk, SemanticChunk
from backend.vector_store import LocalVectorStore
from backend.graph_engine import KnowledgeGraph
from backend.local_llm import (
    get_embedding_fn,
    get_embedding_dim,
    chat_with_qwen,
    extract_entities_via_llm,
)
from backend.session_manager import update_session_meta


def run_ingestion_pipeline(
    session_id: str,
    files: list,
    model: str = "qwen2.5:7b",
    embed_model: str = "nomic-embed-text",
    k_threshold: float = 1.0,
    progress_callback=None,
):
    """
    Full ingestion pipeline: Parse -> Chunk -> Embed -> FAISS Index -> Graph Build.
    
    Args:
        session_id: Unique session UUID
        files: List of (filename, file_bytes) tuples
        model: Qwen model name for entity extraction
        embed_model: Ollama embedding model name
        k_threshold: Semantic chunking threshold multiplier
        progress_callback: Optional callable(step_name, progress_fraction)
    
    Returns:
        dict with pipeline statistics
    """
    def _progress(step, frac):
        if progress_callback:
            progress_callback(step, frac)
    
    # ── Step 1: Parse all documents ──────────────────────────
    _progress("Parsing documents", 0.0)
    all_sections: List[ParsedSection] = []
    for filename, file_bytes in files:
        sections = parse_document(file_bytes, filename)
        all_sections.extend(sections)
    _progress("Parsing documents", 1.0)
    
    if not all_sections:
        return {"error": "No text could be extracted from the uploaded documents."}
    
    # ── Step 2: Generate embeddings & semantic chunking ──────
    _progress("Generating embeddings", 0.1)
    embed_fn = get_embedding_fn(embed_model)
    embedding_dim = get_embedding_dim(embed_model)
    
    chunks = semantic_chunk(
        sections=all_sections,
        embed_fn=embed_fn,
        k_threshold=k_threshold,
        session_id=session_id
    )
    _progress("Semantic chunking complete", 0.4)
    
    if not chunks:
        return {"error": "Chunking produced no output."}
    
    # ── Step 3: Embed all chunks and store in FAISS ──────────
    _progress("Building vector index", 0.5)
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = embed_fn(chunk_texts)
    
    store = LocalVectorStore(session_id=session_id, embedding_dim=embedding_dim)
    store.add_chunks(chunks, chunk_embeddings)
    _progress("Vector index built", 0.7)
    
    # ── Step 4: Build knowledge graph ────────────────────────
    _progress("Constructing knowledge graph", 0.75)
    
    # Create LLM entity extraction function
    def ollama_entity_fn(text: str) -> str:
        try:
            return extract_entities_via_llm(text, model=model)
        except Exception:
            return ""
    
    kg = KnowledgeGraph(session_id=session_id)
    kg.build_from_chunks(chunks, ollama_fn=ollama_entity_fn)
    _progress("Knowledge graph constructed", 1.0)
    
    # Update session metadata
    stats = {
        "documents_parsed": len(files),
        "sections_extracted": len(all_sections),
        "chunks_created": len(chunks),
        "vector_index_size": store.total_chunks,
        "graph_nodes": kg.graph.number_of_nodes(),
        "graph_edges": kg.graph.number_of_edges(),
        "graph_communities": kg.get_graph_stats()["communities"],
        "embedding_dim": embedding_dim,
    }
    
    update_session_meta(
        session_id,
        document_count=len(files),
        chunk_count=len(chunks),
        is_processed=True
    )
    
    return stats


