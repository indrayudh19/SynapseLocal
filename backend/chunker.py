"""
backend/chunker.py
Module 1.2: Semantic Chunking via Cosine Distance Variance

Splits parsed sections into semantically coherent chunks by detecting
topic shifts using cosine distance between consecutive sentence embeddings.

Algorithm:
  1. Split text into sentences S = {s_1, ..., s_n}
  2. Embed each sentence via a local transformer: E(s_i)
  3. Compute cosine distance between consecutive embeddings:
     d_i = 1 - cos(E(s_i), E(s_{i+1}))
  4. Place chunk boundary where d_i > mu + k * sigma (rolling threshold)
"""
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SemanticChunk:
    """A semantically coherent text chunk with metadata."""
    chunk_id: str
    text: str
    source_file: str
    heading: str
    page_or_slide: int
    parent_text: str  # The full parent section text for macro-context retrieval
    sentence_indices: tuple  # (start_idx, end_idx) in the original sentence list


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based boundary detection."""
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out very short fragments (< 10 chars)
    return [s.strip() for s in raw if len(s.strip()) >= 10]


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance: 1 - cosine_similarity."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 1.0
    return 1.0 - (dot / norm)


def semantic_chunk(
    sections: list,
    embed_fn,
    k_threshold: float = 1.0,
    min_chunk_sentences: int = 2,
    max_chunk_sentences: int = 15,
    session_id: str = ""
) -> List[SemanticChunk]:
    """
    Perform semantic chunking on parsed sections.
    
    Args:
        sections: List of ParsedSection objects from parsers.py
        embed_fn: Callable that takes List[str] and returns np.ndarray of shape (n, dim)
        k_threshold: Number of standard deviations above mean for boundary detection
        min_chunk_sentences: Minimum sentences per chunk
        max_chunk_sentences: Maximum sentences per chunk (hard cap)
        session_id: Session identifier for chunk ID prefixing
    
    Returns:
        List of SemanticChunk objects
    """
    all_chunks = []
    global_chunk_idx = 0

    for section in sections:
        sentences = _split_sentences(section.content)
        
        if len(sentences) == 0:
            continue
        
        # If very few sentences, emit as a single chunk
        if len(sentences) <= min_chunk_sentences:
            all_chunks.append(SemanticChunk(
                chunk_id=f"{session_id}_chunk_{global_chunk_idx:04d}",
                text=" ".join(sentences),
                source_file=section.source_file,
                heading=section.heading,
                page_or_slide=section.page_or_slide,
                parent_text=section.content,
                sentence_indices=(0, len(sentences) - 1)
            ))
            global_chunk_idx += 1
            continue

        # Step 2: Embed all sentences in this section
        embeddings = embed_fn(sentences)  # shape: (n, dim)
        
        # Step 3: Compute cosine distances between consecutive sentences
        distances = []
        for i in range(len(sentences) - 1):
            d = _cosine_distance(embeddings[i], embeddings[i + 1])
            distances.append(d)
        
        distances = np.array(distances)
        
        # Step 4: Compute rolling threshold mu + k*sigma
        # Use a global threshold for the section
        mu = np.mean(distances)
        sigma = np.std(distances)
        threshold = mu + k_threshold * sigma

        # Detect boundary indices where distance exceeds threshold
        boundaries = []
        for i, d in enumerate(distances):
            if d > threshold:
                boundaries.append(i + 1)  # boundary AFTER sentence i

        # Build chunks from boundary indices
        boundary_set = set(boundaries)
        chunk_start = 0
        
        for sent_idx in range(1, len(sentences)):
            current_chunk_len = sent_idx - chunk_start
            
            # Force split at max length or at detected boundary
            at_boundary = sent_idx in boundary_set and current_chunk_len >= min_chunk_sentences
            at_max = current_chunk_len >= max_chunk_sentences
            
            if at_boundary or at_max:
                chunk_text = " ".join(sentences[chunk_start:sent_idx])
                all_chunks.append(SemanticChunk(
                    chunk_id=f"{session_id}_chunk_{global_chunk_idx:04d}",
                    text=chunk_text,
                    source_file=section.source_file,
                    heading=section.heading,
                    page_or_slide=section.page_or_slide,
                    parent_text=section.content,
                    sentence_indices=(chunk_start, sent_idx - 1)
                ))
                global_chunk_idx += 1
                chunk_start = sent_idx

        # Emit remaining sentences as final chunk
        if chunk_start < len(sentences):
            chunk_text = " ".join(sentences[chunk_start:])
            all_chunks.append(SemanticChunk(
                chunk_id=f"{session_id}_chunk_{global_chunk_idx:04d}",
                text=chunk_text,
                source_file=section.source_file,
                heading=section.heading,
                page_or_slide=section.page_or_slide,
                parent_text=section.content,
                sentence_indices=(chunk_start, len(sentences) - 1)
            ))
            global_chunk_idx += 1

    return all_chunks
