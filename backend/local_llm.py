"""
backend/local_llm.py
Local LLM Interface: Ollama + Qwen

Provides embedding and chat completion via locally running Ollama server.
All inference is 100% local and offline — no cloud APIs.
"""
import numpy as np
from typing import List, Optional
import ollama


def get_embedding_fn(model_name: str = "nomic-embed-text"):
    """
    Return an embedding function that uses Ollama's local embedding model.
    
    The returned function accepts a list of strings and returns np.ndarray (n, dim).
    Uses nomic-embed-text (768-dim) by default for high-quality local embeddings.
    Falls back to sentence-transformers if Ollama embedding fails.
    """
    def embed_via_ollama(texts: List[str]) -> np.ndarray:
        try:
            response = ollama.embed(model=model_name, input=texts)
            return np.array(response["embeddings"], dtype=np.float32)
        except Exception:
            # Fallback: use sentence-transformers locally
            return _embed_via_sentence_transformers(texts)
    
    return embed_via_ollama


def _embed_via_sentence_transformers(texts: List[str]) -> np.ndarray:
    """Fallback: use sentence-transformers for embedding."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def get_embedding_dim(model_name: str = "nomic-embed-text") -> int:
    """Probe the embedding dimension by embedding a test string."""
    try:
        response = ollama.embed(model=model_name, input=["test"])
        return len(response["embeddings"][0])
    except Exception:
        # Fallback to sentence-transformers dimension
        return 384  # all-MiniLM-L6-v2 dimension


def chat_with_qwen(
    prompt: str,
    model: str = "qwen2.5:7b",
    system_prompt: Optional[str] = None,
    context_chunks: Optional[List[str]] = None,
    temperature: float = 0.3
) -> str:
    """
    Send a prompt to the local Qwen model via Ollama.
    
    Args:
        prompt: User question
        model: Ollama model name
        system_prompt: System instructions for the model
        context_chunks: Retrieved context chunks to include in the prompt
        temperature: Sampling temperature
    
    Returns:
        Model response text
    """
    if system_prompt is None:
        system_prompt = (
            "You are SynapseLocal, a precise research assistant. "
            "Answer questions using ONLY the provided context. "
            "If the context does not contain the answer, say so clearly. "
            "Cite which document sections you referenced."
        )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if context_chunks:
        context_text = "\n\n---\n\n".join(context_chunks)
        messages.append({
            "role": "user",
            "content": (
                f"Context from your knowledge base:\n\n{context_text}\n\n"
                f"---\n\nQuestion: {prompt}\n\n"
                "Answer based on the context above. Cite the relevant sections."
            )
        })
    else:
        messages.append({"role": "user", "content": prompt})
    
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature}
    )
    
    return response["message"]["content"]


def extract_entities_via_llm(text: str, model: str = "qwen2.5:7b") -> str:
    """
    Use Qwen to extract entities from text (used by graph_engine).
    Returns raw comma-separated entity string.
    """
    prompt = (
        "Extract the key concepts, entities, and technical terms from the following text. "
        "Return ONLY a comma-separated list of terms, nothing else.\n\n"
        f"Text: {text[:2000]}\n\n"
        "Entities:"
    )
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    
    return response["message"]["content"]
