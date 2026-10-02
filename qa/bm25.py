"""
qa/bm25.py
Lightweight BM25 implementation using numpy and scipy sparse CSR.
No external BM25 library. Cached to disk per session.
"""
import hashlib
import json
import os

import numpy as np
from scipy import sparse

from qa.text_utils import content_tokens
from qa import paths as qa_paths


def _compute_chunks_hash(chunk_texts: list[str]) -> str:
    """SHA-256 of all chunk texts concatenated."""
    h = hashlib.sha256()
    for t in chunk_texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()


class BM25Index:
    """
    BM25 index over a list of chunk texts.
    k1=1.5, b=0.75. Stores a sparse TF matrix and IDF vector.
    """

    def __init__(self, vocab: dict[str, int], idf: np.ndarray,
                 tf_matrix: sparse.csr_matrix, avgdl: float, doc_lens: np.ndarray):
        self.vocab = vocab
        self.idf = idf              # shape (V,)
        self.tf_matrix = tf_matrix  # shape (N, V)
        self.avgdl = avgdl
        self.doc_lens = doc_lens    # shape (N,)

    def query(self, text: str, top_k: int = 30) -> list[tuple[int, float]]:
        """Return top_k (chunk_index, score) pairs sorted by descending BM25 score."""
        tokens = content_tokens(text)
        if not tokens:
            return []

        k1, b = 1.5, 0.75
        query_indices = [self.vocab[t] for t in tokens if t in self.vocab]
        if not query_indices:
            return []

        # Gather TF columns for query terms: shape (N, q)
        tf_q = self.tf_matrix[:, query_indices].toarray()
        idf_q = self.idf[query_indices]  # shape (q,)

        # BM25 scoring
        dl = self.doc_lens.reshape(-1, 1)
        denom = tf_q + k1 * (1 - b + b * dl / self.avgdl)
        numer = tf_q * (k1 + 1)
        scores_per_term = idf_q * (numer / denom)  # (N, q)
        scores = scores_per_term.sum(axis=1)        # (N,)

        top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


def build_index(chunk_texts: list[str], chunk_headings: list[str]) -> BM25Index:
    """Build a BM25 index from chunk texts and headings."""
    # Tokenize: heading + " " + text
    doc_tokens_list = []
    for heading, text in zip(chunk_headings, chunk_texts):
        combined = (heading + " " + text) if heading else text
        doc_tokens_list.append(content_tokens(combined))

    # Build vocabulary
    vocab: dict[str, int] = {}
    for doc_tokens in doc_tokens_list:
        for token in doc_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    V = len(vocab)
    N = len(doc_tokens_list)
    if V == 0 or N == 0:
        return BM25Index(vocab, np.zeros(0), sparse.csr_matrix((N, 0)), 1.0, np.zeros(N))

    # Build sparse TF matrix
    rows, cols, data = [], [], []
    doc_lens = np.zeros(N, dtype=np.float32)
    for i, doc_tokens in enumerate(doc_tokens_list):
        doc_lens[i] = len(doc_tokens)
        tf: dict[int, int] = {}
        for token in doc_tokens:
            idx = vocab[token]
            tf[idx] = tf.get(idx, 0) + 1
        for idx, count in tf.items():
            rows.append(i)
            cols.append(idx)
            data.append(count)

    tf_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(N, V), dtype=np.float32)
    avgdl = float(doc_lens.mean()) if N > 0 else 1.0

    # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    df = np.array((tf_matrix > 0).sum(axis=0), dtype=np.float32).flatten()
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    return BM25Index(vocab, idf, tf_matrix, avgdl, doc_lens)


def get_or_build(session_id: str, chunk_texts: list[str],
                 chunk_headings: list[str]) -> BM25Index:
    """
    Load cached BM25 index if the chunk store hash matches, otherwise rebuild and cache.
    """
    current_hash = _compute_chunks_hash(chunk_texts)
    hash_file = qa_paths.bm25_hash_path(session_id)
    index_file = qa_paths.bm25_index_path(session_id)
    vocab_file = qa_paths.bm25_vocab_path(session_id)

    # Try loading cache
    if (os.path.exists(hash_file) and os.path.exists(index_file)
            and os.path.exists(vocab_file)):
        try:
            with open(hash_file, "r", encoding="utf-8") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                with open(vocab_file, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                npz = np.load(index_file, allow_pickle=True)
                idf = npz["idf"]
                doc_lens = npz["doc_lens"]
                avgdl = float(npz["avgdl"])
                tf_data = npz["tf_data"]
                tf_indices = npz["tf_indices"]
                tf_indptr = npz["tf_indptr"]
                tf_shape = tuple(npz["tf_shape"])
                tf_matrix = sparse.csr_matrix((tf_data, tf_indices, tf_indptr), shape=tf_shape)
                return BM25Index(vocab, idf, tf_matrix, avgdl, doc_lens)
        except Exception:
            pass  # Rebuild on any error

    # Build fresh
    idx = build_index(chunk_texts, chunk_headings)

    # Cache to disk
    qa_dir = qa_paths.qa_dir(session_id)
    os.makedirs(qa_dir, exist_ok=True)

    np.savez(index_file,
             idf=idx.idf,
             doc_lens=idx.doc_lens,
             avgdl=np.array([idx.avgdl]),
             tf_data=idx.tf_matrix.data,
             tf_indices=idx.tf_matrix.indices,
             tf_indptr=idx.tf_matrix.indptr,
             tf_shape=np.array(idx.tf_matrix.shape))

    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(idx.vocab, f)

    with open(hash_file, "w", encoding="utf-8") as f:
        f.write(current_hash)

    return idx
