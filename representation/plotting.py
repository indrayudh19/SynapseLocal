"""
representation/plotting.py
Shared plotting setup and TF-IDF term extraction helper for representation stages.
Strictly Agg backend, CPU-cheap, deterministic.
"""
import os
from typing import Dict, List


def extract_tfidf_labels(grouped_texts: Dict[int, List[str]], top_n: int = 3) -> Dict[int, str]:
    """
    Extract top N key terms for each group using TF-IDF over concatenated text.
    Shared between Stage 2 (communities) and Stage 3 (clusters).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    labels = {}
    group_keys = list(grouped_texts.keys())
    corpus = [" ".join(grouped_texts[k]).strip() for k in group_keys]

    # Filter out empty texts
    if not any(c for c in corpus):
        return {k: f"Group {k}" for k in group_keys}

    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        )
        X = vec.fit_transform(corpus)
        feature_names = np.array(vec.get_feature_names_out())

        for idx, k in enumerate(group_keys):
            row = X.getrow(idx).toarray().flatten()
            if row.sum() == 0 or len(feature_names) == 0:
                labels[k] = f"Group {k}"
                continue
            top_indices = np.argsort(row)[::-1][:top_n]
            top_terms = [feature_names[i] for i in top_indices if row[i] > 0]
            labels[k] = " / ".join(top_terms) if top_terms else f"Group {k}"
    except Exception:
        for k in group_keys:
            labels[k] = f"Group {k}"

    return labels


def setup_matplotlib():
    """Ensure matplotlib uses Agg backend and consistent styling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    return plt
