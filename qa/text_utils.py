"""
qa/text_utils.py
Sentence splitting, normalization, and content tokenization for the QA pipeline.
"""
import re

_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "has", "have", "been",
    "from", "this", "that", "with", "they", "will", "each", "make",
    "like", "been", "into", "some", "then", "than", "them", "very",
    "when", "what", "your", "how", "its", "also", "more", "these",
    "about", "which", "would", "there", "their", "other",
})

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])')
_NEWLINE_RE = re.compile(r'\n+')


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    Uses punctuation boundaries and newlines (slide bullets are individual sentences).
    Drops fragments under 12 characters unless they look like headings.
    """
    # First split on newlines
    parts = _NEWLINE_RE.split(text.strip())
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Then split on sentence boundaries
        sub_parts = _SENT_SPLIT_RE.split(part)
        for s in sub_parts:
            s = s.strip()
            if not s:
                continue
            # Drop very short fragments unless they look like headings
            if len(s) < 12:
                # Keep if it looks like a heading (starts with uppercase, no trailing period)
                if s[0].isupper() and not s.endswith('.'):
                    sentences.append(s)
                continue
            sentences.append(s)
    return sentences


def normalize(text: str) -> str:
    """
    Lowercase, collapse whitespace, unify quotes and hyphens.
    Used for substring and overlap checks.
    """
    t = text.lower()
    # Unify quotes
    t = t.replace('\u2018', "'").replace('\u2019', "'")
    t = t.replace('\u201c', '"').replace('\u201d', '"')
    t = t.replace('\u2013', '-').replace('\u2014', '-')
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def content_tokens(text: str) -> list[str]:
    """
    Lowercase alphanumeric tokens, length > 2, minus stopwords.
    """
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]
