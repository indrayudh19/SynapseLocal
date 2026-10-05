"""
qa/config.py
Constants for the Document Q&A pipeline.
"""

QA_MODEL = "qwen2.5:3b"
PIPELINE_VERSION = "qa-v1.2"
QA_DEBUG = False                      # keep run artifacts after success if True

NUM_CTX_UNDERSTAND, NUM_CTX_ANSWER = 1024, 3072      # Q4, Q5 and Q6 share NUM_CTX_ANSWER
NUM_PREDICT_UNDERSTAND, NUM_PREDICT_ANSWER, NUM_PREDICT_VERIFY = 220, 360, 12
DEDUPE_JACCARD = 0.8

POLISH_MAX_KEYPOINTS    = 5          # replaces CONSOLIDATE_MAX_CLAIMS
POLISH_MAX_SOURCE_CHARS = 4500       # trim lowest-ranked windows first
POLISH_SUPPORT_MIN      = 0.70       # union token coverage vs key points + source text
POLISH_ANCHOR_MIN       = 0.30       # best single source sentence overlap
POLISH_MIN_WORDS        = 15
POLISH_MAX_HEADINGS     = 3
NUM_PREDICT_POLISH_MAX  = 480
TARGET_WORDS = {"fact":70,"definition":100,"numeric":50,"yes_no":60,"list":140,
                "comparison":200,"procedure":160,"explanation":200,"summary":220,"other":120}

# ── Raw Retrieval (Q2) ──────────────────────────────────────
RAW_RETRIEVE_TOP_K = 30              # max chunks returned from raw text search
BM25_TOPK = 30                       # BM25 candidates per query
DOC_HINT_BOOST = 1.15                # multiplier for chunks from hinted files
PHRASE_MATCH_BONUS = 3.0             # flat bonus for exact key-term substring match
HEADING_MATCH_BONUS = 2.0            # flat bonus for heading containing a key term

# ── LLM Semantic Filter (Q3) ───────────────────────────────
LLM_FILTER_MAX_PASSAGES = 8          # max passages kept after LLM filter
NUM_CTX_FILTER = 2048                # context window for filter stage
NUM_PREDICT_FILTER = 12              # token budget for each relevance verdict

# ── Evidence Windows (shared by Q3 / Q4) ───────────────────
K_PASSAGES, K_PASSAGES_COMPARE = 5, 6
PARENT_MAX_CHARS, WINDOW_MAX_CHARS, MAX_PROMPT_CHARS = 1800, 900, 7000

MAX_CLAIMS, SUPPORT_MIN = 5, 0.5
VERIFY_PARAPHRASES, VERIFY_MAX = True, 3

WORKER_TIMEOUT_S, STALE_LOCK_S = 180, 900
