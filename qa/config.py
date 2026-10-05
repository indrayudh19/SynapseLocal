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

DENSE_TOPK, BM25_TOPK, FUSED_CAND, SUBQ_TOPK = 30, 30, 24, 12
RRF_K = 60
LIST_WEIGHTS = {"dense_original": 1.0, "dense_rewrite": 0.7, "dense_hyde": 0.7,
                "bm25_terms": 1.0, "bm25_question": 0.7}
DOC_HINT_BOOST = 1.15

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_MAXLEN, RERANK_BATCH = 384, 16
K_PASSAGES, K_PASSAGES_COMPARE = 5, 6
PARENT_MAX_CHARS, WINDOW_MAX_CHARS, MAX_PROMPT_CHARS = 1800, 900, 7000
NO_ANSWER_SCORE = -4.0               # starting value; best cross-encoder logit below this -> not found
COVERAGE_MARGIN = 6.0                # per-document coverage rule (section 7)

MAX_CLAIMS, SUPPORT_MIN = 5, 0.5
VERIFY_PARAPHRASES, VERIFY_MAX = True, 3

WORKER_TIMEOUT_S, STALE_LOCK_S = 180, 900
