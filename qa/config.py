"""
qa/config.py
Constants for the Document Q&A pipeline.
"""

QA_MODEL = "qwen2.5:3b"
PIPELINE_VERSION = "qa-v1.1"
QA_DEBUG = False                      # keep run artifacts after success if True

NUM_CTX_UNDERSTAND, NUM_CTX_ANSWER = 1024, 3072      # Q4 and Q5 share NUM_CTX_ANSWER
NUM_PREDICT_UNDERSTAND, NUM_PREDICT_ANSWER, NUM_PREDICT_VERIFY = 220, 360, 12
NUM_PREDICT_CONSOLIDATE = 260
CONSOLIDATE_MAX_CLAIMS = 5
CONSOLIDATE_MAX_POINTS = 4
CONSOLIDATE_SUPPORT_MIN = 0.7      # stricter than SUPPORT_MIN: this stage only rewrites verified material
DEDUPE_JACCARD = 0.8
CONSOLIDATE_MAX_WORDS = 120

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
