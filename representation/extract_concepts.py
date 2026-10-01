"""
representation/extract_concepts.py
Stage 1: LLM Concept Extraction via Ollama (Qwen 2.5 3B).
Strictly sequential execution, constrained JSON output, grounding validation.
"""
import gc
import hashlib
import json
import os
import re
import time
from typing import Callable, Dict, List, Optional, Set

from representation import paths as rep_paths
from representation.concept_schema import (
    EXTRACT_MODEL,
    PROMPT_VERSION,
    MAX_CHARS,
    MIN_CHARS,
    MAX_CONCEPTS,
    MAX_RELATIONS,
    Extraction,
)

SYSTEM_PROMPT = """You extract a concept map from study notes. Output JSON only.
Rules:
1. concepts: key technical terms from the TEXT, each 1-4 words, noun phrases, at most 8.
   Prefer terms that are defined, categorized, or contrasted in the text.
   Never output author names, citations, numbers, section names, or generic words
   (paper, method, results, approach, model, figure, table).
2. relations: only between concepts you listed, at most 8. Allowed values:
   - type_of:    A is a kind/subtype of B
   - part_of:    A is a component of B
   - example_of: A is an instance of B
   - uses:       A uses or depends on B
   - related_to: A is closely related to B (only if none of the above fits)
3. For type_of, part_of, example_of the source is ALWAYS the narrower/smaller concept.
4. Use only information stated in the TEXT."""

FEW_SHOT_USER = "TEXT: Relational databases store data in tables. PostgreSQL and MySQL are popular examples. Indexes speed up queries on a table."
FEW_SHOT_ASSISTANT = json.dumps({
    "concepts": [
        {"name": "relational database"},
        {"name": "table"},
        {"name": "PostgreSQL"},
        {"name": "MySQL"},
        {"name": "index"},
        {"name": "query"},
    ],
    "relations": [
        {"source": "PostgreSQL", "relation": "example_of", "target": "relational database"},
        {"source": "MySQL", "relation": "example_of", "target": "relational database"},
        {"source": "table", "relation": "part_of", "target": "relational database"},
        {"source": "index", "relation": "uses", "target": "table"},
    ],
})

GENERIC_STOPLIST = {
    "paper", "method", "approach", "model", "results",
    "figure", "table", "section", "data", "system", "work",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "that", "this", "it", "its", "into", "their", "such",
}

REF_REGEX = re.compile(r'et al\.|arXiv|pp\.|\b(19|20)\d{2}\b.*\.$|doi', re.IGNORECASE)


def _truncate_sentence_boundary(text: str, max_chars: int) -> str:
    """Truncate text to max_chars at a sentence boundary."""
    if len(text) <= max_chars:
        return text
    sub = text[:max_chars]
    # Look for last sentence terminator
    matches = list(re.finditer(r'([.!?])(?:\s+|$)', sub))
    if matches and matches[-1].end() >= MIN_CHARS:
        return sub[:matches[-1].end()].strip()
    # Fall back to last whitespace
    last_space = sub.rfind(" ")
    if last_space >= MIN_CHARS:
        return sub[:last_space].strip()
    return sub.strip()


def _is_boilerplate(text: str, seen_hashes: Set[str]) -> bool:
    """Check if chunk text should be skipped."""
    if len(text) < MIN_CHARS:
        return True
    
    # Digit ratio check
    digits = sum(1 for c in text if c.isdigit())
    if len(text) > 0 and (digits / len(text)) > 0.35:
        return True
    
    # Reference list check
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        matches = sum(1 for line in lines if REF_REGEX.search(line))
        if (matches / len(lines)) > 0.30:
            return True
            
    # Near-duplicate check
    norm = re.sub(r'\s+', ' ', text.strip().lower())
    h = hashlib.md5(norm.encode("utf-8")).hexdigest()
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False


def _is_grounded_concept(concept_name: str, chunk_text_lower: str) -> bool:
    """Validate concept against chunk text and stoplist."""
    norm = concept_name.strip().lower()
    if not norm or norm in GENERIC_STOPLIST:
        return False
    
    # Word tokens
    tokens = [w for w in re.findall(r'\b\w+\b', norm) if w not in STOPWORDS]
    if not tokens:
        return False
    
    found = sum(1 for t in tokens if t in chunk_text_lower)
    if (found / len(tokens)) >= 0.60:
        return True
    
    # Acronym check (e.g. uppercase acronym in original text)
    if re.fullmatch(r'[A-Za-z0-9]{2,6}', concept_name.strip()):
        if re.search(r'\b' + re.escape(concept_name.strip()) + r'\b', chunk_text_lower, re.IGNORECASE):
            return True
            
    return False


def run(
    session_id: str,
    max_chunks: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Execute Stage 1: Extract concepts and relations from parent chunks using Ollama.
    Sequential execution, handles checkpointing and resume.
    """
    import ollama

    chunks_file = rep_paths.chunks_jsonl_path(session_id)
    if not os.path.exists(chunks_file):
        return {"status": "error", "message": f"chunks.jsonl not found for session {session_id}"}

    raw_path = rep_paths.concepts_raw_path(session_id)
    log_path = rep_paths.extract_log_path(session_id)
    hash_path = rep_paths.extract_hash_path(session_id)

    # Check model presence via ollama.list()
    try:
        models_resp = ollama.list()
        avail_names = []
        if hasattr(models_resp, "models"):
            avail_names = [m.model for m in models_resp.models]
        elif isinstance(models_resp, dict) and "models" in models_resp:
            avail_names = [m.get("model", m.get("name", "")) for m in models_resp["models"]]
        
        has_model = any(EXTRACT_MODEL in name for name in avail_names)
        if not has_model:
            return {
                "status": "error",
                "message": f"Model '{EXTRACT_MODEL}' not found in Ollama. Please run 'ollama pull {EXTRACT_MODEL}'.",
            }
    except Exception as e:
        return {"status": "error", "message": f"Cannot connect to Ollama: {str(e)}"}

    # Content hash checking for complete run cache
    mtime = str(os.path.getmtime(chunks_file))
    cur_hash = hashlib.md5(f"{EXTRACT_MODEL}:{PROMPT_VERSION}:{max_chunks}:{mtime}".encode("utf-8")).hexdigest()
    if os.path.exists(hash_path) and os.path.exists(raw_path) and os.path.exists(log_path):
        try:
            with open(hash_path, "r", encoding="utf-8") as f:
                saved_hash = f.read().strip()
            if saved_hash == cur_hash:
                with open(log_path, "r", encoding="utf-8") as f:
                    cached_log = json.load(f)
                return {"status": "ok", "cached": True, **cached_log}
        except Exception:
            pass

    with rep_paths.session_lock(session_id):
        # 1. Read chunks and aggregate into parent chunks
        parent_map: Dict[str, dict] = {}
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                pid = c.get("parent_id") or c.get("id")
                if pid not in parent_map:
                    parent_map[pid] = {
                        "parent_id": pid,
                        "chunk_id": c.get("id", pid),
                        "source_file": c.get("source_file", ""),
                        "source_type": c.get("source_type", ""),
                        "texts": [],
                    }
                parent_map[pid]["texts"].append(c.get("text", ""))

        # 2. Filter boilerplate and truncate
        eligible_chunks: List[dict] = []
        seen_hashes: Set[str] = set()
        skipped_boilerplate = 0

        for pid, pdata in parent_map.items():
            full_text = " ".join(pdata["texts"]).strip()
            if _is_boilerplate(full_text, seen_hashes):
                skipped_boilerplate += 1
                continue
            trunc_text = _truncate_sentence_boundary(full_text, MAX_CHARS)
            eligible_chunks.append({
                "parent_id": pid,
                "chunk_id": pdata["chunk_id"],
                "source_file": pdata["source_file"],
                "source_type": pdata["source_type"],
                "text": trunc_text,
            })

        chunks_total = len(eligible_chunks)
        if max_chunks is not None and max_chunks > 0:
            eligible_chunks = eligible_chunks[:max_chunks]

        # 3. Check existing checkpoint for resume
        processed_chunk_ids: Set[str] = set()
        concepts_kept = 0
        concepts_dropped = 0
        relations_kept = 0
        relations_dropped = 0
        failed = 0

        # Read checkpoint if exists
        if os.path.exists(raw_path):
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        processed_chunk_ids.add(rec.get("chunk_id"))
                        c_list = rec.get("concepts", [])
                        r_list = rec.get("relations", [])
                        concepts_kept += len(c_list)
                        relations_kept += len(r_list)
                        if rec.get("status") == "failed":
                            failed += 1
            except Exception:
                processed_chunk_ids.clear()
                with open(raw_path, "w", encoding="utf-8") as f:
                    pass

        schema = Extraction.model_json_schema()
        consecutive_failures = 0
        total_time = 0.0
        newly_processed = 0

        total_to_process = len(eligible_chunks)
        if on_progress:
            on_progress(len(processed_chunk_ids), total_to_process)

        # Append mode for streaming writes
        with open(raw_path, "a", encoding="utf-8") as out_f:
            for idx, chunk in enumerate(eligible_chunks):
                cid = chunk["chunk_id"]
                if cid in processed_chunk_ids:
                    continue

                t0 = time.time()
                chunk_text = chunk["text"]
                chunk_text_lower = chunk_text.lower()

                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": FEW_SHOT_USER},
                    {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
                    {"role": "user", "content": f"TEXT: {chunk_text}"},
                ]

                parsed: Optional[Extraction] = None
                # Call Ollama with 1 retry
                for attempt in range(2):
                    try:
                        resp = ollama.chat(
                            model=EXTRACT_MODEL,
                            messages=msgs,
                            format=schema,
                            options={
                                "temperature": 0,
                                "seed": 42,
                                "num_ctx": 2048,
                                "num_predict": 512,
                            },
                            keep_alive="10m",
                        )
                        raw_content = resp["message"]["content"]
                        parsed = Extraction.model_validate_json(raw_content)
                        break
                    except Exception:
                        if attempt == 1:
                            parsed = None

                elapsed = time.time() - t0
                total_time += elapsed
                newly_processed += 1

                if parsed is None:
                    consecutive_failures += 1
                    failed += 1
                    record = {
                        "chunk_id": cid,
                        "parent_id": chunk["parent_id"],
                        "source_file": chunk["source_file"],
                        "source_type": chunk["source_type"],
                        "status": "failed",
                        "concepts": [],
                        "relations": [],
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    if consecutive_failures >= 5:
                        # Abort after 5 consecutive failures
                        try:
                            ollama.generate(model=EXTRACT_MODEL, prompt="", keep_alive=0)
                        except Exception:
                            pass
                        return {
                            "status": "error",
                            "message": "5 consecutive LLM extraction failures. Check if Ollama is running or pull qwen2.5:3b.",
                        }
                    continue

                consecutive_failures = 0

                # Grounding validation
                valid_concepts_map: Dict[str, str] = {}  # norm -> original display name
                for c in parsed.concepts:
                    c_name = c.name.strip()
                    if _is_grounded_concept(c_name, chunk_text_lower):
                        valid_concepts_map[c_name.lower()] = c_name
                    else:
                        concepts_dropped += 1

                kept_c_names = set(valid_concepts_map.keys())
                kept_concepts = [{"name": name} for name in valid_concepts_map.values()]
                concepts_kept += len(kept_concepts)

                kept_relations = []
                for r in parsed.relations:
                    s_norm = r.source.strip().lower()
                    t_norm = r.target.strip().lower()
                    if s_norm in kept_c_names and t_norm in kept_c_names and s_norm != t_norm:
                        kept_relations.append({
                            "source": valid_concepts_map[s_norm],
                            "relation": r.relation,
                            "target": valid_concepts_map[t_norm],
                        })
                    else:
                        relations_dropped += 1
                relations_kept += len(kept_relations)

                record = {
                    "chunk_id": cid,
                    "parent_id": chunk["parent_id"],
                    "source_file": chunk["source_file"],
                    "source_type": chunk["source_type"],
                    "status": "ok",
                    "concepts": kept_concepts,
                    "relations": kept_relations,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                processed_chunk_ids.add(cid)

                if on_progress:
                    on_progress(len(processed_chunk_ids), total_to_process)

        # Unload the model
        try:
            ollama.generate(model=EXTRACT_MODEL, prompt="", keep_alive=0)
        except Exception:
            pass

        gc.collect()

        avg_sec = round(total_time / max(1, newly_processed), 2) if newly_processed > 0 else 0.0

        stats = {
            "status": "ok",
            "chunks_total": chunks_total,
            "processed": len(processed_chunk_ids),
            "skipped_boilerplate": skipped_boilerplate,
            "failed": failed,
            "avg_seconds_per_chunk": avg_sec,
            "concepts_kept": concepts_kept,
            "concepts_dropped": concepts_dropped,
            "relations_kept": relations_kept,
            "relations_dropped": relations_dropped,
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(cur_hash)

        return stats
