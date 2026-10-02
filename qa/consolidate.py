"""
qa/consolidate.py
Q6: Consolidate — turn verified claims into one structured, readable answer.
Runs in-process on the already loaded Qwen 2.5 3B model.
Deterministic claim preparation, LLM rewriting, strict code verification, and fallback.
"""
import json
import os
import re
import time

import ollama

from qa.config import (
    QA_MODEL, NUM_CTX_ANSWER, NUM_PREDICT_CONSOLIDATE,
    CONSOLIDATE_MAX_CLAIMS, CONSOLIDATE_MAX_POINTS,
    CONSOLIDATE_SUPPORT_MIN, DEDUPE_JACCARD, CONSOLIDATE_MAX_WORDS,
)
from qa.schema import Consolidated
from qa.text_utils import content_tokens
from qa import paths as qa_paths

_NUM_RE = re.compile(r'\b\d+(?:\.\d+)?%?\b')

_SYSTEM_PROMPT = (
    "You rewrite verified statements into one clear answer. Output JSON only.\n"
    "You get a QUESTION and numbered STATEMENTS (c1, c2, ...). Use ONLY facts stated in the statements.\n"
    "Rules:\n"
    "1. lead: one sentence that directly answers the question, using the wording of the statements.\n"
    "2. points: up to 4 further statements that add distinct information. Never repeat the lead or each other.\n"
    "   - list questions: one item per point\n"
    "   - procedure questions: one step per point, in order\n"
    "   - comparison questions: one point per side, then one point on the difference only if the statements state it\n"
    "3. Each lead and point cites the statement ids it relies on, for example [\"c1\",\"c3\"].\n"
    "4. Do not add facts, examples, numbers or explanations that are not in the statements.\n"
    "5. Merge duplicates. Drop statements that do not help answer the question.\n"
    "6. Plain, direct sentences. No hedging, no commentary."
)

_FEW_SHOT_USER = (
    "QUESTION: What do mitochondria do?\n"
    "STATEMENTS:\n"
    "c1: Mitochondria produce most of the cell's ATP through cellular respiration.\n"
    "c2: ATP is produced by mitochondria.\n"
    "c3: They have their own small genome."
)

_FEW_SHOT_ASSISTANT = (
    '{"lead":{"text":"Mitochondria produce most of the cell\'s ATP through cellular respiration.","cites":["c1","c2"]},'
    '"points":[{"text":"Mitochondria also have their own small genome.","cites":["c3"]}]}'
)


def _light_stem(word: str) -> str:
    """Lightweight suffix stemmer."""
    for suffix in ("ing", "ies", "es", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "ies":
                return word[:-3] + "y"
            return word[:-len(suffix)]
    return word


def _stemmed_tokens(text: str) -> set[str]:
    """Light-stemmed content tokens set."""
    return {_light_stem(t) for t in content_tokens(text)}


def _jaccard(s1: set, s2: set) -> float:
    """Jaccard similarity between two sets."""
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _extract_markers(support_ids: list[str]) -> list[int]:
    """Derive sorted unique passage numbers (pid) from support sentence ids (e.g. '2.3' -> 2)."""
    pids = set()
    for sid in support_ids:
        try:
            pid_str = str(sid).split(".")[0]
            pids.add(int(pid_str))
        except (ValueError, IndexError):
            pass
    return sorted(pids)


def _determine_style(intent: str, num_points: int) -> str:
    """Choose display style from question intent and points count."""
    if intent in ("list", "comparison"):
        return "bullets"
    elif intent == "procedure":
        return "steps"
    else:
        return "bullets" if num_points >= 3 else "paragraph"


def _make_fallback(capped_claims: list[dict], intent: str, excluded: list, dropped: list, start_time: float) -> dict:
    """Deterministic fallback built from prepared claims."""
    consolidate_s = round(time.time() - start_time, 2)
    if not capped_claims:
        lead = {"text": "No answer could be determined from the documents.", "cites": [], "markers": []}
        return {
            "lead": lead,
            "points": [],
            "style": "paragraph",
            "fallback": True,
            "claims_used": [],
            "excluded": excluded,
            "dropped": dropped,
            "consolidate_s": consolidate_s,
        }

    first = capped_claims[0]
    lead = {
        "text": first.get("text", ""),
        "cites": [first["cid"]],
        "markers": _extract_markers(first.get("support", [])),
    }
    points = []
    for c in capped_claims[1:]:
        points.append({
            "text": c.get("text", ""),
            "cites": [c["cid"]],
            "markers": _extract_markers(c.get("support", [])),
        })

    style = _determine_style(intent, len(points))
    claims_used = [c["cid"] for c in capped_claims]
    return {
        "lead": lead,
        "points": points,
        "style": style,
        "fallback": True,
        "claims_used": claims_used,
        "excluded": excluded,
        "dropped": dropped,
        "consolidate_s": consolidate_s,
    }


def run(session_id: str, run_id: str) -> dict | None:
    """
    Q6: Consolidate verified claims into a cohesive, structured answer.
    Never raises into the orchestrator; returns fallback on any error.
    Returns None if status is not_found.
    """
    start_time = time.time()
    excluded: list[dict] = []
    dropped: list[dict] = []
    capped_claims: list[dict] = []
    intent = "other"

    try:
        # Check files
        v_path = qa_paths.verified_path(session_id, run_id)
        a_path = qa_paths.answer_path(session_id, run_id)
        u_path = qa_paths.understanding_path(session_id, run_id)
        r_path = qa_paths.reranked_path(session_id, run_id)

        # 1. Load understanding
        question = ""
        if os.path.exists(u_path):
            try:
                with open(u_path, "r", encoding="utf-8") as f:
                    u_data = json.load(f)
                intent = u_data.get("intent", "other")
                question = u_data.get("focus", "") or ""
            except Exception:
                pass

        # 2. Load reranked sentence map for verification lookup
        sid_map: dict[str, str] = {}
        if os.path.exists(r_path):
            try:
                with open(r_path, "r", encoding="utf-8") as f:
                    r_data = json.load(f)
                for p in r_data.get("passages", []):
                    for s in p.get("window", []):
                        sid_map[s["sid"]] = s["text"]
            except Exception:
                pass

        # 3. Load final claims from verified.json (or answer.json)
        source_data = {}
        if os.path.exists(v_path):
            with open(v_path, "r", encoding="utf-8") as f:
                source_data = json.load(f)
        elif os.path.exists(a_path):
            with open(a_path, "r", encoding="utf-8") as f:
                source_data = json.load(f)

        if source_data.get("status") == "not_found":
            return None

        raw_claims = source_data.get("claims", [])
        if not raw_claims:
            res = _make_fallback([], intent, excluded, dropped, start_time)
            _write_consolidated(session_id, run_id, res)
            return res

        # ── 4.1 Prepare claims (deterministic) ────────────────────
        kept_raw: list[dict] = []
        for c in raw_claims:
            kind = c.get("kind", "paraphrase")
            is_quote = (kind == "quote")
            is_verified_para = (kind == "paraphrase" and c.get("verified") is True)
            is_high_support = (c.get("support_score", 0.0) >= CONSOLIDATE_SUPPORT_MIN)
            if is_quote or is_verified_para or is_high_support:
                kept_raw.append(c)
            else:
                excluded.append(c)

        if not kept_raw:
            kept_raw = list(raw_claims)
            excluded = []

        # Deduplicate
        deduped: list[dict] = []
        for c in kept_raw:
            c_tokens = _stemmed_tokens(c.get("text", ""))
            merged = False
            for existing in deduped:
                e_tokens = _stemmed_tokens(existing.get("text", ""))
                if _jaccard(c_tokens, e_tokens) >= DEDUPE_JACCARD:
                    # Merge later into earlier: union support ids
                    existing_support = list(existing.get("support", []))
                    for sid in c.get("support", []):
                        if sid not in existing_support:
                            existing_support.append(sid)
                    existing["support"] = existing_support
                    merged = True
                    break
            if not merged:
                deduped.append(dict(c))

        # Cap at CONSOLIDATE_MAX_CLAIMS and assign ids c1..cn
        capped_claims = deduped[:CONSOLIDATE_MAX_CLAIMS]
        claims_by_id: dict[str, dict] = {}
        for i, c in enumerate(capped_claims, start=1):
            cid = f"c{i}"
            c["cid"] = cid
            claims_by_id[cid] = c

        # Single-claim shortcut
        if len(capped_claims) == 1:
            only = capped_claims[0]
            lead_markers = _extract_markers(only.get("support", []))
            lead = {
                "text": only.get("text", ""),
                "cites": [only["cid"]],
                "markers": lead_markers,
            }
            res = {
                "lead": lead,
                "points": [],
                "style": _determine_style(intent, 0),
                "fallback": False,
                "claims_used": [only["cid"]],
                "excluded": excluded,
                "dropped": dropped,
                "consolidate_s": round(time.time() - start_time, 2),
            }
            _write_consolidated(session_id, run_id, res)
            return res

        # ── 4.2 LLM call ──────────────────────────────────────────
        user_prompt_lines = [f"QUESTION: {question}", "STATEMENTS:"]
        for c in capped_claims:
            user_prompt_lines.append(f"{c['cid']}: {c.get('text', '')}")
        user_prompt = "\n".join(user_prompt_lines)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _FEW_SHOT_USER},
            {"role": "assistant", "content": _FEW_SHOT_ASSISTANT},
            {"role": "user", "content": user_prompt},
        ]

        parsed = None
        for attempt in range(2):
            try:
                response = ollama.chat(
                    model=QA_MODEL,
                    messages=messages,
                    format=Consolidated.model_json_schema(),
                    options={
                        "temperature": 0,
                        "seed": 42,
                        "num_ctx": NUM_CTX_ANSWER,
                        "num_predict": NUM_PREDICT_CONSOLIDATE,
                    },
                    keep_alive="0",
                )
                raw_json = response["message"]["content"]
                parsed_data = json.loads(raw_json)
                parsed = Consolidated(**parsed_data)
                break
            except Exception:
                if attempt == 0:
                    continue
                parsed = None

        if not parsed:
            res = _make_fallback(capped_claims, intent, excluded, dropped, start_time)
            _write_consolidated(session_id, run_id, res)
            return res

        # ── 4.3 Verification of the output (code, mandatory) ──────
        def _check_statement(stmt_text: str, stmt_cites: list[str]) -> tuple[bool, list[str]]:
            # 1. Drop cited ids that do not exist
            valid_cites = [cid for cid in stmt_cites if cid in claims_by_id]
            if not valid_cites:
                return False, []

            # 2. Support check & Number check
            ref_tokens: set[str] = set()
            cited_texts: list[str] = []
            for cid in valid_cites:
                c_obj = claims_by_id[cid]
                c_text = c_obj.get("text", "")
                cited_texts.append(c_text)
                ref_tokens.update(_stemmed_tokens(c_text))
                for sid in c_obj.get("support", []):
                    if sid in sid_map:
                        s_text = sid_map[sid]
                        cited_texts.append(s_text)
                        ref_tokens.update(_stemmed_tokens(s_text))

            stmt_tokens = _stemmed_tokens(stmt_text)
            if not stmt_tokens:
                return False, []

            overlap_ratio = len(stmt_tokens & ref_tokens) / len(stmt_tokens)
            if overlap_ratio < CONSOLIDATE_SUPPORT_MIN:
                return False, []

            # Number check
            all_cited_text = " ".join(cited_texts)
            stmt_numbers = _NUM_RE.findall(stmt_text)
            cited_numbers = set(_NUM_RE.findall(all_cited_text))
            for num in stmt_numbers:
                if num not in cited_numbers:
                    return False, []

            return True, valid_cites

        def _markers_for_cites(cites: list[str]) -> list[int]:
            supports: list[str] = []
            for cid in cites:
                if cid in claims_by_id:
                    supports.extend(claims_by_id[cid].get("support", []))
            return _extract_markers(supports)

        # Check lead
        lead_valid, lead_cites = _check_statement(parsed.lead.text, parsed.lead.cites)
        lead_dict: dict | None = None
        if lead_valid:
            lead_dict = {
                "text": parsed.lead.text,
                "cites": lead_cites,
                "markers": _markers_for_cites(lead_cites),
            }
        else:
            dropped.append({"text": parsed.lead.text, "cites": parsed.lead.cites, "reason": "lead_verification_failed"})

        # Check points
        surviving_points: list[dict] = []
        for pt in parsed.points:
            pt_valid, pt_cites = _check_statement(pt.text, pt.cites)
            if not pt_valid:
                dropped.append({"text": pt.text, "cites": pt.cites, "reason": "support_or_num_failed"})
                continue

            pt_tokens = _stemmed_tokens(pt.text)

            # Jaccard with lead (if lead exists)
            if lead_dict:
                lead_tokens = _stemmed_tokens(lead_dict["text"])
                if _jaccard(pt_tokens, lead_tokens) >= DEDUPE_JACCARD:
                    dropped.append({"text": pt.text, "cites": pt_cites, "reason": "duplicate_with_lead"})
                    continue

            # Jaccard with earlier surviving points
            is_dup = False
            for prev_pt in surviving_points:
                prev_tokens = _stemmed_tokens(prev_pt["text"])
                if _jaccard(pt_tokens, prev_tokens) >= DEDUPE_JACCARD:
                    dropped.append({"text": pt.text, "cites": pt_cites, "reason": "duplicate_with_prev_point"})
                    is_dup = True
                    break
            if is_dup:
                continue

            surviving_points.append({
                "text": pt.text,
                "cites": pt_cites,
                "markers": _markers_for_cites(pt_cites),
            })

        # Cap points and check word count
        surviving_points = surviving_points[:CONSOLIDATE_MAX_POINTS]

        # 5. If lead was dropped but point survives, promote first point to lead
        if not lead_dict:
            if surviving_points:
                lead_dict = surviving_points.pop(0)
            else:
                res = _make_fallback(capped_claims, intent, excluded, dropped, start_time)
                _write_consolidated(session_id, run_id, res)
                return res

        # Trim trailing points until total words <= CONSOLIDATE_MAX_WORDS
        def _total_words(ld: dict, pts: list[dict]) -> int:
            return len(ld["text"].split()) + sum(len(p["text"].split()) for p in pts)

        while surviving_points and _total_words(lead_dict, surviving_points) > CONSOLIDATE_MAX_WORDS:
            dropped_pt = surviving_points.pop()
            dropped.append({"text": dropped_pt["text"], "cites": dropped_pt["cites"], "reason": "word_limit_trimmed"})

        # Gather claims_used
        used_set = set(lead_dict["cites"])
        for p in surviving_points:
            used_set.update(p["cites"])
        claims_used = sorted(used_set)

        style = _determine_style(intent, len(surviving_points))
        consolidate_s = round(time.time() - start_time, 2)

        res = {
            "lead": lead_dict,
            "points": surviving_points,
            "style": style,
            "fallback": False,
            "claims_used": claims_used,
            "excluded": excluded,
            "dropped": dropped,
            "consolidate_s": consolidate_s,
        }
        _write_consolidated(session_id, run_id, res)
        return res

    except Exception as e:
        # Fallback on any error
        dropped.append({"text": "", "cites": [], "reason": f"exception: {str(e)}"})
        res = _make_fallback(capped_claims, intent, excluded, dropped, start_time)
        try:
            _write_consolidated(session_id, run_id, res)
        except Exception:
            pass
        return res


def _write_consolidated(session_id: str, run_id: str, data: dict):
    """Write consolidated.json to run directory."""
    if not qa_paths.session_exists(session_id):
        return
    out_path = qa_paths.consolidated_path(session_id, run_id)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
