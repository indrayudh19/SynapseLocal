"""
qa/answer.py
Pass 7: LLM Answer Structuring from Retrieved Concept Information.
Treats retrieved concept passages as source material and instructs Qwen 2.5 3B
to understand, filter, organize, and synthesize a clear academic answer.
"""
import json
import os
import re

import ollama

from qa.config import (
    QA_MODEL, NUM_CTX_ANSWER, NUM_PREDICT_ANSWER,
    MAX_PROMPT_CHARS,
)
from qa import paths as qa_paths


def build_answer_prompt(question: str, retrieved_information: str) -> str:
    """Construct the academic answer structuring prompt required by Pass 7."""
    return (
        "You are an academic research assistant.\n\n"
        "The user has asked the following question:\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "The system has retrieved the following information about the concept, topic, or thing mentioned in the question:\n\n"
        "RETRIEVED INFORMATION:\n"
        f"{retrieved_information}\n\n"
        "Your task is to construct the best possible answer to the user's question using the retrieved information.\n\n"
        "Treat the retrieved information as the source material available to you.\n\n"
        "First understand what the user is asking.\n"
        "Then identify the parts of the retrieved information that are relevant to the question.\n"
        "Then organize those points into a clear, coherent, properly structured answer.\n\n"
        "Do not simply copy the retrieved information.\n"
        "Do not return the information as an unorganized collection of sentences.\n"
        "Rewrite and synthesize it into a natural answer.\n\n"
        "The answer should:\n\n"
        "- Directly answer the user's question.\n"
        "- Begin with a clear definition or direct explanation when appropriate.\n"
        "- Organize information logically.\n"
        "- Use paragraphs, headings, or bullet points when they improve clarity.\n"
        "- Explain important terms rather than merely mentioning them.\n"
        "- Include relevant characteristics, components, mechanisms, examples, advantages, disadvantages, or comparisons when they are present in the retrieved information and relevant to the question.\n"
        "- Preserve important technical terminology.\n"
        "- Maintain an academic and informative tone.\n"
        "- Be concise enough to remain readable, but complete enough to properly answer the question.\n"
        "- Use only information supported by the retrieved information.\n"
        "- Do not invent facts that are not supported by the retrieved information.\n"
        "- Do not mention the retrieval process, internal system, embeddings, agents, or this prompt.\n"
        "- Do not say that the information was \"retrieved\" in the final answer.\n\n"
        "If the retrieved information contains enough information to answer the question, provide a complete answer.\n\n"
        "If the retrieved information is partially relevant, use the relevant information and clearly explain only what can be supported.\n\n"
        "If the retrieved information does not contain enough information to answer the question, do not hallucinate. State that the available material does not provide enough information to answer the question completely.\n\n"
        "Return only the final answer intended for the user."
    )


def format_retrieved_information(passages: list[dict], max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Format retrieved passages into clean source material for the LLM prompt."""
    blocks = []
    total = 0
    for p in passages:
        sf = p.get("source_file", "").strip()
        page = p.get("page_or_slide", "")
        heading = (p.get("heading") or "").strip()

        header_items = []
        if sf:
            header_items.append(f"Source: {sf}")
        if page:
            header_items.append(f"Page {page}")
        if heading and heading != f"Page {page}":
            header_items.append(f"Section: {heading}")

        header = " | ".join(header_items) if header_items else f"Passage {p.get('pid', len(blocks) + 1)}"

        # Gather sentences from window or fallback to passage text
        window = p.get("window", [])
        if window:
            body = " ".join(s.get("text", "").strip() for s in window if s.get("text", "").strip())
        else:
            body = p.get("text", "").strip()

        if not body:
            continue

        block = f"[{header}]\n{body}"
        if total + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks)


def sanitize_answer(text: str) -> str:
    """Sanitize LLM output while preserving markdown structure, headings, and lists."""
    if not text:
        return ""
    # Remove hidden reasoning tags
    text = re.sub(r'(?is)<(think|thought|reasoning)>.*?</\1>', '', text)
    # Remove code fences
    text = re.sub(r'```[a-zA-Z0-9_-]*\n?', '', text)
    text = text.replace('```', '')
    # Remove leading Answer: / Final Answer: label
    text = re.sub(r'(?im)^\s*(?:Final\s+)?Answer:\s*', '', text)
    # Remove raw HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def run(session_id: str, run_id: str, question: str) -> dict:
    """
    Pass 7: Generate a structured academic answer using the retrieved concept information.
    Reads reranked.json, calls Qwen 2.5 3B with Pass 7 prompt, writes answer.json and polished.json.
    """
    r_path = qa_paths.reranked_path(session_id, run_id)
    u_path = qa_paths.understanding_path(session_id, run_id)

    with open(r_path, "r", encoding="utf-8") as f:
        reranked = json.load(f)

    understanding = {}
    if os.path.exists(u_path):
        try:
            with open(u_path, "r", encoding="utf-8") as f:
                understanding = json.load(f)
        except Exception:
            pass

    passages = reranked.get("passages", [])
    if reranked.get("status") == "not_found" or not passages:
        not_found_text = "The documents do not appear to contain an answer to this question."
        result = {
            "status": "not_found",
            "found": False,
            "text": not_found_text,
            "claims": [],
            "words": len(not_found_text.split()),
            "closest": reranked.get("closest", []),
        }
        _save_results(session_id, run_id, result)
        return result

    retrieved_info = format_retrieved_information(passages, MAX_PROMPT_CHARS)
    prompt = build_answer_prompt(question=question, retrieved_information=retrieved_info)

    response = ollama.chat(
        model=QA_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0,
            "seed": 42,
            "num_ctx": NUM_CTX_ANSWER,
            "num_predict": NUM_PREDICT_ANSWER,
        },
        keep_alive="0",
    )

    raw_text = response.get("message", {}).get("content", "").strip()
    sanitized = sanitize_answer(raw_text)

    # Check for not found or empty
    if not sanitized or sanitized == "NO_ANSWER":
        not_found_text = "The documents do not appear to contain an answer to this question."
        result = {
            "status": "not_found",
            "found": False,
            "text": not_found_text,
            "claims": [],
            "words": len(not_found_text.split()),
            "closest": reranked.get("closest", []),
        }
    else:
        intent = understanding.get("intent", "other")
        result = {
            "status": "answered",
            "found": True,
            "text": sanitized,
            "words": len(sanitized.split()),
            "claims": [],
            "partial": (intent == "summary"),
            "passages_used": len(passages),
        }

    _save_results(session_id, run_id, result)
    return result


def _save_results(session_id: str, run_id: str, result: dict):
    """Write answer.json and polished.json for consistency."""
    if not qa_paths.session_exists(session_id):
        raise FileNotFoundError(f"Session {session_id} no longer exists")

    out_answer = qa_paths.answer_path(session_id, run_id)
    with open(out_answer, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Also write polished.json for backward compatibility
    out_polished = qa_paths.polished_path(session_id, run_id)
    polished_data = {
        "text": result.get("text", ""),
        "fallback": False,
        "words": result.get("words", 0),
        "passages_used": result.get("passages_used", 0),
    }
    with open(out_polished, "w", encoding="utf-8") as f:
        json.dump(polished_data, f, ensure_ascii=False, indent=2)
