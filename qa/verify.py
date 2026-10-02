"""
qa/verify.py
Q5: Verify — check paraphrased claims against evidence using Qwen 2.5 3B.
Same model and num_ctx as Q4 (no reload). Model is kept loaded (keep_alive="60s") for Q6.
"""
import json

import ollama

from qa.config import (
    QA_MODEL, NUM_CTX_ANSWER, NUM_PREDICT_VERIFY,
    VERIFY_PARAPHRASES, VERIFY_MAX,
)
from qa import paths as qa_paths


def run(session_id: str, run_id: str) -> dict:
    """
    Q5: Verify paraphrase claims.
    Reads answer.json and reranked.json, writes verified.json.
    Leaves model loaded for Q6; orchestrator handles final unload.
    """
    a_path = qa_paths.answer_path(session_id, run_id)
    r_path = qa_paths.reranked_path(session_id, run_id)

    with open(a_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)
    with open(r_path, "r", encoding="utf-8") as f:
        reranked = json.load(f)

    # Build sentence map from passages
    sid_map: dict[str, str] = {}
    for p in reranked.get("passages", []):
        for s in p.get("window", []):
            sid_map[s["sid"]] = s["text"]

    claims = answer_data.get("claims", [])
    dropped_claims: list[dict] = []

    if VERIFY_PARAPHRASES and claims:
        # Find paraphrase claims, prioritizing weak ones
        paraphrase_claims = [
            (i, c) for i, c in enumerate(claims)
            if c.get("kind") in ("paraphrase", "weak")
        ]
        # Sort: weak first
        paraphrase_claims.sort(key=lambda x: (0 if x[1].get("kind") == "weak" else 1))
        paraphrase_claims = paraphrase_claims[:VERIFY_MAX]

        indices_to_remove = set()
        for idx, claim in paraphrase_claims:
            # Build evidence text
            cited_sents = []
            for sid in claim.get("support", []):
                if sid in sid_map:
                    cited_sents.append(sid_map[sid])
            evidence_text = "\n".join(cited_sents)

            if not evidence_text.strip():
                continue

            prompt = (
                'Decide whether the CLAIM is fully supported by the EVIDENCE. '
                'Answer {"supported": true} or {"supported": false}.\n'
                'Say true only if every fact in the claim appears in the evidence.\n'
                f'CLAIM: {claim["text"]}\n'
                f'EVIDENCE:\n{evidence_text}'
            )

            try:
                response = ollama.chat(
                    model=QA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    format={"type": "object", "properties": {"supported": {"type": "boolean"}}, "required": ["supported"]},
                    options={
                        "temperature": 0,
                        "seed": 42,
                        "num_ctx": NUM_CTX_ANSWER,
                        "num_predict": NUM_PREDICT_VERIFY,
                    },
                    keep_alive="60s",
                )
                verdict = json.loads(response["message"]["content"])
                supported = verdict.get("supported", True)
            except Exception:
                supported = True  # On error, keep the claim

            if not supported:
                indices_to_remove.add(idx)
                dropped_claims.append(claim)
            else:
                claim["verified"] = True

        # Remove unsupported claims
        if indices_to_remove:
            claims = [c for i, c in enumerate(claims) if i not in indices_to_remove]

    # Quote claims need no verification — mark them
    for claim in claims:
        if claim.get("kind") == "quote":
            claim["verified"] = True

    # Extractive fallback if all claims removed
    if not claims and answer_data.get("status") == "answered":
        # Use top sentences from passages
        all_sents = []
        for p in reranked.get("passages", []):
            for s in p.get("window", []):
                all_sents.append(s)
        all_sents.sort(key=lambda s: s.get("score", 0), reverse=True)
        for s in all_sents[:2]:
            claims.append({
                "text": s["text"],
                "support": [s["sid"]],
                "kind": "extractive_fallback",
                "support_score": 1.0,
                "verified": True,
            })

    result = {
        "status": answer_data.get("status", "answered"),
        "partial": answer_data.get("partial", False),
        "claims": claims,
        "dropped_claims": dropped_claims,
    }

    # Carry forward not_found closest
    if answer_data.get("status") == "not_found":
        result["closest"] = answer_data.get("closest", [])

    out_path = qa_paths.verified_path(session_id, run_id)
    if not qa_paths.session_exists(session_id):
        raise FileNotFoundError(f"Session {session_id} no longer exists")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
