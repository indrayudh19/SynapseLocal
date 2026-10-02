# PASS 4B: Answer Consolidation (Q6) on top of the working Q&A pipeline

Extraction works well. This pass adds one stage, `qa/consolidate.py`, that turns the verified claims into one structured, readable answer. The screen then shows the consolidated answer; the extracted claims move to a collapsed section.

## 0. Operating Rules (read first, obey throughout)

1. **Build, don't deliberate.** Read only the files in section 2. No repo scans, no plan restatement, no summaries before coding.
2. **Additive.** One new stage, small edits to the files listed in section 2. Q1-Q5 logic, retrieval, reranking, storage layout and everything else stay as they are.
3. **Do not touch `representation/`.** Not one line, no imports from it.
4. **No unrequested work.** No new features, no tests, no evaluation harness, no new dependencies.
5. **One stage at a time, memory discipline unchanged (4 GB).** Q6 runs after Q5, in process, on the already loaded model, and the model is unloaded afterwards. No parallel calls, no subprocess.
6. **Verify statically only** (section 9). The user checks behavior.

---

## 1. Goal and Architecture

```
Q4 ANSWER -> Q5 VERIFY -> [Q6 CONSOLIDATE] -> answers.jsonl -> UI
                          Qwen 2.5 3B, same loaded model
```

Q6 reads **only the final claims** (`verified.json`, or `answer.json` when verification is off) plus the question and its `intent` from `understanding.json`. The LLM never sees the passages again. It can only reorganize what the claims already say: merge duplicates, order points, and write direct sentences. Code then checks every output sentence against the claims and their source sentences, and falls back to the claims if anything fails.

---

## 2. Files

Read once, then edit: `qa/config.py`, `qa/schema.py`, `qa/paths.py`, `qa/store.py`, `qa/verify.py`, `qa/orchestrator.py`, `app.py` (Local RAG Chat tab only).

New: `qa/consolidate.py`.

Protected: `representation/`, `qa/understand.py`, `qa/retrieve.py`, `qa/rerank.py`, `qa/answer.py` logic, `qa/worker.py`, `qa/bm25.py`, the Concept Clusters tab, the sidebar, and everything under `backend/`.

---

## 3. Constants and Schema

`qa/config.py` (add; bump `PIPELINE_VERSION = "qa-v1.1"`):

```python
NUM_PREDICT_CONSOLIDATE = 260
CONSOLIDATE_MAX_CLAIMS = 5
CONSOLIDATE_MAX_POINTS = 4
CONSOLIDATE_SUPPORT_MIN = 0.7      # stricter than SUPPORT_MIN: this stage only rewrites verified material
DEDUPE_JACCARD = 0.8
CONSOLIDATE_MAX_WORDS = 120
```

`qa/schema.py` (add):

```python
class Statement(BaseModel):
    text: str = Field(max_length=300)
    cites: list[str] = Field(min_length=1, max_length=4)    # claim ids "c1".."c5"

class Consolidated(BaseModel):
    lead: Statement
    points: list[Statement] = Field(max_length=4)
```

`qa/paths.py`: add `consolidated.json` to the run-directory files.

---

## 4. Q6: Consolidate (`qa/consolidate.py`)

`run(session_id, run_id) -> dict`

Skip entirely (return `None`, UI uses the claims) when the status is `not_found`.

### 4.1 Prepare claims (deterministic, no LLM)

1. Load the final claims. Keep: all `quote` claims, all `paraphrase` claims with `verified: true`, and unverified paraphrase or fallback claims with `support_score >= CONSOLIDATE_SUPPORT_MIN`. Record the rest as `excluded`. If that leaves nothing, use all claims (never produce an empty answer).
2. **Deduplicate:** if the Jaccard similarity of two claims' light-stemmed `content_tokens` is >= `DEDUPE_JACCARD`, merge the later into the earlier and **union their `support` ids** so no citation is lost.
3. Cap at `CONSOLIDATE_MAX_CLAIMS`, keeping order. Assign ids `c1..cn`.
4. **Single-claim shortcut:** if one claim remains, set `lead` to that claim's text verbatim, `points = []`, skip the LLM, and go to section 4.4.

### 4.2 LLM call

Same model (`QA_MODEL`), same `num_ctx` as Q4/Q5 (`NUM_CTX_ANSWER`, so no reload), `num_predict NUM_PREDICT_CONSOLIDATE`, `temperature 0`, `seed 42`, `format=Consolidated.model_json_schema()`, `keep_alive "0"`. Retry once on a parse failure, then fall back (4.5).

System message (stable prefix; the question and statements go last):

```
You rewrite verified statements into one clear answer. Output JSON only.
You get a QUESTION and numbered STATEMENTS (c1, c2, ...). Use ONLY facts stated in the statements.
Rules:
1. lead: one sentence that directly answers the question, using the wording of the statements.
2. points: up to 4 further statements that add distinct information. Never repeat the lead or each other.
   - list questions: one item per point
   - procedure questions: one step per point, in order
   - comparison questions: one point per side, then one point on the difference only if the statements state it
3. Each lead and point cites the statement ids it relies on, for example ["c1","c3"].
4. Do not add facts, examples, numbers or explanations that are not in the statements.
5. Merge duplicates. Drop statements that do not help answer the question.
6. Plain, direct sentences. No hedging, no commentary.
```

One few-shot pair (non-cloud domain):

```
USER:      QUESTION: What do mitochondria do?
           STATEMENTS:
           c1: Mitochondria produce most of the cell's ATP through cellular respiration.
           c2: ATP is produced by mitochondria.
           c3: They have their own small genome.
ASSISTANT: {"lead":{"text":"Mitochondria produce most of the cell's ATP through cellular respiration.","cites":["c1","c2"]},
            "points":[{"text":"Mitochondria also have their own small genome.","cites":["c3"]}]}
USER:      QUESTION: <question>
           STATEMENTS:
           c1: <claim text>
           ...
```

### 4.3 Verification of the output (code, mandatory)

For the lead and each point:

1. Drop cited ids that do not exist; drop the statement if no valid id remains.
2. **Support check.** Let `ref` = light-stemmed `content_tokens` of the cited claims' texts **plus the source sentences those claims cite** (resolve the sentence ids from `reranked.json`; this is a code-only lookup, the LLM does not see them). Require `|tokens(statement) intersect ref| / |tokens(statement)| >= CONSOLIDATE_SUPPORT_MIN`. Every number in the statement (regex for digits, decimals, percentages) must also appear in the cited text. Otherwise drop the statement.
3. Drop a point whose token Jaccard with the lead or an earlier point is >= `DEDUPE_JACCARD`.
4. Keep at most `CONSOLIDATE_MAX_POINTS` points and trim trailing points until the total is under `CONSOLIDATE_MAX_WORDS`.
5. If the lead was dropped but a point survives, promote the first surviving point to lead. If nothing survives, use the fallback (4.5).

### 4.4 Citation markers and style (code)

- A statement's markers are the sorted unique passage numbers (`pid`) derived from the `support` ids of the claims it cites (for example `c1` supports `2.3`, so passage `[2]`). Render as `[2][3]`.
- **Style is chosen by code from `intent`**, not by the model: `list` and `comparison` render points as bullets; `procedure` renders numbered steps; every other intent renders a short paragraph (lead followed by points as flowing sentences), switching to bullets when there are 3 or more points.

### 4.5 Fallback (the pipeline must never fail because of Q6)

On an LLM error, timeout, repeated parse failure, or an empty verified result: build the output from the deduplicated claims (first claim as lead, the rest as points, original wording, markers from their supports) and set `fallback: true`.

### 4.6 Output

Write `consolidated.json`:

```json
{"lead":{"text":"...","cites":["c1","c2"],"markers":[2,3]},
 "points":[{"text":"...","cites":["c3"],"markers":[4]}],
 "style":"paragraph|bullets|steps","fallback":false,
 "claims_used":["c1","c2","c3"],"excluded":[],"dropped":[],"consolidate_s":0.0}
```

---

## 5. Orchestrator and Q5 Changes

- `orchestrator.py`: after Q5 (or after Q4 when verification is off), update the status label to "Consolidating the answer", run Q6, and then unload the model. Move the model **unload into the orchestrator's final `finally`** (and drop it from Q5's `finally`) so the model stays loaded from Q4 through Q6 and is always unloaded exactly once, including on errors.
- `verify.py`: use `keep_alive "60s"` and remove its own unload; no other changes.
- Q6 never raises into the orchestrator; it returns the fallback instead. The run directory cleanup is unchanged.

---

## 6. Storage (`store.py`)

Add a `consolidated` field to each `answers.jsonl` record (the object from section 4.6) and `timings.consolidate_s`. Keep `claims` as they are (pre-consolidation). Do not rewrite or migrate existing records: records without `consolidated` must still load and display.

---

## 7. UI (Local RAG Chat tab only)

The screen shows the consolidated answer. Everything else is secondary and collapsed.

1. **Main area:** lead, then points per `style` (paragraph, bullets, numbered), each statement followed by its citation markers (`[2][3]`). Escape Markdown special characters in all document-derived text; no `unsafe_allow_html`.
2. Under it, one caption: "Consolidated from N extracted statements. Open Sources for the original text." If `fallback` is true, caption instead: "Consolidation unavailable; showing extracted statements."
3. **Collapsed expanders**, in this order:
   - **Sources:** unchanged.
   - **Extracted statements (before consolidation):** the old claim list with its `Quoted from source` / `Synthesized (verified)` / `Extractive` badges and the dropped or excluded claims.
   - **How this was answered:** unchanged, plus the consolidation time.
4. **History:** records that have `consolidated` render as above; older records without it render with the previous claim view. Not found, error and `partial` handling are unchanged.
5. Remove the per-claim badges from the main answer area. No new widgets.

---

## 8. Known Limits (do not add machinery for these)

- Q6 can only reorganize what the claims contain. If extraction missed a fact, it will not appear.
- The lexical support check cannot catch every semantic error, such as a flipped negation. That is why Sources stays one click away.
- It adds one LLM call per question (short prompt, at most 260 output tokens).

---

## 9. Done Criteria (static checks plus one smoke run)

- [ ] `python -m py_compile` on `qa/`; `git diff` shows no changes under `representation/`, `backend/`, or the protected `qa/` files from section 2.
- [ ] Grep: `consolidate.py` imports no `torch`, `transformers` or `sentence_transformers`.
- [ ] Smoke run with one question: `answers.jsonl` gets a record with `consolidated` and `timings.consolidate_s`; the main screen shows the lead, points and markers only; `ollama ps` shows no model resident afterward.
- [ ] Grep confirms the fallback path is called on exceptions and on an empty result.
- [ ] An older record without `consolidated` still renders.
- [ ] Session delete still removes everything under `qa/`.

If a check fails, fix that specific issue and stop.

---

## 10. Final Reminder to the Agent

Add one stage that rewrites verified claims into a lead plus supporting points, chosen style by intent, every sentence checked against the claims and their source sentences, and a fallback to the claims. Keep the model loaded from Q4 through Q6 and unload once at the end. Touch nothing else, add no dependencies, and spend no tokens on extra analysis.
