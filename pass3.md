# PASS 3: LLM Concept Extraction + Concept Clustering (replaces old clustering)

## 0. Operating Rules (read first, obey throughout)

1. **Build, don't deliberate.** Read only the files listed in section 2. Do not scan the repo, restate the plan, or summarize before coding. Write code.
2. **No unrequested work.** No extra refactors, no new features, no docs beyond docstrings, no test suites. One smoke run per stage (section 10), nothing more.
3. **Do not touch unrelated code.** Protected paths are listed in section 2. If a change is not required by this spec, do not make it.
4. **One stage at a time, at runtime and while building.** Finish and save one stage before starting the next. Never draft all three at once.
5. **Hardware budget is a hard constraint.** Sequential execution only, no threads, no multiprocessing, no parallel LLM calls, small context window, unload the model when done (section 8).
6. **Wire to the UI after the code exists** (section 9). The user verifies behavior. Do not run extended validation or benchmarks.
7. **Remove, don't keep.** Old clustering code is deleted completely, not commented out, not wrapped in a flag.

**Goal:** a concept map where a local LLM (Qwen 2.5 3B via Ollama) extracts concepts and typed relations from the document, a lightweight algorithm clusters them, and the result is drawn as hubs with satellites:

```
        [SaaS] [PaaS]                 [GPU]
            \   |                        |
          ( Infrastructure ) ======== ( Hardware ) -- [TPU]
            /   |                        |
        [IaaS] ...                    [accelerators]
```

Big circle = hub concept. Small circles = its members. Hub-to-hub links show how topics relate. Distance reflects link strength.

---

## 1. Architecture

```
chunks.jsonl (from pass1, unchanged)
   |
   v
[Stage 1] EXTRACT   (Ollama only)  -> concepts_raw.jsonl        (end of task)
   |
   v
[Stage 2] CONSOLIDATE (no model)   -> concept_graph.json        (end of task)
   |
   v
[Stage 3] CLUSTER + RENDER (no model) -> concept_clusters.json, concept_map.png   (end of task)
```

- Three independent services, one `run(session_id, ...) -> dict` each. Each reads only the previous stage's files from disk, writes its own outputs, frees memory, returns stats.
- Stage 1 is the only stage that uses an LLM. Stages 2 and 3 load no model and no embedding library.
- The UI only loads saved files. It never computes extraction, graph or clustering at render time.

---

## 2. Scope

**Read once, then edit/replace:**
- The current `representation/cluster.py` (to delete its contents) and one grep for what imports from it or from its plotting helpers.
- `representation/paths.py` (append new path helpers only).
- `app.py` (Cluster tab only).
- `requirements.txt`.

**Protected, do not modify:**
- `representation/embed_store.py`, `representation/graph.py` and its `graph.png` / `graph_meta.json` outputs.
- All pass1 ingestion and chunking code.
- All existing LLM inference code: `local_llm.py`, retriever, reranker, chat tab. **Do not import from or edit `local_llm.py`.** Stage 1 makes its own direct Ollama calls in a new file.
- Session management and chat delete logic (new artifacts live under the session folder, so existing delete already covers them).

**Delete (old clustering):**
- Everything in `cluster.py`: PCA + MiniBatchKMeans, silhouette / Davies-Bouldin selection, 2D PCA projection, scatter rendering.
- Outputs `clusters.json` and `cluster_map.png`, and any scatter helper in `plotting.py` that nothing else uses (check with one grep; keep helpers `graph.py` still imports).
- UI metric cards for Optimal k, Silhouette, Davies-Bouldin, PCA variance.
- Keep `scikit-learn` in requirements only if `graph.py` still imports it (TF-IDF labels). Otherwise remove it.

**New dependencies:** `ollama`, `pydantic` (likely already present), `networkx`, `matplotlib`. Nothing else. No `rapidfuzz`, no `sentence-transformers` in these stages.

---

## 3. Layout and Artifacts

New / rewritten files:

```
representation/
  concept_schema.py     # constants + pydantic models (new)
  extract_concepts.py   # Stage 1 (new)
  build_concept_graph.py# Stage 2 (new)
  cluster.py            # Stage 3 (REWRITTEN completely)
  concept_plot.py       # matplotlib hub-and-spoke renderer (new)
  paths.py              # append helpers only
```

Per-session artifacts under `data/sessions/<session_id>/representation/`:

```
concepts_raw.jsonl     # Stage 1 checkpoint, one line per processed chunk
extract_log.json       # Stage 1 stats
concept_graph.json     # Stage 2
concept_clusters.json  # Stage 3
concept_map.png        # Stage 3
```

Add one path function per artifact to `paths.py`, each taking `session_id`. No global paths.

`concept_schema.py` constants (single source of truth):

```python
EXTRACT_MODEL   = "qwen2.5:3b"
PROMPT_VERSION  = "v1"
MAX_CHARS       = 1200      # chunk text sent to the LLM
MIN_CHARS       = 200       # skip shorter chunks
MAX_CONCEPTS    = 8
MAX_RELATIONS   = 8
GLOBAL_FREQ     = 0.40      # concept in >40% of processed chunks = global topic
MIN_CLUSTER_SIZE= 3
MAX_CLUSTERS    = 12        # drawn
MAX_SPOKES      = 8         # drawn per hub
HIERARCHY_RELS  = ("type_of", "part_of", "example_of")
REL_WEIGHT      = {"type_of":3.0, "part_of":3.0, "example_of":3.0, "uses":1.5, "related_to":1.0}
```

---

## 4. Stage 1: Extract Concepts (`extract_concepts.py`)

**Input:** `chunks.jsonl`. **Uses:** Ollama only. **Output:** `concepts_raw.jsonl`, `extract_log.json`.

`run(session_id, max_chunks=None, on_progress=None) -> dict`

### 4.1 Chunk selection (local to this stage, do not modify the chunker)

- Use **parent chunks** (unique `parent_id`; concatenate or take the parent text), not child chunks. Fewer calls, more context per call.
- Skip a chunk if any of: shorter than `MIN_CHARS`; digit ratio > 0.35; looks like a reference list (more than 30% of lines match `et al\.|arXiv|pp\.|\b(19|20)\d{2}\b.*\.$|doi`); near-duplicate (hash of normalized text already seen).
- Truncate text to `MAX_CHARS` at a sentence boundary.
- `max_chunks` (UI "sample size") takes the first N eligible chunks, for fast trial runs.

### 4.2 Constrained output schema

Use Ollama structured outputs: pass `format=Extraction.model_json_schema()` so decoding is constrained to valid JSON. This is what makes a 3B model reliable enough.

```python
class Concept(BaseModel):
    name: str                      # 1-4 words

class Relation(BaseModel):
    source: str
    relation: Literal["type_of","part_of","example_of","uses","related_to"]
    target: str

class Extraction(BaseModel):
    concepts:  list[Concept]  = Field(max_length=8)
    relations: list[Relation] = Field(max_length=8)
```

### 4.3 Prompt (use exactly; do not iterate on it)

System message:

```
You extract a concept map from study notes. Output JSON only.
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
4. Use only information stated in the TEXT.
```

One few-shot pair (deliberately **not** cloud computing, to avoid the model copying it into cloud documents):

```
USER:      TEXT: Relational databases store data in tables. PostgreSQL and MySQL are popular examples. Indexes speed up queries on a table.
ASSISTANT: {"concepts":[{"name":"relational database"},{"name":"table"},{"name":"PostgreSQL"},{"name":"MySQL"},{"name":"index"},{"name":"query"}],
            "relations":[{"source":"PostgreSQL","relation":"example_of","target":"relational database"},
                         {"source":"MySQL","relation":"example_of","target":"relational database"},
                         {"source":"table","relation":"part_of","target":"relational database"},
                         {"source":"index","relation":"uses","target":"table"}]}
USER:      TEXT: <chunk text>
```

### 4.4 Call settings

```python
ollama.chat(model=EXTRACT_MODEL, messages=msgs, format=schema,
            options={"temperature": 0, "seed": 42, "num_ctx": 2048, "num_predict": 512},
            keep_alive="10m")
```

- Strictly sequential, one request at a time.
- Parse with pydantic. On failure retry **once**, then record `status:"failed"` and continue. Abort the stage after 5 consecutive failures with a clear message (e.g. Ollama not running, or run `ollama pull qwen2.5:3b`).
- Check the model exists once at start via `ollama.list()`.

### 4.5 Grounding validation (mandatory, this is the quality gate)

A 3B model hallucinates. Validate every item against the chunk text before saving:

- **Concept kept** only if at least 60% of its non-stopword tokens appear in the chunk text (case-insensitive), or it is an acronym/expansion pair found in the text. This also removes any leakage from the few-shot example.
- Drop concepts matching a small generic stoplist (`paper, method, approach, model, results, figure, table, section, data, system, work`).
- **Relation kept** only if both endpoints are kept concepts, they are not identical, and `relation` is in the enum.
- Record counts of dropped concepts and relations in the log (grounding yield is the first quality metric).

### 4.6 Checkpointing and resume

- Append one JSON line per processed chunk immediately after validation: `{chunk_id, parent_id, source_file, source_type, status, concepts:[...], relations:[...]}`.
- On rerun, skip chunk ids already present for the same `(EXTRACT_MODEL, PROMPT_VERSION)`. If either changed, start fresh.
- `on_progress(done, total)` callback for the UI progress bar.

### 4.7 End of stage

- Write `extract_log.json`: `{chunks_total, processed, skipped_boilerplate, failed, avg_seconds_per_chunk, concepts_kept, concepts_dropped, relations_kept, relations_dropped}`.
- **Unload the model:** `ollama.generate(model=EXTRACT_MODEL, prompt="", keep_alive=0)`.
- Release memory (`gc.collect()`) and return stats. Do not start Stage 2.

---

## 5. Stage 2: Consolidate (`build_concept_graph.py`)

**Input:** `concepts_raw.jsonl`, `chunks.jsonl` (only for acronym patterns and source files). **No model, no LLM.** **Output:** `concept_graph.json`.

`run(session_id) -> dict`

1. **Normalize keys:** lowercase, strip punctuation except hyphens, collapse whitespace, drop leading articles, light singularization (trailing `s` removed if length > 3 and not `ss`). Display name = most frequent surface form.
2. **Acronym aliases:** regex over chunk text for `Long Form (ABC)` patterns to map acronym keys to their expansion keys (so "IaaS" and "Infrastructure as a Service" merge, as do "EC2" and "Elastic Compute Cloud" when the text defines them).
3. **Fuzzy merge:** within blocks (same first character and similar length), merge keys with `difflib.SequenceMatcher` ratio >= 0.88 via union-find. No subset merging ("cloud" must not merge into "cloud computing"). No embeddings here; semantic synonym merging is out of scope for this pass.
4. **Aggregate concepts:** `{key, name, support (distinct chunks), sources (set of source_file), chunk_ids}`.
5. **Aggregate relations:** per `(source, relation, target)`: `support` = distinct chunks. Edge weight = `REL_WEIGHT[rel] * (1 + ln(support))`.
6. **Hierarchy sanity:** for `HIERARCHY_RELS`, if both directions exist between two concepts keep the one with higher support (drop both on tie). Remove cycles in the hierarchy subgraph by deleting the weakest edge in each cycle (`nx.find_cycle` loop).
7. **Global topics:** concepts with `support / processed_chunks > GLOBAL_FREQ` are flagged `global: true`. They connect everything and destroy community structure, so Stage 3 excludes them (they are listed separately in output).
8. **Keep** only concepts with at least one relation for the clustering graph. Concepts with no relations go to `unlinked` in the JSON (not drawn).
9. **Persist** `concept_graph.json`: `{concepts, relations, global_topics, unlinked, stats:{n_concepts, n_relations, n_hierarchy_edges, n_global, n_unlinked}}`.

End of stage: write, return stats. Do not start Stage 3.

---

## 6. Stage 3: Cluster + Render (`cluster.py` rewritten, `concept_plot.py` new)

**Input:** `concept_graph.json`. **No model.** **Output:** `concept_clusters.json`, `concept_map.png`.

`run(session_id) -> dict`

### 6.1 Quality gate

If linked, non-global concepts `< 8` or relations `< 6`, write `concept_clusters.json` with `{"status": "insufficient_structure", "n_concepts":..., "n_relations":...}`, skip rendering, and return. The UI shows a message suggesting more chunks or a richer document. Never draw a meaningless plot.

### 6.2 Community detection

1. Build an undirected weighted `networkx.Graph` from non-global concepts and their relation weights.
2. Run `networkx.algorithms.community.louvain_communities(G, weight="weight", resolution=r, seed=42)` for `r in [0.8, 1.0, 1.3, 1.7]`. Choose the resolution with the highest modularity whose **median community size is between 3 and 12**. Fall back to `r=1.0`.
3. **Merge small communities:** any community smaller than `MIN_CLUSTER_SIZE` merges into the neighboring community with the largest total inter-edge weight. If it has no neighbor, mark its members as orphans (listed in JSON, not drawn).

### 6.3 Hub selection

Per community, hub = argmax of the lexicographic tuple:
`(number of in-community concepts that point to it via HIERARCHY_RELS, weighted degree, PageRank)`; deterministic tie-break by name. Members are the spokes. Per spoke, record `link_to_hub`: `"direct:<relation>"` if a hierarchy edge to the hub exists, otherwise `"indirect"`.

### 6.4 Cluster-level graph

Quotient graph: one node per cluster, edge weight = sum of inter-cluster concept edge weights. For each link store the strongest underlying edge `{source, relation, target}` for labeling. Keep, per cluster, only its top 2 strongest links to avoid a hairball.

### 6.5 Metrics (replace silhouette etc.)

`{n_concepts_clustered, n_clusters, modularity, resolution, hub_coverage (direct spokes / all spokes), cross_source_concepts (concepts supported by >=2 distinct source files), n_orphans, n_global}`.

### 6.6 Persist `concept_clusters.json`

```json
{
  "status": "ok",
  "clusters": [{"id": 0, "hub": "infrastructure", "size": 7,
                "members": [{"name": "SaaS", "weight": 4.1, "link_to_hub": "direct:type_of", "sources": ["slides.pptx","notes.pdf"]}],
                "source_mix": {"slides.pptx": 0.6, "notes.pdf": 0.4}}],
  "links":    [{"a": 0, "b": 1, "weight": 5.2, "top_edge": {"source": "IaaS", "relation": "uses", "target": "hardware"}}],
  "global_topics": ["..."], "orphans": ["..."],
  "metrics": {}
}
```

### 6.7 Render (`concept_plot.py`, static matplotlib)

`matplotlib.use("Agg")`, `figsize=(11, 8)`, `dpi=110`, match the app's existing dark theme (dark background, light text), `plt.close(fig)` after saving.

- **Draw limits:** top `MAX_CLUSTERS` clusters by size x total weight. Per hub, top `MAX_SPOKES` members by weight; annotate `+k more` under the hub if truncated (the UI table lists everything).
- **Hubs:** large `matplotlib.patches.Circle`, color per cluster from `tab20`, radius `0.55 + 0.12*sqrt(n_members)`, bold label wrapped inside (about 16 chars/line).
- **Spokes:** small circles evenly spaced on a ring around the hub, radius `0.18-0.30` scaled by normalized member weight. Ring radius grows with spoke count so spokes never overlap. Label outside the circle, radial, 8pt, wrapped. Thin line hub to spoke.
- **Cross-document concepts:** spokes supported by >= 2 source files get a gold outline. This shows the PPT + notes overlap directly.
- **Cluster positions:** `networkx.spring_layout(quotient, weight="weight", seed=42)`. Stronger link = closer, weak/unlinked = far. Then run a short relaxation loop (<= 200 numpy iterations) pushing apart any two clusters whose outer discs (ring radius + spoke radius) overlap, then rescale to the canvas.
- **Hub-to-hub links:** line from hub edge to hub edge, width `0.8 + 0.6*ln(1+weight)`, small midpoint label showing the top edge relation (e.g. `uses`).
- Title: `Concept Map (clusters: k | concepts: n | modularity: Q)`.

End of stage: write files, return the metrics dict.

---

## 7. Known Failure Modes (design already accounts for these; do not add more machinery)

- **Weak extractor (3B):** mitigated by schema-constrained decoding, closed relation set, one chunk per call, temperature 0, grounding validation, direction/cycle sanity checks.
- **Few-shot leakage:** mitigated by a non-domain example plus grounding check.
- **Generic mega-hubs** (document title concept in most chunks): mitigated by the `GLOBAL_FREQ` exclusion.
- **Boilerplate/reference chunks:** filtered locally in Stage 1; pass1 chunker stays untouched.
- **Sparse relations:** quality gate in 6.1 instead of a misleading plot.

---

## 8. Execution and Hardware Rules (non-negotiable)

- One stage at a time per session. Enforce with the same lock file as pass2 (`data/sessions/<session_id>/.running`, always removed in `finally`). Block other stages for that session while one runs.
- No threads, no multiprocessing, no async fan-out, no concurrent Ollama requests.
- `num_ctx=2048`, `num_predict=512`, chunk text <= 1200 chars. Do not raise these.
- Stage 1 must not import `sentence_transformers`, `faiss` or `sklearn`. Stages 2 and 3 import no model library. Import `ollama` only inside Stage 1's `run()`.
- Unload the Ollama model at the end of Stage 1 (`keep_alive=0`).
- Cache by content: each stage writes a small hash of its inputs + parameters; if unchanged, return immediately. Stage 1 additionally resumes from `concepts_raw.jsonl` after an interruption.
- Fixed seeds everywhere.

---

## 9. UI Wiring (after the code is written)

Replace the content of the Cluster tab only (rename its label to "Concept Clusters"). Do not touch the chat tab or the Knowledge Graph tab.

1. Three steps, each with its own button and a status derived from file existence:
   - **1. Extract concepts (Qwen)**: number input "Sample size (chunks)" defaulting to all (min 5), `st.progress` driven by `on_progress`, estimated time shown as `chunks x avg_seconds` once known. Re-running resumes from the checkpoint. A small "Reset extraction" button deletes `concepts_raw.jsonl`.
   - **2. Consolidate**: enabled when `concepts_raw.jsonl` exists.
   - **3. Build concept map**: enabled when `concept_graph.json` exists.
2. Each button calls only its own stage, wrapped in `st.spinner`. No auto-chaining and no "run all" button.
3. Display after Stage 3: `st.image(concept_map.png)`, metric cards (concepts, relations, clusters, modularity, cross-source concepts), a cluster table (hub, size, members, source mix), and an expander with global topics, orphans, and the extraction log.
4. If `status == "insufficient_structure"`, show the message instead of an image.
5. Never recompute on rerun; load files. Switching chats loads that chat's own artifacts; a chat with none shows "not built yet".
6. Update `requirements.txt` per section 2.

---

## 10. Done Criteria (minimal verification)

Run only what is needed to confirm the wiring:

- [ ] Stage 1 on a sample (10-15 chunks) writes `concepts_raw.jsonl` and `extract_log.json`, then Ollama unloads the model.
- [ ] Stage 2 runs from those files alone and writes `concept_graph.json`.
- [ ] Stage 3 runs from `concept_graph.json` alone and writes `concept_clusters.json` and `concept_map.png` (or the insufficient-structure status).
- [ ] The Cluster tab shows the three buttons and the image; deleting a chat removes its entire session folder.
- [ ] No leftover imports of `KMeans`, `silhouette_score`, `davies_bouldin_score`, PCA scatter code; no changes to protected files.

If a check fails, fix that specific issue and stop. Do not run broader audits.

---

## 11. Final Reminder to the Agent

Three decoupled stages, one at a time: extract (LLM), consolidate (no model), cluster and render (no model). Closed relation set, grounded extraction, hub-and-spoke output. Delete the old clustering completely, touch nothing unrelated, spend no tokens on extra analysis.
