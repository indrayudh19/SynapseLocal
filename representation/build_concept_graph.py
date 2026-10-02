"""
representation/build_concept_graph.py
Stage 2: Consolidate concepts and typed relations into a clean graph.
Normalizes keys, resolves acronym aliases, fuzzy-merges variations,
performs suffix merge, embedding-assisted duplicate detection,
optional Qwen merge judge, enforces hierarchy DAG consistency,
filters global hub topics, and saves concept vectors for Stage 3.
"""
from collections import Counter, defaultdict
import difflib
import gc
import hashlib
import json
import math
import os
import re
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from representation import paths as rep_paths
from representation.concept_schema import (
    GLOBAL_FREQ,
    HIERARCHY_RELS,
    REL_WEIGHT,
    SUFFIXES,
    MERGE_AUTO_SIM,
    MERGE_JUDGE_BAND,
    MERGE_JUDGE_MAX,
    LLM_MERGE_JUDGE,
    EXTRACT_MODEL,
)

# The same embedding model name that embed_store.py uses
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_GENERIC_MODIFIERS = {"business", "core", "general"}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "that", "this", "it", "its", "into", "their", "such",
}


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if self.rank[root_a] < self.rank[root_b]:
                root_a, root_b = root_b, root_a
            self.parent[root_b] = root_a
            if self.rank[root_a] == self.rank[root_b]:
                self.rank[root_a] += 1


def normalize_key(text: str) -> str:
    """Normalize a concept string to a canonical dictionary key."""
    s = text.strip().lower()
    # Strip punctuation except hyphens
    s = re.sub(r'[^\w\s\-]', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Drop leading articles
    for article in ("the ", "a ", "an "):
        if s.startswith(article):
            s = s[len(article):].strip()
            break
    # Light singularization (trailing s removed if length > 3 and not ss)
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s.strip()


def extract_acronym_aliases(chunks_file: str) -> Dict[str, str]:
    """Scan chunks for 'Long Form (ABC)' definitions and map acronym -> expansion."""
    aliases: Dict[str, str] = {}
    if not os.path.exists(chunks_file):
        return aliases

    pattern = re.compile(r'\b([A-Z][A-Za-z0-9\s\-]{2,40}?)\s*\(([A-Z0-9]{2,8})\)')

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            text = chunk.get("text", "")
            matches = pattern.findall(text)
            for long_form, acronym in matches:
                words = [w for w in re.split(r'[\s\-]+', long_form) if w]
                initials = "".join(w[0] for w in words).upper()
                if initials == acronym.upper() or acronym.upper() in initials:
                    norm_long = normalize_key(long_form)
                    norm_acr = normalize_key(acronym)
                    if norm_long and norm_acr and norm_long != norm_acr:
                        aliases[norm_acr] = norm_long

    return aliases


def run(session_id: str) -> dict:
    """
    Execute Stage 2: Consolidate concepts and relations from concepts_raw.jsonl into concept_graph.json.
    Now includes suffix merge, embedding-assisted duplicate detection, optional Qwen merge judge,
    and saves concept_vectors.npy for Stage 3.
    """
    raw_path = rep_paths.concepts_raw_path(session_id)
    if not os.path.exists(raw_path):
        return {"status": "error", "message": f"concepts_raw.jsonl not found for session {session_id}"}

    chunks_file = rep_paths.chunks_jsonl_path(session_id)
    out_path = rep_paths.concept_graph_path(session_id)
    hash_path = rep_paths.concept_graph_hash_path(session_id)
    vectors_path = rep_paths.concept_vectors_path(session_id)

    # Check cache
    mtime = str(os.path.getmtime(raw_path))
    cur_hash = hashlib.md5(f"stage2v2:{mtime}".encode("utf-8")).hexdigest()
    if os.path.exists(hash_path) and os.path.exists(out_path):
        try:
            with open(hash_path, "r", encoding="utf-8") as f:
                if f.read().strip() == cur_hash:
                    with open(out_path, "r", encoding="utf-8") as out_f:
                        data = json.load(out_f)
                    return {"status": "ok", "cached": True, **data.get("stats", {})}
        except Exception:
            pass

    with rep_paths.session_lock(session_id):
        # 1. Read raw concepts and relations per chunk
        records = []
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        processed_chunks_count = len(records)
        if processed_chunks_count == 0:
            return {"status": "error", "message": "No chunks found in concepts_raw.jsonl"}

        # 2. Extract acronym aliases from chunks
        acronym_aliases = extract_acronym_aliases(chunks_file)

        # 3. Collect surface forms, kinds, and raw keys
        raw_keys: Set[str] = set()
        surface_forms: Dict[str, Counter] = defaultdict(Counter)
        kind_votes: Dict[str, Counter] = defaultdict(Counter)
        # Per-concept: main_topic_count, first_order, sources, chunk_ids, best_sentence
        concept_meta: Dict[str, dict] = defaultdict(lambda: {
            "main_topic_count": 0,
            "first_order": 999999,
            "sources": set(),
            "chunk_ids": set(),
            "best_sentence": "",
            "best_sentence_order": 999999,
        })

        for rec in records:
            for c in rec.get("concepts", []):
                raw_name = c.get("name", "").strip()
                if not raw_name:
                    continue
                k = normalize_key(raw_name)
                if k:
                    raw_keys.add(k)
                    surface_forms[k][raw_name] += 1
                    c_kind = c.get("kind", "category")
                    kind_votes[k][c_kind] += 1

        # Track main_topic_count and other metadata per concept key
        for rec in records:
            main_topic = rec.get("main_topic")
            order = rec.get("order", 999999)
            source_file = rec.get("source_file", "")
            cid = rec.get("chunk_id", "")

            for c in rec.get("concepts", []):
                raw_name = c.get("name", "").strip()
                if not raw_name:
                    continue
                k = normalize_key(raw_name)
                if not k or k not in raw_keys:
                    continue
                meta = concept_meta[k]
                meta["chunk_ids"].add(cid)
                if source_file:
                    meta["sources"].add(source_file)
                if order < meta["first_order"]:
                    meta["first_order"] = order
                if main_topic and normalize_key(main_topic) == k:
                    meta["main_topic_count"] += 1

            # Store best evidence sentences from relations
            for r in rec.get("relations", []):
                ev = r.get("evidence", "")
                if ev:
                    s_key = normalize_key(r.get("source", ""))
                    t_key = normalize_key(r.get("target", ""))
                    for rk in [s_key, t_key]:
                        if rk in raw_keys:
                            m = concept_meta[rk]
                            if order < m["best_sentence_order"] or not m["best_sentence"]:
                                m["best_sentence"] = ev
                                m["best_sentence_order"] = order

        # 4. Union-Find clustering for aliases & fuzzy matches
        uf = UnionFind()
        merge_report: List[dict] = []
        for k in raw_keys:
            uf.find(k)

        # Apply acronym aliases
        for acr, exp in acronym_aliases.items():
            if acr in raw_keys and exp in raw_keys:
                uf.union(exp, acr)

        # Fuzzy merge within blocks (same first char & similar length)
        by_first_char = defaultdict(list)
        for k in raw_keys:
            by_first_char[k[0]].append(k)

        n_merged_lexical = 0
        for char, block in by_first_char.items():
            block_len = len(block)
            for i in range(block_len):
                k1 = block[i]
                for j in range(i + 1, block_len):
                    k2 = block[j]
                    len_diff = abs(len(k1) - len(k2))
                    if len_diff > 3:
                        continue
                    
                    # Prevent subset merging e.g. "cloud" vs "cloud computing"
                    words1 = set(k1.split())
                    words2 = set(k2.split())
                    if words1 != words2 and (words1.issubset(words2) or words2.issubset(words1)):
                        continue

                    ratio = difflib.SequenceMatcher(None, k1, k2).ratio()
                    if ratio >= 0.88:
                        uf.union(k1, k2)
                        merge_report.append({"a": k1, "b": k2, "cosine": None, "decision": "lexical", "by": "difflib"})
                        n_merged_lexical += 1

        # 5. Suffix merge (lexical, no model)
        for k in list(raw_keys):
            words = k.split()
            if len(words) >= 2:
                last_word = words[-1]
                if last_word in SUFFIXES:
                    base = " ".join(words[:-1])
                    if base in raw_keys:
                        uf.union(base, k)
                        merge_report.append({"a": k, "b": base, "cosine": None, "decision": "lexical", "by": "suffix"})
                        n_merged_lexical += 1

                # Generic modifier merge: "business domain" -> "domain"
                first_word = words[0]
                if first_word in _GENERIC_MODIFIERS and len(words) >= 2:
                    rest = " ".join(words[1:])
                    if rest in raw_keys:
                        uf.union(rest, k)
                        merge_report.append({"a": k, "b": rest, "cosine": None, "decision": "lexical", "by": "modifier"})
                        n_merged_lexical += 1

        # 6. Embedding-assisted duplicate detection
        n_merged_embedding = 0
        n_judged = 0
        concept_keys_list = sorted(raw_keys)  # deterministic order
        concept_names_list = []
        for k in concept_keys_list:
            # Use most common surface form for embedding
            sf_counter = surface_forms.get(k, Counter())
            best_name = sf_counter.most_common(1)[0][0] if sf_counter else k
            concept_names_list.append(normalize_key(best_name))

        embeddings = None
        if len(concept_keys_list) >= 2:
            try:
                from sentence_transformers import SentenceTransformer
                embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
                embeddings = embed_model.encode(
                    concept_names_list,
                    normalize_embeddings=True,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)
                del embed_model
                gc.collect()
            except Exception:
                embeddings = None

        judge_pairs: List[Tuple[int, int, float]] = []  # (i, j, cosine)

        if embeddings is not None and len(embeddings) >= 2:
            # Compute pairwise cosine (embeddings are L2-normalized)
            sim_matrix = embeddings @ embeddings.T

            for i in range(len(concept_keys_list)):
                for j in range(i + 1, len(concept_keys_list)):
                    cos_val = float(sim_matrix[i, j])
                    if cos_val < MERGE_JUDGE_BAND[0]:
                        continue

                    k1 = concept_keys_list[i]
                    k2 = concept_keys_list[j]

                    # Already merged?
                    if uf.find(k1) == uf.find(k2):
                        continue

                    # Lexical gate: share >= 1 non-stopword token, or acronym alias link, or cosine >= 0.93
                    tokens1 = set(k1.split()) - _STOPWORDS
                    tokens2 = set(k2.split()) - _STOPWORDS
                    shared_tokens = tokens1 & tokens2
                    acronym_linked = (k1 in acronym_aliases and acronym_aliases[k1] == k2) or \
                                     (k2 in acronym_aliases and acronym_aliases[k2] == k1)
                    gate_passed = bool(shared_tokens) or acronym_linked or cos_val >= 0.93

                    if not gate_passed:
                        continue

                    if cos_val >= MERGE_AUTO_SIM:
                        uf.union(k1, k2)
                        merge_report.append({"a": k1, "b": k2, "cosine": round(cos_val, 4), "decision": "auto", "by": "embedding"})
                        n_merged_embedding += 1
                    elif cos_val >= MERGE_JUDGE_BAND[0]:
                        judge_pairs.append((i, j, cos_val))

            del sim_matrix

        # 6.3 Qwen merge judge (tiny, constrained)
        if LLM_MERGE_JUDGE and judge_pairs:
            # Sort by cosine descending, take top MERGE_JUDGE_MAX
            judge_pairs.sort(key=lambda x: x[2], reverse=True)
            judge_pairs = judge_pairs[:MERGE_JUDGE_MAX]

            try:
                import ollama

                judge_schema = {
                    "type": "object",
                    "properties": {"same": {"type": "boolean"}},
                    "required": ["same"],
                }

                for i, j, cos_val in judge_pairs:
                    k1 = concept_keys_list[i]
                    k2 = concept_keys_list[j]

                    if uf.find(k1) == uf.find(k2):
                        continue

                    name1 = surface_forms.get(k1, Counter()).most_common(1)[0][0] if surface_forms.get(k1) else k1
                    name2 = surface_forms.get(k2, Counter()).most_common(1)[0][0] if surface_forms.get(k2) else k2
                    ctx1 = concept_meta[k1].get("best_sentence", "")
                    ctx2 = concept_meta[k2].get("best_sentence", "")

                    prompt_text = (
                        f"Do these two terms from the same document refer to the same concept?\n"
                        f"A: {name1}\n"
                        f"B: {name2}\n"
                        f"Context A: {ctx1}\n"
                        f"Context B: {ctx2}\n"
                        f'Answer with JSON {{"same": true|false}}. Say true only if they are the same thing, not merely related.'
                    )

                    try:
                        resp = ollama.chat(
                            model=EXTRACT_MODEL,
                            messages=[{"role": "user", "content": prompt_text}],
                            format=judge_schema,
                            options={
                                "temperature": 0,
                                "num_ctx": 2048,
                                "num_predict": 16,
                            },
                            keep_alive="10m",
                        )
                        result = json.loads(resp["message"]["content"])
                        n_judged += 1
                        if result.get("same", False):
                            uf.union(k1, k2)
                            merge_report.append({"a": k1, "b": k2, "cosine": round(cos_val, 4), "decision": "judge_yes", "by": "qwen"})
                            n_merged_embedding += 1
                        else:
                            merge_report.append({"a": k1, "b": k2, "cosine": round(cos_val, 4), "decision": "judge_no", "by": "qwen"})
                    except Exception:
                        merge_report.append({"a": k1, "b": k2, "cosine": round(cos_val, 4), "decision": "judge_error", "by": "qwen"})

                # Unload the model
                try:
                    ollama.generate(model=EXTRACT_MODEL, prompt="", keep_alive=0)
                except Exception:
                    pass
            except ImportError:
                pass

        # Map each raw key to canonical key
        canonical_map: Dict[str, str] = {}
        for k in raw_keys:
            canonical_map[k] = uf.find(k)

        # Also map acronym aliases directly
        for acr, exp in acronym_aliases.items():
            if exp in canonical_map:
                canonical_map[acr] = canonical_map[exp]

        # Best display name per canonical key: highest support surface form
        canonical_surfaces: Dict[str, Counter] = defaultdict(Counter)
        for k, counter in surface_forms.items():
            canon = canonical_map[k]
            canonical_surfaces[canon].update(counter)

        display_names: Dict[str, str] = {}
        for canon, counter in canonical_surfaces.items():
            display_names[canon] = counter.most_common(1)[0][0]

        # Determine kind per canonical key: majority vote with tie-break order
        kind_priority = {"category": 0, "component": 1, "practice": 2, "attribute": 3, "example": 4}
        canonical_kinds: Dict[str, Counter] = defaultdict(Counter)
        for k, counter in kind_votes.items():
            canon = canonical_map[k]
            canonical_kinds[canon].update(counter)

        canon_kind: Dict[str, str] = {}
        for canon, counter in canonical_kinds.items():
            if not counter:
                canon_kind[canon] = "category"
                continue
            max_count = counter.most_common(1)[0][1]
            tied = [k for k, v in counter.items() if v == max_count]
            if len(tied) == 1:
                canon_kind[canon] = tied[0]
            else:
                # Tie-break: category > component > practice > attribute > example
                canon_kind[canon] = min(tied, key=lambda x: kind_priority.get(x, 99))

        # 7. Aggregate concepts: support, sources, chunk_ids, main_topic_count, first_order, best_sentence
        concept_support: Dict[str, Set[str]] = defaultdict(set)  # canon -> set of chunk_ids
        concept_sources: Dict[str, Set[str]] = defaultdict(set)  # canon -> set of source_files
        concept_main_topic_count: Dict[str, int] = defaultdict(int)
        concept_first_order: Dict[str, int] = {}
        concept_best_sentence: Dict[str, str] = {}

        for k in raw_keys:
            canon = canonical_map.get(k, k)
            meta = concept_meta[k]
            concept_support[canon].update(meta["chunk_ids"])
            concept_sources[canon].update(meta["sources"])
            concept_main_topic_count[canon] += meta["main_topic_count"]
            if canon not in concept_first_order or meta["first_order"] < concept_first_order[canon]:
                concept_first_order[canon] = meta["first_order"]
            if meta["best_sentence"] and (canon not in concept_best_sentence or not concept_best_sentence[canon]):
                concept_best_sentence[canon] = meta["best_sentence"]

        # 8. Aggregate relations
        rel_chunks: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
        rel_evidence: Dict[Tuple[str, str, str], Tuple[int, str]] = {}  # -> (best_order, evidence)

        for rec in records:
            cid = rec.get("chunk_id", "")
            order = rec.get("order", 999999)
            for r in rec.get("relations", []):
                s_raw = r.get("source", "").strip()
                rel_type = r.get("relation", "").strip()
                t_raw = r.get("target", "").strip()
                ev = r.get("evidence", "")
                s_key = normalize_key(s_raw)
                t_key = normalize_key(t_raw)
                s_canon = canonical_map.get(s_key, s_key)
                t_canon = canonical_map.get(t_key, t_key)

                if s_canon and t_canon and s_canon != t_canon and rel_type in REL_WEIGHT:
                    triple = (s_canon, rel_type, t_canon)
                    rel_chunks[triple].add(cid)
                    # Track highest-order-support evidence
                    if triple not in rel_evidence or order < rel_evidence[triple][0]:
                        rel_evidence[triple] = (order, ev)

        # Compute support & edge weights
        aggregated_rels: Dict[Tuple[str, str, str], dict] = {}
        for (s, rel, t), chks in rel_chunks.items():
            supp = len(chks)
            w = REL_WEIGHT[rel] * (1.0 + math.log(supp))
            aggregated_rels[(s, rel, t)] = {
                "support": supp,
                "weight": round(w, 3),
                "chunk_ids": sorted(list(chks)),
                "evidence": rel_evidence.get((s, rel, t), (0, ""))[1],
            }

        # 9. Hierarchy sanity check
        pair_to_hier: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)
        for (s, rel, t) in list(aggregated_rels.keys()):
            if rel in HIERARCHY_RELS:
                ordered_pair = tuple(sorted([s, t]))
                pair_to_hier[ordered_pair].append((s, rel, t))

        for pair, edges in pair_to_hier.items():
            if len(edges) > 1:
                dir_a = [e for e in edges if e[0] == pair[0]]
                dir_b = [e for e in edges if e[0] == pair[1]]
                if dir_a and dir_b:
                    supp_a = max(aggregated_rels[e]["support"] for e in dir_a)
                    supp_b = max(aggregated_rels[e]["support"] for e in dir_b)
                    if supp_a > supp_b:
                        for e in dir_b:
                            del aggregated_rels[e]
                    elif supp_b > supp_a:
                        for e in dir_a:
                            del aggregated_rels[e]
                    else:
                        for e in edges:
                            if e in aggregated_rels:
                                del aggregated_rels[e]

        # Break cycles in hierarchy subgraph
        hier_graph = nx.DiGraph()
        for (s, rel, t), data in aggregated_rels.items():
            if rel in HIERARCHY_RELS:
                hier_graph.add_edge(s, t, weight=data["weight"], rel_tuple=(s, rel, t))

        while True:
            try:
                cycle = nx.find_cycle(hier_graph, orientation="original")
                weakest_edge = None
                min_w = float("inf")
                for u, v, _ in cycle:
                    w = hier_graph[u][v]["weight"]
                    if w < min_w:
                        min_w = w
                        weakest_edge = (u, v)

                if weakest_edge:
                    u, v = weakest_edge
                    rel_tup = hier_graph[u][v]["rel_tuple"]
                    hier_graph.remove_edge(u, v)
                    if rel_tup in aggregated_rels:
                        del aggregated_rels[rel_tup]
                else:
                    break
            except nx.NetworkXNoCycle:
                break

        # 10. Global topics identification
        global_topics: List[str] = []
        for canon, chks in concept_support.items():
            freq = len(chks) / max(1, processed_chunks_count)
            if freq > GLOBAL_FREQ:
                global_topics.append(display_names.get(canon, canon))

        global_canon_set = {normalize_key(g) for g in global_topics}

        # 11. Linked concepts vs Unlinked
        linked_canon: Set[str] = set()
        for (s, rel, t) in aggregated_rels.keys():
            linked_canon.add(s)
            linked_canon.add(t)

        concepts_out = []
        unlinked_out = []

        for canon, chks in concept_support.items():
            name = display_names.get(canon, canon)
            is_glob = canon in global_canon_set
            c_entry = {
                "key": canon,
                "name": name,
                "kind": canon_kind.get(canon, "category"),
                "support": len(chks),
                "main_topic_count": concept_main_topic_count.get(canon, 0),
                "first_order": concept_first_order.get(canon, 999999),
                "sources": sorted(list(concept_sources[canon])),
                "chunk_ids": sorted(list(chks)),
                "best_sentence": concept_best_sentence.get(canon, ""),
                "global": is_glob,
            }
            if canon in linked_canon:
                concepts_out.append(c_entry)
            else:
                unlinked_out.append(name)

        relations_out = []
        n_hierarchy = 0
        for (s, rel, t), data in aggregated_rels.items():
            if rel in HIERARCHY_RELS:
                n_hierarchy += 1
            relations_out.append({
                "source": display_names.get(s, s),
                "source_key": s,
                "relation": rel,
                "target": display_names.get(t, t),
                "target_key": t,
                "support": data["support"],
                "weight": data["weight"],
                "chunk_ids": data["chunk_ids"],
                "evidence": data["evidence"],
            })

        # 12. Concept vectors for Stage 3
        # Embed "name. best_sentence" for every non-global linked concept
        vec_concepts = [c for c in concepts_out if not c.get("global", False)]
        concept_vectors = None

        if vec_concepts:
            try:
                from sentence_transformers import SentenceTransformer
                embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
                texts_to_embed = []
                for c in vec_concepts:
                    sent = c.get("best_sentence", "")
                    name = c.get("name", "")
                    texts_to_embed.append(f"{name}. {sent}" if sent else name)

                concept_vectors = embed_model.encode(
                    texts_to_embed,
                    normalize_embeddings=True,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)

                del embed_model
                gc.collect()
            except Exception:
                concept_vectors = None

        # Assign vec_index to each concept
        vec_key_to_idx = {}
        if vec_concepts and concept_vectors is not None:
            for vi, c in enumerate(vec_concepts):
                vec_key_to_idx[c["key"]] = vi

        for c in concepts_out:
            c["vec_index"] = vec_key_to_idx.get(c["key"], -1)

        # Save concept_vectors.npy
        if concept_vectors is not None:
            np.save(vectors_path, concept_vectors)

        # Free embedding memory
        del embeddings
        gc.collect()

        stats = {
            "n_concepts": len(concepts_out),
            "n_relations": len(relations_out),
            "n_hierarchy_edges": n_hierarchy,
            "n_global": len(global_topics),
            "n_unlinked": len(unlinked_out),
            "n_merged_lexical": n_merged_lexical,
            "n_merged_embedding": n_merged_embedding,
            "n_judged": n_judged,
        }

        output_payload = {
            "concepts": concepts_out,
            "relations": relations_out,
            "global_topics": global_topics,
            "unlinked": unlinked_out,
            "merge_report": merge_report,
            "stats": stats,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(cur_hash)

        return {"status": "ok", **stats}
