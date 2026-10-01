"""
representation/build_concept_graph.py
Stage 2: Consolidate concepts and typed relations into a clean graph.
Pure algorithmic (no LLM, no embeddings).
Normalizes keys, resolves acronym aliases, fuzzy-merges variations,
enforces hierarchy DAG consistency, and filters global hub topics.
"""
from collections import Counter, defaultdict
import difflib
import hashlib
import json
import math
import os
import re
from typing import Dict, List, Set, Tuple

import networkx as nx

from representation import paths as rep_paths
from representation.concept_schema import (
    GLOBAL_FREQ,
    HIERARCHY_RELS,
    REL_WEIGHT,
)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


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
    """
    raw_path = rep_paths.concepts_raw_path(session_id)
    if not os.path.exists(raw_path):
        return {"status": "error", "message": f"concepts_raw.jsonl not found for session {session_id}"}

    chunks_file = rep_paths.chunks_jsonl_path(session_id)
    out_path = rep_paths.concept_graph_path(session_id)
    hash_path = rep_paths.concept_graph_hash_path(session_id)

    # Check cache
    mtime = str(os.path.getmtime(raw_path))
    cur_hash = hashlib.md5(f"stage2:{mtime}".encode("utf-8")).hexdigest()
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

        # 3. Collect surface forms and raw keys
        raw_keys: Set[str] = set()
        surface_forms: Dict[str, Counter] = defaultdict(Counter)

        for rec in records:
            for c in rec.get("concepts", []):
                raw_name = c.get("name", "").strip()
                if not raw_name:
                    continue
                k = normalize_key(raw_name)
                if k:
                    raw_keys.add(k)
                    surface_forms[k][raw_name] += 1

        # 4. Union-Find clustering for aliases & fuzzy matches
        uf = UnionFind()
        for k in raw_keys:
            uf.find(k)

        # Apply acronym aliases
        for acr, exp in acronym_aliases.items():
            if acr in raw_keys and exp in raw_keys:
                uf.union(exp, acr)

        # Fuzzy merge within blocks (same first char & similar length)
        # Block by first character
        by_first_char = defaultdict(list)
        for k in raw_keys:
            by_first_char[k[0]].append(k)

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

        # Map each raw key to canonical key
        canonical_map: Dict[str, str] = {}
        for k in raw_keys:
            canonical_map[k] = uf.find(k)

        # Also map acronym aliases directly
        for acr, exp in acronym_aliases.items():
            if exp in canonical_map:
                canonical_map[acr] = canonical_map[exp]

        # Best display name per canonical key
        canonical_surfaces: Dict[str, Counter] = defaultdict(Counter)
        for k, counter in surface_forms.items():
            canon = canonical_map[k]
            canonical_surfaces[canon].update(counter)

        display_names: Dict[str, str] = {}
        for canon, counter in canonical_surfaces.items():
            # Pick most frequent surface form
            display_names[canon] = counter.most_common(1)[0][0]

        # 5. Aggregate concepts: support, sources, chunk_ids
        concept_support: Dict[str, Set[str]] = defaultdict(set) # canon -> set of chunk_ids
        concept_sources: Dict[str, Set[str]] = defaultdict(set) # canon -> set of source_files

        for rec in records:
            cid = rec.get("chunk_id", "")
            sfile = rec.get("source_file", "")
            seen_in_chunk = set()
            for c in rec.get("concepts", []):
                raw_name = c.get("name", "").strip()
                k = normalize_key(raw_name)
                if not k:
                    continue
                canon = canonical_map.get(k, k)
                seen_in_chunk.add(canon)

            for canon in seen_in_chunk:
                concept_support[canon].add(cid)
                if sfile:
                    concept_sources[canon].add(sfile)

        # 6. Aggregate relations
        # Map: (canon_src, rel, canon_tgt) -> set of chunk_ids
        rel_chunks: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

        for rec in records:
            cid = rec.get("chunk_id", "")
            for r in rec.get("relations", []):
                s_raw = r.get("source", "").strip()
                rel_type = r.get("relation", "").strip()
                t_raw = r.get("target", "").strip()
                s_key = normalize_key(s_raw)
                t_key = normalize_key(t_raw)
                s_canon = canonical_map.get(s_key, s_key)
                t_canon = canonical_map.get(t_key, t_key)

                if s_canon and t_canon and s_canon != t_canon and rel_type in REL_WEIGHT:
                    rel_chunks[(s_canon, rel_type, t_canon)].add(cid)

        # Compute support & edge weights
        # Structure: {(s, rel, t): {"support": N, "weight": W}}
        aggregated_rels: Dict[Tuple[str, str, str], dict] = {}
        for (s, rel, t), chks in rel_chunks.items():
            supp = len(chks)
            w = REL_WEIGHT[rel] * (1.0 + math.log(supp))
            aggregated_rels[(s, rel, t)] = {
                "support": supp,
                "weight": round(w, 3),
            }

        # 7. Hierarchy sanity check:
        # Check bidirectional hierarchy edges
        pair_to_hier: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)
        for (s, rel, t) in list(aggregated_rels.keys()):
            if rel in HIERARCHY_RELS:
                ordered_pair = tuple(sorted([s, t]))
                pair_to_hier[ordered_pair].append((s, rel, t))

        for pair, edges in pair_to_hier.items():
            if len(edges) > 1:
                # Check if opposite directions exist
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
                        # Tie: drop both
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
                # Find weakest edge in cycle
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

        # 8. Global topics identification
        global_topics: List[str] = []
        for canon, chks in concept_support.items():
            freq = len(chks) / max(1, processed_chunks_count)
            if freq > GLOBAL_FREQ:
                global_topics.append(display_names.get(canon, canon))

        global_canon_set = {normalize_key(g) for g in global_topics}

        # 9. Linked concepts vs Unlinked
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
                "support": len(chks),
                "sources": sorted(list(concept_sources[canon])),
                "chunk_ids": sorted(list(chks)),
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
            })

        stats = {
            "n_concepts": len(concepts_out),
            "n_relations": len(relations_out),
            "n_hierarchy_edges": n_hierarchy,
            "n_global": len(global_topics),
            "n_unlinked": len(unlinked_out),
        }

        output_payload = {
            "concepts": concepts_out,
            "relations": relations_out,
            "global_topics": global_topics,
            "unlinked": unlinked_out,
            "stats": stats,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(cur_hash)

        return {"status": "ok", **stats}
