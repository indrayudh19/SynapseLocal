"""
representation/cluster.py
Stage 3: Concept Community Detection and Hub-and-Spoke Clustering.
Completely replaces old KMeans/PCA clustering.
Pure algorithmic (no LLM, no embeddings).
"""
import hashlib
import json
import math
import os
from typing import Dict, List, Set, Tuple

import networkx as nx
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity as nx_modularity
import numpy as np

from representation import paths as rep_paths
from representation.concept_plot import render_concept_map
from representation.concept_schema import (
    HIERARCHY_RELS,
    MIN_CLUSTER_SIZE,
)


def run(session_id: str) -> dict:
    """
    Execute Stage 3: Community detection on concept graph, hub identification,
    and rendering to concept_map.png.
    """
    graph_path = rep_paths.concept_graph_path(session_id)
    if not os.path.exists(graph_path):
        return {"status": "error", "message": f"concept_graph.json not found for session {session_id}"}

    out_json = rep_paths.concept_clusters_path(session_id)
    out_png = rep_paths.concept_map_path(session_id)
    hash_path = rep_paths.concept_clusters_hash_path(session_id)

    # Check cache
    mtime = str(os.path.getmtime(graph_path))
    cur_hash = hashlib.md5(f"stage3:{mtime}".encode("utf-8")).hexdigest()
    if os.path.exists(hash_path) and os.path.exists(out_json):
        try:
            with open(hash_path, "r", encoding="utf-8") as f:
                if f.read().strip() == cur_hash:
                    with open(out_json, "r", encoding="utf-8") as jf:
                        cached_data = json.load(jf)
                    return {"status": "ok", "cached": True, **cached_data.get("metrics", {})}
        except Exception:
            pass

    with rep_paths.session_lock(session_id):
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        concepts_raw_list = graph_data.get("concepts", [])
        relations_raw_list = graph_data.get("relations", [])
        global_topics = graph_data.get("global_topics", [])

        # Concept index by key and name
        concept_by_key: Dict[str, dict] = {c["key"]: c for c in concepts_raw_list}
        name_to_key: Dict[str, str] = {c["name"]: c["key"] for c in concepts_raw_list}

        # Filter out global topics
        non_global_keys: Set[str] = {c["key"] for c in concepts_raw_list if not c.get("global", False)}

        # Filter relations where both endpoints are non-global
        valid_relations = []
        for r in relations_raw_list:
            s_key = r.get("source_key") or name_to_key.get(r.get("source"))
            t_key = r.get("target_key") or name_to_key.get(r.get("target"))
            if s_key in non_global_keys and t_key in non_global_keys:
                valid_relations.append({**r, "source_key": s_key, "target_key": t_key})

        linked_concepts = set()
        for r in valid_relations:
            linked_concepts.add(r["source_key"])
            linked_concepts.add(r["target_key"])

        # 6.1 Quality gate
        if len(linked_concepts) < 8 or len(valid_relations) < 6:
            gate_payload = {
                "status": "insufficient_structure",
                "message": (
                    f"Insufficient structure detected ({len(linked_concepts)} linked concepts, "
                    f"{len(valid_relations)} relations). Please process more chunks or upload richer study materials."
                ),
                "n_concepts": len(linked_concepts),
                "n_relations": len(valid_relations),
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(gate_payload, f, indent=2)
            with open(hash_path, "w", encoding="utf-8") as f:
                f.write(cur_hash)
            return gate_payload

        # 6.2 Community detection
        # Build undirected weighted graph
        G = nx.Graph()
        for k in linked_concepts:
            G.add_node(k)

        for r in valid_relations:
            u, v = r["source_key"], r["target_key"]
            w = r.get("weight", 1.0)
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)

        # Global PageRank for hub ranking
        pagerank_scores = nx.pagerank(G, weight="weight")

        # Hierarchy edges directed lookup
        # (src, tgt) -> list of hierarchy relations
        hier_edges: Dict[Tuple[str, str], str] = {}
        for r in valid_relations:
            if r.get("relation") in HIERARCHY_RELS:
                hier_edges[(r["source_key"], r["target_key"])] = r["relation"]

        best_comms = None
        best_modularity = -1.0
        chosen_r = 1.0

        for r_cand in [0.8, 1.0, 1.3, 1.7]:
            comms = louvain_communities(G, weight="weight", resolution=r_cand, seed=42)
            if not comms:
                continue
            sizes = [len(c) for c in comms]
            med_size = float(np.median(sizes))
            mod = nx_modularity(G, comms, weight="weight")
            if 3.0 <= med_size <= 12.0:
                if mod > best_modularity:
                    best_modularity = mod
                    best_comms = comms
                    chosen_r = r_cand

        if best_comms is None:
            best_comms = louvain_communities(G, weight="weight", resolution=1.0, seed=42)
            chosen_r = 1.0
            best_modularity = nx_modularity(G, best_comms, weight="weight")

        # Convert communities to mutable list of sets
        communities = [set(c) for c in best_comms]

        # 6.2.3 Merge small communities (< MIN_CLUSTER_SIZE)
        orphans: List[str] = []
        changed = True
        while changed:
            changed = False
            for i in range(len(communities) - 1, -1, -1):
                comm = communities[i]
                if len(comm) < MIN_CLUSTER_SIZE:
                    # Find neighboring community with highest inter-edge weight
                    best_neighbor_idx = None
                    max_inter_w = 0.0
                    for j in range(len(communities)):
                        if i == j:
                            continue
                        inter_w = 0.0
                        for u in comm:
                            for v in communities[j]:
                                if G.has_edge(u, v):
                                    inter_w += G[u][v]["weight"]
                        if inter_w > max_inter_w:
                            max_inter_w = inter_w
                            best_neighbor_idx = j

                    if best_neighbor_idx is not None and max_inter_w > 0:
                        communities[best_neighbor_idx].update(comm)
                        communities.pop(i)
                        changed = True
                        break
                    else:
                        # No neighbors: mark as orphans
                        for u in comm:
                            orphans.append(concept_by_key[u]["name"])
                        communities.pop(i)
                        changed = True
                        break

        # Map each node to its community index
        node_to_cluster: Dict[str, int] = {}
        for c_idx, comm in enumerate(communities):
            for node in comm:
                node_to_cluster[node] = c_idx

        # 6.3 Hub selection per community
        clusters_out = []
        total_spokes = 0
        direct_spokes = 0
        cross_source_set = set()

        for c_idx, comm in enumerate(communities):
            comm_list = list(comm)
            
            # Hub scoring tuple: (hier_in_degree, weighted_degree_in_comm, pagerank, display_name)
            def hub_score(node):
                # Count in-community nodes pointing to this node via hierarchy relations
                hier_in = sum(1 for other in comm_list if other != node and (other, node) in hier_edges)
                w_deg = sum(G[node][nbr]["weight"] for nbr in G.neighbors(node) if nbr in comm)
                pr = pagerank_scores.get(node, 0.0)
                name = concept_by_key[node]["name"]
                return (hier_in, w_deg, pr, name)

            # Sort descending by hub score
            sorted_nodes = sorted(comm_list, key=hub_score, reverse=True)
            hub_key = sorted_nodes[0]
            hub_name = concept_by_key[hub_key]["name"]

            members_out = []
            source_counter: Dict[str, int] = {}

            for node in sorted_nodes:
                c_data = concept_by_key[node]
                srcs = c_data.get("sources", [])
                if len(srcs) >= 2:
                    cross_source_set.add(node)
                for s in srcs:
                    source_counter[s] = source_counter.get(s, 0) + 1

                if node == hub_key:
                    continue  # spokes are members other than hub

                total_spokes += 1
                link_to_hub = "indirect"
                if (node, hub_key) in hier_edges:
                    rel_type = hier_edges[(node, hub_key)]
                    link_to_hub = f"direct:{rel_type}"
                    direct_spokes += 1
                elif (hub_key, node) in hier_edges:
                    rel_type = hier_edges[(hub_key, node)]
                    link_to_hub = f"direct:{rel_type}"
                    direct_spokes += 1

                # Weighted degree in G as member weight
                w_member = sum(G[node][nbr]["weight"] for nbr in G.neighbors(node) if nbr in comm)

                members_out.append({
                    "name": c_data["name"],
                    "weight": round(w_member, 2),
                    "link_to_hub": link_to_hub,
                    "sources": srcs,
                })

            # Calculate source mix fractions
            tot_src = sum(source_counter.values())
            source_mix = {
                sf: round(cnt / tot_src, 2)
                for sf, cnt in source_counter.items()
            } if tot_src > 0 else {}

            clusters_out.append({
                "id": c_idx,
                "hub": hub_name,
                "size": len(comm),
                "members": members_out,
                "source_mix": source_mix,
            })

        # 6.4 Cluster-level quotient graph
        # Inter-cluster edges
        # Map: (c_a, c_b) -> {"weight": total_w, "top_edge": ...}
        inter_links: Dict[Tuple[int, int], dict] = {}
        for r in valid_relations:
            s_key = r["source_key"]
            t_key = r["target_key"]
            if s_key not in node_to_cluster or t_key not in node_to_cluster:
                continue
            c_a = node_to_cluster[s_key]
            c_b = node_to_cluster[t_key]
            if c_a == c_b:
                continue

            pair = (min(c_a, c_b), max(c_a, c_b))
            w = r.get("weight", 1.0)
            if pair not in inter_links:
                inter_links[pair] = {
                    "weight": 0.0,
                    "top_edge": {
                        "source": r.get("source"),
                        "relation": r.get("relation"),
                        "target": r.get("target"),
                        "weight": w,
                    },
                }
            inter_links[pair]["weight"] += w
            if w > inter_links[pair]["top_edge"]["weight"]:
                inter_links[pair]["top_edge"] = {
                    "source": r.get("source"),
                    "relation": r.get("relation"),
                    "target": r.get("target"),
                    "weight": w,
                }

        # Keep per cluster only its top 2 strongest links
        cluster_edges: Dict[int, List[Tuple[float, Tuple[int, int]]]] = {c["id"]: [] for c in clusters_out}
        for (a, b), data in inter_links.items():
            w = data["weight"]
            cluster_edges[a].append((w, (a, b)))
            cluster_edges[b].append((w, (a, b)))

        kept_link_pairs: Set[Tuple[int, int]] = set()
        for cid, edges in cluster_edges.items():
            edges.sort(key=lambda x: x[0], reverse=True)
            for _, pair in edges[:2]:
                kept_link_pairs.add(pair)

        links_out = []
        for pair in kept_link_pairs:
            data = inter_links[pair]
            links_out.append({
                "a": pair[0],
                "b": pair[1],
                "weight": round(data["weight"], 2),
                "top_edge": {
                    "source": data["top_edge"]["source"],
                    "relation": data["top_edge"]["relation"],
                    "target": data["top_edge"]["target"],
                },
            })

        # 6.5 Metrics
        n_concepts_clustered = sum(c["size"] for c in clusters_out)
        hub_coverage = round(direct_spokes / max(1, total_spokes), 3)

        metrics = {
            "n_concepts_clustered": n_concepts_clustered,
            "n_clusters": len(clusters_out),
            "modularity": round(best_modularity, 3),
            "resolution": chosen_r,
            "hub_coverage": hub_coverage,
            "cross_source_concepts": len(cross_source_set),
            "n_orphans": len(orphans),
            "n_global": len(global_topics),
        }

        output_payload = {
            "status": "ok",
            "clusters": clusters_out,
            "links": links_out,
            "global_topics": global_topics,
            "orphans": orphans,
            "metrics": metrics,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        # 6.7 Render concept map
        render_concept_map(clusters_out, links_out, metrics, out_png)

        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(cur_hash)

        return {"status": "ok", **metrics}
