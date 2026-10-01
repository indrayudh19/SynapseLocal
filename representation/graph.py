"""
representation/graph.py
Stage 2 Service: Semantic kNN Knowledge Graph.
Reads embeddings.npy and chunks.jsonl, builds semantic graph, computes
PageRank, Louvain communities, cross-source ratios, and renders graph.png.

Strict service contract:
- Single entrypoint: run(session_id)
- NO embedding model, NO LLM imports
- Requires Stage 1 artifacts; returns clear error message if absent
- Isolated lock via .running file removed in finally block
- Persists: graph_edges.npz, graph_meta.json, graph.png, graph.hash
"""
import os
from representation.paths import (
    session_dir,
    representation_dir,
    embeddings_path,
    chunks_jsonl_path,
    graph_edges_path,
    graph_meta_path,
    graph_png_path,
    graph_hash_path,
    lock_path,
)


def _compute_input_hash(e_path: str, c_path: str) -> str:
    """Compute combined sha256 hash of Stage 1 input artifacts."""
    import hashlib
    hasher = hashlib.sha256()
    for p in [e_path, c_path]:
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()


def run(session_id: str) -> dict:
    """
    Run Stage 2: Construct knowledge graph and render static visualization.
    
    Args:
        session_id: Target session identifier
        
    Returns:
        Small stats dict
    """
    e_path = embeddings_path(session_id)
    c_path = chunks_jsonl_path(session_id)

    # 1. Enforce prerequisite: Stage 1 must exist
    if not (os.path.exists(e_path) and os.path.exists(c_path)):
        return {
            "status": "error",
            "message": "Stage 1 artifacts missing. Run embed_store first."
        }

    # 2. Enforce session lock
    s_dir = session_dir(session_id)
    l_path = lock_path(session_id)
    if os.path.exists(l_path):
        return {
            "status": "error",
            "message": f"Another service is currently running for session {session_id}."
        }

    with open(l_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    try:
        # Lazy imports of heavy libraries (NetworkX, Scipy, Sklearn, Matplotlib)
        import gc
        import json
        import numpy as np
        import scipy.sparse as sp
        import networkx as nx
        from representation.plotting import setup_matplotlib, extract_tfidf_labels

        r_dir = representation_dir(session_id)
        current_hash = _compute_input_hash(e_path, c_path)
        h_path = graph_hash_path(session_id)
        npz_path = graph_edges_path(session_id)
        meta_path = graph_meta_path(session_id)
        png_path = graph_png_path(session_id)

        # Check cache
        if (
            os.path.exists(h_path)
            and os.path.exists(npz_path)
            and os.path.exists(meta_path)
            and os.path.exists(png_path)
        ):
            with open(h_path, "r", encoding="utf-8") as f:
                if f.read().strip() == current_hash:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        cached_meta = json.load(mf)
                    return {
                        "status": "ok",
                        "nodes": cached_meta["global"]["nodes"],
                        "edges": cached_meta["global"]["edges"],
                        "communities": cached_meta["global"]["communities"],
                        "density": cached_meta["global"]["density"],
                        "cached": True,
                    }

        # Step A: Load inputs
        embeddings = np.load(e_path).astype(np.float32)
        chunks = []
        with open(c_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        n_chunks = len(chunks)
        if n_chunks == 0 or len(embeddings) != n_chunks:
            return {
                "status": "error",
                "message": f"Chunk count mismatch between embeddings ({len(embeddings)}) and chunks.jsonl ({n_chunks})."
            }

        # Step B: kNN search (top k=6 neighbors per node, excluding self)
        k_neighbors = min(6, n_chunks - 1)
        sim_pairs = []  # list of (i, j, sim)
        all_top_sims = []

        if k_neighbors > 0:
            if n_chunks <= 5000:
                # Direct dot product (embeddings are L2-normalized, dot product == cosine similarity)
                sim_matrix = embeddings @ embeddings.T
                for i in range(n_chunks):
                    # Exclude self
                    sim_matrix[i, i] = -1.0
                    top_j = np.argpartition(sim_matrix[i], -k_neighbors)[-k_neighbors:]
                    for j in top_j:
                        val = float(sim_matrix[i, j])
                        if val > 0:
                            sim_pairs.append((i, j, val))
                            all_top_sims.append(val)
                del sim_matrix
            else:
                import faiss
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                D, I = index.search(embeddings, k_neighbors + 1)
                for i in range(n_chunks):
                    for idx_pos in range(k_neighbors + 1):
                        j = int(I[i, idx_pos])
                        val = float(D[i, idx_pos])
                        if j != i and j >= 0:
                            sim_pairs.append((i, j, val))
                            all_top_sims.append(val)
                del index

        # Step C: Edge filtering with adaptive threshold
        if all_top_sims:
            mean_sim = float(np.mean(all_top_sims))
            std_sim = float(np.std(all_top_sims))
            tau = max(0.45, mean_sim + 1.0 * std_sim)
        else:
            tau = 0.45

        G = nx.Graph()
        for idx, chunk in enumerate(chunks):
            G.add_node(
                idx,
                id=chunk.get("id", f"c_{idx}"),
                parent_id=chunk.get("parent_id", ""),
                source_file=chunk.get("source_file", "unknown"),
                source_type=chunk.get("source_type", "DOC"),
                page_or_slide=chunk.get("page_or_slide", 1),
                text=chunk.get("text", ""),
            )

        # Collect candidate edges
        candidate_edges = {}
        for i, j, sim in sim_pairs:
            if sim >= tau:
                edge_key = (min(i, j), max(i, j))
                if edge_key not in candidate_edges or sim > candidate_edges[edge_key]:
                    candidate_edges[edge_key] = sim

        # Filter duplicate parent-adjacent edges:
        # Keep them only if either node would otherwise have degree 0
        node_degrees = {i: 0 for i in range(n_chunks)}
        for (u, v), sim in candidate_edges.items():
            if chunks[u].get("parent_id") != chunks[v].get("parent_id"):
                node_degrees[u] += 1
                node_degrees[v] += 1

        for (u, v), sim in candidate_edges.items():
            is_same_parent = (chunks[u].get("parent_id") == chunks[v].get("parent_id"))
            if not is_same_parent or (node_degrees[u] == 0 or node_degrees[v] == 0):
                G.add_edge(u, v, weight=sim)
                node_degrees[u] += 1
                node_degrees[v] += 1

        # Step D: Graph metrics (networkx, weighted)
        try:
            pagerank = nx.pagerank(G, weight="weight", alpha=0.85)
        except Exception:
            pagerank = {i: 1.0 / n_chunks for i in range(n_chunks)}

        degree = dict(G.degree())

        # Cross-source edge ratio
        cross_source_ratio = {}
        for i in range(n_chunks):
            neighbors = list(G.neighbors(i))
            if not neighbors:
                cross_source_ratio[i] = 0.0
            else:
                src_i = chunks[i].get("source_file")
                cross_count = sum(1 for nbr in neighbors if chunks[nbr].get("source_file") != src_i)
                cross_source_ratio[i] = round(cross_count / len(neighbors), 4)

        # Louvain communities
        try:
            communities_list = list(nx.community.louvain_communities(G, weight="weight", seed=42))
        except Exception:
            communities_list = [{i} for i in range(n_chunks)]

        node_to_community = {}
        for cid, comm in enumerate(communities_list):
            for node_idx in comm:
                node_to_community[node_idx] = cid

        # Community TF-IDF labels
        comm_texts = {}
        for cid, comm in enumerate(communities_list):
            comm_texts[cid] = [chunks[idx].get("text", "") for idx in comm]
        comm_labels = extract_tfidf_labels(comm_texts, top_n=3)

        # Step E: Build metadata and persist
        nodes_meta = []
        for i in range(n_chunks):
            c = chunks[i]
            nodes_meta.append({
                "id": c.get("id", f"c_{i}"),
                "index": i,
                "source_file": c.get("source_file", "unknown"),
                "source_type": c.get("source_type", "DOC"),
                "page_or_slide": c.get("page_or_slide", 1),
                "pagerank": round(pagerank.get(i, 0.0), 6),
                "degree": degree.get(i, 0),
                "community": node_to_community.get(i, 0),
                "cross_source_ratio": cross_source_ratio.get(i, 0.0),
                "text_snippet": c.get("text", "")[:120],
            })

        comm_meta = []
        for cid, comm in enumerate(communities_list):
            sources = sorted(list(set(chunks[idx].get("source_file", "") for idx in comm)))
            comm_meta.append({
                "community": cid,
                "label": comm_labels.get(cid, f"Community {cid}"),
                "size": len(comm),
                "sources": sources,
            })

        global_stats = {
            "nodes": n_chunks,
            "edges": G.number_of_edges(),
            "density": round(nx.density(G), 5),
            "communities": len(communities_list),
            "threshold_tau": round(tau, 4),
        }

        # Persist sparse adjacency
        adj_matrix = nx.to_scipy_sparse_array(G, weight="weight", format="csr")
        sp.save_npz(npz_path, adj_matrix)

        # Persist JSON metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "global": global_stats,
                "nodes": nodes_meta,
                "communities": comm_meta,
            }, f, indent=2)

        # Step F: Render once to graph.png (static matplotlib)
        plt = setup_matplotlib()
        fig, ax = plt.subplots(figsize=(9, 7), dpi=110)
        fig.patch.set_facecolor("#050507")
        ax.set_facecolor("#050507")

        # Select at most 120 nodes: top by PageRank + top of each community
        selected_nodes = set()
        for comm in communities_list:
            if comm:
                top_comm_node = max(comm, key=lambda idx: pagerank.get(idx, 0.0))
                selected_nodes.add(top_comm_node)

        # Fill remainder with highest PageRank nodes
        sorted_by_pr = sorted(range(n_chunks), key=lambda idx: pagerank.get(idx, 0.0), reverse=True)
        for idx in sorted_by_pr:
            if len(selected_nodes) >= min(120, n_chunks):
                break
            selected_nodes.add(idx)

        sub_G = G.subgraph(selected_nodes)

        # Subgraph layout
        pos = nx.spring_layout(sub_G, k=0.6, iterations=40, seed=42)

        # Palette for communities
        palette = [
            "#00ff66", "#a855f7", "#38ef7d", "#c084fc",
            "#06b6d4", "#f59e0b", "#ef4444", "#ec4899",
            "#14b8a6", "#8b5cf6", "#f97316", "#22c55e",
        ]

        # Draw edges: differentiate intra-source vs cross-source
        intra_edges = []
        cross_edges = []
        for u, v in sub_G.edges():
            if chunks[u].get("source_file") == chunks[v].get("source_file"):
                intra_edges.append((u, v))
            else:
                cross_edges.append((u, v))

        # Intra-source edges: subtle / lighter
        nx.draw_networkx_edges(
            sub_G, pos, edgelist=intra_edges,
            ax=ax, edge_color="#272738", width=0.9, alpha=0.5
        )
        # Cross-source edges: prominent / purple neon
        nx.draw_networkx_edges(
            sub_G, pos, edgelist=cross_edges,
            ax=ax, edge_color="#a855f7", width=1.6, alpha=0.85
        )

        # Draw nodes by source type marker
        marker_map = {
            "PDF": "o",
            "PPTX": "s",
            "PPT": "s",
            "TXT": "^",
            "MD": "D",
        }

        max_pr = max(pagerank.values()) if pagerank else 1.0

        # Group selected nodes by source_type for clean scatter rendering
        nodes_by_marker = {}
        for node in sub_G.nodes():
            stype = chunks[node].get("source_type", "DOC").upper()
            m = marker_map.get(stype, "o")
            nodes_by_marker.setdefault(m, []).append(node)

        for m, m_nodes in nodes_by_marker.items():
            node_colors = [palette[node_to_community.get(n, 0) % len(palette)] for n in m_nodes]
            node_sizes = [max(40, (pagerank.get(n, 0.0) / max_pr) * 450) for n in m_nodes]
            nx.draw_networkx_nodes(
                sub_G, pos, nodelist=m_nodes,
                ax=ax, node_color=node_colors, node_size=node_sizes,
                node_shape=m, edgecolors="#ffffff", linewidths=0.5, alpha=0.92
            )

        # Label top 15 nodes by PageRank using TF-IDF key terms
        top_15_nodes = sorted(list(selected_nodes), key=lambda idx: pagerank.get(idx, 0.0), reverse=True)[:15]
        labels_dict = {}
        for n in top_15_nodes:
            # Extract 1-2 words from text or heading
            txt = chunks[n].get("text", "")
            words = [w for w in txt.split() if len(w) > 3 and w.isalpha()][:2]
            labels_dict[n] = " ".join(words) if words else f"#{n}"

        nx.draw_networkx_labels(
            sub_G, pos, labels=labels_dict,
            font_size=7.5, font_color="#f3f4f6", font_family="sans-serif",
            ax=ax, bbox=dict(boxstyle="round,pad=0.2", facecolor="#12121a", alpha=0.75, edgecolor="#242436", lw=0.5)
        )

        ax.set_title(
            f"Knowledge Graph (Louvain Communities: {len(communities_list)} | Nodes: {n_chunks} | Cross-Edges: {len(cross_edges)})",
            color="#00ff66", fontsize=11, fontfamily="monospace", pad=12
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(png_path, dpi=110, facecolor="#050507", edgecolor="none")
        plt.close(fig)

        # Write hash file
        with open(h_path, "w", encoding="utf-8") as f:
            f.write(current_hash)

        stats = {
            "status": "ok",
            "nodes": n_chunks,
            "edges": G.number_of_edges(),
            "communities": len(communities_list),
            "density": round(nx.density(G), 5),
            "cached": False,
        }

        # Free in-memory objects
        del G
        del sub_G
        del adj_matrix
        del embeddings
        del chunks
        gc.collect()

        return stats

    finally:
        if os.path.exists(l_path):
            try:
                os.remove(l_path)
            except OSError:
                pass
