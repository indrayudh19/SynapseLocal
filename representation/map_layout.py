"""
representation/map_layout.py
Stage 4: Interactive Concept Map Layout & HTML Generation.
Precomputes concentric ring layout and classical MDS + relaxation placement
for an interactive SVG concept map.
Zero model calls, numpy + networkx + stdlib only.
"""

import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from representation.paths import (
    concept_clusters_path,
    concept_graph_path,
    concept_map_hash_path,
    concept_map_html_path,
    session_lock,
)

# -------------------------------------------------------------------------
# 3.1 Constants
# -------------------------------------------------------------------------
EXPAND_PX, DETAIL_PX = 240, 640
MAX_MEMBERS_PER_CLUSTER, MAX_MEMBERS_TOTAL = 60, 1500  # rest summarized as "+k more" in the panel
SPREAD = 1.4              # global spacing multiplier (the map must feel spread out)
HALO_GAP = 140            # min world-unit gap between cluster halos
HUB_R_BASE, HUB_R_PER = 34, 6  # hub radius = base + per*sqrt(n)
RING_START, RING_STEP, ARC_SPACING = 56, 52, 36
PALETTE = [
    "#e8703a", "#3b8fd9", "#4cae4f", "#d94b4b", "#9b6fd1", "#e0a458",
    "#8c6d62", "#5fb3b3", "#d96aa7", "#a0b2d6", "#b5c94f", "#7a7f8c"
]

LAYOUT_CONSTANTS_HASH = hashlib.sha256(
    f"{EXPAND_PX}-{DETAIL_PX}-{MAX_MEMBERS_PER_CLUSTER}-{MAX_MEMBERS_TOTAL}-{SPREAD}-{HALO_GAP}-{HUB_R_BASE}-{HUB_R_PER}-{RING_START}-{RING_STEP}-{ARC_SPACING}".encode("utf-8")
).hexdigest()[:12]


def _compute_input_hash(clusters_content: bytes, graph_content: bytes) -> str:
    """Computes SHA-256 hash of cluster and graph input files plus layout constants."""
    h = hashlib.sha256()
    h.update(clusters_content)
    h.update(graph_content)
    h.update(LAYOUT_CONSTANTS_HASH.encode("utf-8"))
    return h.hexdigest()


def _sanitize_for_script(json_str: str) -> str:
    """
    Sanitize JSON string for safe embedding into an HTML <script> tag.
    Replaces &, <, >, and unicode line terminators U+2028 and U+2029.
    """
    return (
        json_str.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _compute_cluster_halo_and_rings(n_members: int) -> Tuple[float, float, List[Tuple[float, int]]]:
    """
    Calculates hub radius, halo radius, and ring capacities for n_members.
    Returns: (hubR, haloR, rings_info)
    rings_info: list of (r_k, capacity_k)
    """
    hub_r = HUB_R_BASE + HUB_R_PER * math.sqrt(max(0, n_members))
    rings: List[Tuple[float, int]] = []
    
    if n_members == 0:
        r0 = hub_r + RING_START
        halo_r = r0 + 70.0
        return hub_r, halo_r, [(r0, math.floor(2 * math.pi * r0 / ARC_SPACING))]

    cur_members = n_members
    k = 0
    while cur_members > 0 or k == 0:
        r_k = hub_r + RING_START + k * RING_STEP
        cap_k = max(1, math.floor(2 * math.pi * r_k / ARC_SPACING))
        rings.append((r_k, cap_k))
        cur_members -= cap_k
        k += 1

    outermost_r = rings[-1][0]
    halo_r = outermost_r + 70.0
    return hub_r, halo_r, rings


def _place_clusters_mds_and_relax(
    n_clusters: int,
    halo_radii: List[float],
    links: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Step 3.3: Classical MDS on cluster affinity + SPREAD + relaxation (<= 300 iterations).
    Returns coords: np.ndarray of shape (n_clusters, 2).
    """
    if n_clusters == 0:
        return np.zeros((0, 2), dtype=float)
    if n_clusters == 1:
        return np.zeros((1, 2), dtype=float)
    if n_clusters == 2:
        d = (halo_radii[0] + halo_radii[1] + HALO_GAP) * SPREAD * 0.5
        return np.array([[-d, 0.0], [d, 0.0]], dtype=float)

    # Build affinity matrix M
    M = np.zeros((n_clusters, n_clusters), dtype=float)
    for link in links:
        a = link.get("a", 0)
        b = link.get("b", 0)
        if 0 <= a < n_clusters and 0 <= b < n_clusters and a != b:
            w = float(link.get("affinity", link.get("weight", 0.0)))
            if w > M[a, b]:
                M[a, b] = w
                M[b, a] = w

    # Distance matrix D_ab = 1 / (M_ab + 0.05)
    D = np.zeros((n_clusters, n_clusters), dtype=float)
    finite_mask = M > 0
    D[finite_mask] = 1.0 / (M[finite_mask] + 0.05)
    D[~finite_mask] = np.inf
    np.fill_diagonal(D, 0.0)

    finite_d = D[np.isfinite(D)]
    max_finite = float(np.max(finite_d)) if len(finite_d) > 0 else 10.0
    missing_dist = 1.5 * max_finite
    D[~np.isfinite(D)] = missing_dist
    np.fill_diagonal(D, 0.0)

    # Classical MDS centering
    H = np.eye(n_clusters) - (1.0 / n_clusters) * np.ones((n_clusters, n_clusters))
    B = -0.5 * H @ (D ** 2) @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    v0 = eigvecs[:, 0] * math.sqrt(max(float(eigvals[0]), 0.0))
    v1 = eigvecs[:, 1] * math.sqrt(max(float(eigvals[1]), 0.0))
    coords = np.column_stack([v0, v1])

    # Check degenerate / collapsed case
    std_coords = float(np.std(coords))
    if std_coords < 1e-4:
        rng = np.random.RandomState(42)
        coords = rng.randn(n_clusters, 2) * 150.0

    # Step 2: Multiply by SPREAD
    coords *= SPREAD

    # Step 3: Relaxation (<= 300 iterations)
    # Guarantee halos do not overlap and maintain minimum gap >= HALO_GAP
    for it in range(350):
        any_overlap = False
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dx = coords[j, 0] - coords[i, 0]
                dy = coords[j, 1] - coords[i, 1]
                dist = math.hypot(dx, dy)
                min_dist = halo_radii[i] + halo_radii[j] + HALO_GAP
                if dist < min_dist:
                    any_overlap = True
                    overlap = min_dist - dist
                    if dist < 1e-5:
                        ang = (i * 1.6180339887 + j) % (2.0 * math.pi)
                        ux, uy = math.cos(ang), math.sin(ang)
                    else:
                        ux, uy = dx / dist, dy / dist
                    push = overlap * 0.51
                    coords[i, 0] -= ux * push
                    coords[i, 1] -= uy * push
                    coords[j, 0] += ux * push
                    coords[j, 1] += uy * push
        if not any_overlap:
            break

    # Final enforcement pass to strictly guarantee min_gap >= HALO_GAP
    for _ in range(50):
        overlap_found = False
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dx = coords[j, 0] - coords[i, 0]
                dy = coords[j, 1] - coords[i, 1]
                dist = math.hypot(dx, dy)
                min_dist = halo_radii[i] + halo_radii[j] + HALO_GAP
                if dist < min_dist:
                    overlap_found = True
                    diff = (min_dist - dist) + 0.1
                    ux, uy = (dx / dist, dy / dist) if dist > 1e-5 else (1.0, 0.0)
                    coords[i, 0] -= ux * diff * 0.5
                    coords[i, 1] -= uy * diff * 0.5
                    coords[j, 0] += ux * diff * 0.5
                    coords[j, 1] += uy * diff * 0.5
        if not overlap_found:
            break

    return coords


def _compute_links_mst_and_labels(
    n_clusters: int,
    raw_links: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Step 3.4: Draw maximum spanning tree over cluster affinity + additional links
    with affinity >= median affinity.
    """
    if n_clusters < 2 or not raw_links:
        return []

    G = nx.Graph()
    G.add_nodes_from(range(n_clusters))

    weights: List[float] = []
    link_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for link in raw_links:
        a = int(link.get("a", 0))
        b = int(link.get("b", 0))
        if 0 <= a < n_clusters and 0 <= b < n_clusters and a != b:
            w = float(link.get("affinity", link.get("weight", 0.0)))
            weights.append(w)
            edge_key = tuple(sorted((a, b)))
            # keep link with largest weight if duplicates
            if edge_key not in link_map or w > link_map[edge_key].get("w", 0.0):
                top_edge = link.get("top_edge", {})
                link_map[edge_key] = {
                    "a": a,
                    "b": b,
                    "w": w,
                    "rel": top_edge.get("relation", ""),
                    "src": top_edge.get("source", ""),
                    "tgt": top_edge.get("target", ""),
                }
            G.add_edge(a, b, weight=w)

    # Maximum spanning forest (guard for disconnected components)
    tree_edges: Set[Tuple[int, int]] = set()
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(subgraph) > 1:
            mst = nx.maximum_spanning_tree(subgraph, weight="weight")
            for u, v in mst.edges():
                tree_edges.add(tuple(sorted((u, v))))

    median_w = float(np.median(weights)) if weights else 0.0

    out_links: List[Dict[str, Any]] = []
    for edge_key, l_data in link_map.items():
        is_tree = edge_key in tree_edges
        if is_tree or l_data["w"] >= median_w:
            out_links.append({
                "a": l_data["a"],
                "b": l_data["b"],
                "w": round(l_data["w"], 2),
                "rel": l_data["rel"],
                "src": l_data["src"],
                "tgt": l_data["tgt"],
                "tree": is_tree,
            })

    # Sort tree edges first, then by weight descending
    out_links.sort(key=lambda x: (not x["tree"], -x["w"]))
    return out_links


def run(session_id: str) -> Dict[str, Any]:
    """
    Main entry point for Stage 4: Layout & HTML Map Generation.
    Returns: {n_clusters, n_members_drawn, n_links, html_kb, status}
    """
    cl_path = concept_clusters_path(session_id)
    gr_path = concept_graph_path(session_id)
    out_html = concept_map_html_path(session_id)
    hash_p = concept_map_hash_path(session_id)

    if not os.path.exists(cl_path):
        return {"status": "no_clusters"}

    with session_lock(session_id):
        with open(cl_path, "rb") as f:
            clusters_raw = f.read()

        graph_raw = b""
        if os.path.exists(gr_path):
            with open(gr_path, "rb") as f:
                graph_raw = f.read()

        current_hash = _compute_input_hash(clusters_raw, graph_raw)

        # Check hash cache
        if os.path.exists(hash_p) and os.path.exists(out_html):
            try:
                with open(hash_p, "r", encoding="utf-8") as f:
                    cached_hash = f.read().strip()
                if cached_hash == current_hash:
                    # Return cached result stats
                    file_size_kb = round(os.path.getsize(out_html) / 1024, 1)
                    clusters_json = json.loads(clusters_raw.decode("utf-8"))
                    n_cl = len(clusters_json.get("clusters", []))
                    return {
                        "status": "ok",
                        "n_clusters": n_cl,
                        "n_members_drawn": -1,  # cached
                        "n_links": len(clusters_json.get("links", [])),
                        "html_kb": file_size_kb,
                        "cached": True,
                    }
            except Exception:
                pass

        # Parse inputs
        clusters_data = json.loads(clusters_raw.decode("utf-8"))
        if clusters_data.get("status") != "ok":
            return {"status": clusters_data.get("status", "error")}

        graph_data: Dict[str, Any] = {}
        if graph_raw:
            try:
                graph_data = json.loads(graph_raw.decode("utf-8"))
            except Exception:
                pass

        # Build concept & relation index from concept_graph.json
        concepts_by_name: Dict[str, Dict[str, Any]] = {}
        for c in graph_data.get("concepts", []):
            c_name = c.get("name")
            if c_name:
                concepts_by_name[c_name] = c

        # Compute global max support for importance normalization
        max_support = 1
        for c in concepts_by_name.values():
            s = int(c.get("support", 1))
            if s > max_support:
                max_support = s

        # Relations lookup by concept name
        relations_by_concept: Dict[str, List[Dict[str, Any]]] = {}
        for rel in graph_data.get("relations", []):
            s_name = rel.get("source")
            t_name = rel.get("target")
            r_type = rel.get("relation", "")
            ev = rel.get("evidence", "") or ""
            sup = int(rel.get("support", 1))

            if s_name:
                relations_by_concept.setdefault(s_name, []).append({
                    "t": t_name, "r": r_type, "ev": ev[:160], "sup": sup
                })
            if t_name:
                relations_by_concept.setdefault(t_name, []).append({
                    "t": s_name, "r": r_type, "ev": ev[:160], "sup": sup
                })

        for c_name in relations_by_concept:
            relations_by_concept[c_name].sort(key=lambda x: -x["sup"])

        raw_clusters = clusters_data.get("clusters", [])
        raw_bridges = clusters_data.get("bridges", [])
        n_clusters = len(raw_clusters)

        # -----------------------------------------------------------------
        # 3.2 Member sorting and caps
        # -----------------------------------------------------------------
        # Prepare member items per cluster
        cluster_members_prep: List[List[Dict[str, Any]]] = []
        for cl in raw_clusters:
            mems = list(cl.get("members", []))
            # Calculate importance and membership
            max_w = max([float(m.get("weight", 1.0)) for m in mems], default=1.0)
            prep_list = []
            for m in mems:
                m_name = m.get("name", "")
                c_info = concepts_by_name.get(m_name, {})
                sup = int(c_info.get("support", 1))
                imp = round(min(1.0, max(0.1, (sup / max_support) ** 0.5)), 2)
                raw_w = float(m.get("weight", 1.0))
                mem_score = round(min(1.0, max(0.2, raw_w / max(max_w, 1e-4))), 2)

                prep_list.append({
                    "name": m_name,
                    "kind": c_info.get("kind", "concept"),
                    "imp": imp,
                    "mem": mem_score,
                    "weight": raw_w,
                    "link_to_hub": m.get("link_to_hub", "indirect"),
                    "sources": m.get("sources", []),
                    "best_sentence": (c_info.get("best_sentence") or "")[:200],
                })

            # Sort members by importance descending (tie-break by weight)
            prep_list.sort(key=lambda x: (-x["imp"], -x["weight"]))
            # Cluster cap
            prep_list = prep_list[:MAX_MEMBERS_PER_CLUSTER]
            cluster_members_prep.append(prep_list)

        # Global cap MAX_MEMBERS_TOTAL: trim from largest clusters first
        total_mems = sum(len(lst) for lst in cluster_members_prep)
        while total_mems > MAX_MEMBERS_TOTAL:
            # Find cluster with max drawn members
            max_idx = max(range(n_clusters), key=lambda i: len(cluster_members_prep[i]))
            if len(cluster_members_prep[max_idx]) <= 1:
                break
            cluster_members_prep[max_idx].pop()
            total_mems -= 1

        # Calculate hub radius, halo, and ring capacities
        hub_radii: List[float] = []
        halo_radii: List[float] = []
        cluster_rings: List[List[Tuple[float, int]]] = []
        for c_idx in range(n_clusters):
            n_drawn = len(cluster_members_prep[c_idx])
            h_r, halo_r, rings = _compute_cluster_halo_and_rings(n_drawn)
            hub_radii.append(h_r)
            halo_radii.append(halo_r)
            cluster_rings.append(rings)

        # -----------------------------------------------------------------
        # 3.3 Cluster placement (Classical MDS + SPREAD + Relaxation)
        # -----------------------------------------------------------------
        raw_links = clusters_data.get("links", [])
        cluster_coords = _place_clusters_mds_and_relax(n_clusters, halo_radii, raw_links)

        # Assert minimum gap >= HALO_GAP
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                d = math.hypot(
                    cluster_coords[j, 0] - cluster_coords[i, 0],
                    cluster_coords[j, 1] - cluster_coords[i, 1],
                )
                min_allowed = halo_radii[i] + halo_radii[j] + HALO_GAP
                assert d >= min_allowed - 1.0, f"Halo overlap between cluster {i} and {j}: dist={d}, min={min_allowed}"

        # -----------------------------------------------------------------
        # Step 4: Bridge partners index
        # -----------------------------------------------------------------
        # raw_bridges: [{"concept": ..., "from": ..., "to": ..., "ratio": ...}] or similar
        bridge_map_by_name: Dict[str, Dict[str, Any]] = {}
        for b_entry in raw_bridges:
            c_name = b_entry.get("concept")
            if c_name:
                to_c = b_entry.get("to")
                # resolve to cluster index if string
                to_idx = None
                if isinstance(to_c, int):
                    to_idx = to_c
                else:
                    for idx, c in enumerate(raw_clusters):
                        if c.get("hub") == to_c:
                            to_idx = idx
                            break
                if to_idx is not None and 0 <= to_idx < n_clusters:
                    bridge_map_by_name[c_name] = {
                        "to": to_idx,
                        "ratio": round(float(b_entry.get("ratio", 0.5)), 2),
                    }

        # -----------------------------------------------------------------
        # 3.2 Member positioning in concentric rings
        # -----------------------------------------------------------------
        final_members: List[Dict[str, Any]] = []
        final_bridges: List[Dict[str, Any]] = []
        out_clusters: List[Dict[str, Any]] = []

        for c_idx in range(n_clusters):
            cl = raw_clusters[c_idx]
            cx, cy = float(cluster_coords[c_idx, 0]), float(cluster_coords[c_idx, 1])
            hub_r = hub_radii[c_idx]
            halo_r = halo_radii[c_idx]
            color = PALETTE[c_idx % len(PALETTE)]
            mems = cluster_members_prep[c_idx]
            n_drawn = len(mems)

            # Record cluster keywords (top 5 by imp/weight)
            keywords = [m["name"] for m in mems[:5]]

            out_clusters.append({
                "id": c_idx,
                "title": cl.get("hub", f"Cluster {c_idx}"),
                "x": round(cx, 1),
                "y": round(cy, 1),
                "hubR": round(hub_r, 1),
                "halo": round(halo_r, 1),
                "color": color,
                "n": cl.get("size", len(cl.get("members", []))),
                "drawn": n_drawn,
                "keywords": keywords,
                "sourceMix": cl.get("source_mix", {}),
            })

            if n_drawn == 0:
                continue

            rings = cluster_rings[c_idx]
            # Partition members into rings according to ring capacities
            # Ring 0 has capacity rings[0][1]
            ring_assignments: List[List[Dict[str, Any]]] = [[] for _ in rings]
            cur_r_idx = 0
            for rank_idx, m in enumerate(mems):
                m["rank"] = rank_idx
                while cur_r_idx < len(rings) and len(ring_assignments[cur_r_idx]) >= rings[cur_r_idx][1]:
                    cur_r_idx += 1
                if cur_r_idx < len(rings):
                    ring_assignments[cur_r_idx].append(m)
                else:
                    ring_assignments[-1].append(m)

            # Calculate member coordinates per ring
            for r_k_idx, r_mems in enumerate(ring_assignments):
                r_radius, _ = rings[r_k_idx]
                m_count = len(r_mems)
                if m_count == 0:
                    continue

                for slot_idx, m_info in enumerate(r_mems):
                    m_name = m_info["name"]
                    # Angle calculation
                    if r_k_idx == 0 and m_name in bridge_map_by_name and n_clusters > 1:
                        # Angle towards partner cluster
                        target_c = bridge_map_by_name[m_name]["to"]
                        px, py = float(cluster_coords[target_c, 0]), float(cluster_coords[target_c, 1])
                        theta = math.atan2(py - cy, px - cx)
                    else:
                        base_offset = r_k_idx * 0.35
                        theta = base_offset + slot_idx * (2.0 * math.pi / max(m_count, 1))

                    mx = cx + r_radius * math.cos(theta)
                    my = cy + r_radius * math.sin(theta)
                    m_id = f"c{c_idx}m{m_info['rank']}"
                    imp_val = m_info["imp"]
                    dot_r = round(5.0 + 9.0 * imp_val, 1)
                    direct_flag = m_info["link_to_hub"].startswith("direct")
                    multi_flag = len(m_info["sources"]) >= 2

                    # Top relations for this concept
                    top_rels = relations_by_concept.get(m_name, [])[:3]

                    final_members.append({
                        "id": m_id,
                        "c": c_idx,
                        "name": m_name,
                        "kind": m_info["kind"],
                        "x": round(mx, 1),
                        "y": round(my, 1),
                        "r": dot_r,
                        "imp": imp_val,
                        "mem": m_info["mem"],
                        "direct": direct_flag,
                        "multi": multi_flag,
                        "src": m_info["sources"],
                        "rank": m_info["rank"],
                        "sent": m_info["best_sentence"],
                        "rels": top_rels,
                    })

                    if m_name in bridge_map_by_name:
                        b_partner = bridge_map_by_name[m_name]
                        final_bridges.append({
                            "m": m_id,
                            "to": b_partner["to"],
                            "ratio": b_partner["ratio"],
                        })

        # -----------------------------------------------------------------
        # 3.4 Cluster Links (MST + median affinity)
        # -----------------------------------------------------------------
        final_links = _compute_links_mst_and_labels(n_clusters, raw_links)

        # -----------------------------------------------------------------
        # 3.5 Assembly and HTML export
        # -----------------------------------------------------------------
        # Calculate viewBox
        if n_clusters > 0:
            min_x = min(float(cluster_coords[i, 0]) - halo_radii[i] for i in range(n_clusters)) - 200.0
            max_x = max(float(cluster_coords[i, 0]) + halo_radii[i] for i in range(n_clusters)) + 200.0
            min_y = min(float(cluster_coords[i, 1]) - halo_radii[i] for i in range(n_clusters)) - 200.0
            max_y = max(float(cluster_coords[i, 1]) + halo_radii[i] for i in range(n_clusters)) + 200.0
        else:
            min_x, min_y, max_x, max_y = -500.0, -400.0, 500.0, 400.0

        raw_metrics = clusters_data.get("metrics", {})
        map_payload = {
            "meta": {
                "title": "Concept Map",
                "metrics": {
                    "clusters": int(raw_metrics.get("n_clusters", n_clusters)),
                    "concepts": int(raw_metrics.get("n_concepts_clustered", len(final_members))),
                    "modularity": float(raw_metrics.get("modularity", 0.0)),
                    "silhouette": float(raw_metrics.get("silhouette", 0.0)),
                },
                "globalTopics": clusters_data.get("global_topics", []),
                "expandPx": EXPAND_PX,
                "detailPx": DETAIL_PX,
                "viewBox": [round(min_x, 1), round(min_y, 1), round(max_x - min_x, 1), round(max_y - min_y, 1)],
            },
            "clusters": out_clusters,
            "members": final_members,
            "links": final_links,
            "bridges": final_bridges,
        }

        # JSON serialization & sanitization
        json_str = json.dumps(map_payload, ensure_ascii=False, separators=(",", ":"))
        sanitized_json = _sanitize_for_script(json_str)

        # Read template
        template_path = os.path.join(os.path.dirname(__file__), "map_template.html")
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found at {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            tmpl_content = f.read()

        html_out_content = tmpl_content.replace("/*__MAP_DATA__*/", sanitized_json)

        # Write HTML
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_out_content)

        # Write cache hash
        with open(hash_p, "w", encoding="utf-8") as f:
            f.write(current_hash)

        html_kb = round(os.path.getsize(out_html) / 1024, 1)

        return {
            "status": "ok",
            "n_clusters": n_clusters,
            "n_members_drawn": len(final_members),
            "n_links": len(final_links),
            "html_kb": html_kb,
        }
