"""
representation/concept_plot.py
Matplotlib hub-and-spoke renderer for concept communities.
Dark theme matching SynapseLocal matrix aesthetic.
"""
import math
import textwrap
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np

from representation.concept_schema import MAX_CLUSTERS, MAX_SPOKES


def render_concept_map(
    clusters: List[dict],
    links: List[dict],
    metrics: dict,
    out_png_path: str,
) -> None:
    """Render the hub-and-spoke concept map to a PNG file."""
    if not clusters:
        return

    # Sort and take top MAX_CLUSTERS by (size * total_weight)
    def cluster_score(c):
        total_w = sum(m.get("weight", 1.0) for m in c.get("members", []))
        return c.get("size", 0) * total_w

    sorted_clusters = sorted(clusters, key=cluster_score, reverse=True)[:MAX_CLUSTERS]
    active_cluster_ids = {c["id"] for c in sorted_clusters}
    cluster_by_id = {c["id"]: c for c in sorted_clusters}

    # Filter links between active clusters
    active_links = [
        lnk for lnk in links
        if lnk.get("a") in active_cluster_ids and lnk.get("b") in active_cluster_ids
    ]

    # Build quotient graph for layout
    Q = nx.Graph()
    for c in sorted_clusters:
        Q.add_node(c["id"])
    for lnk in active_links:
        Q.add_edge(lnk["a"], lnk["b"], weight=lnk.get("weight", 1.0))

    if len(sorted_clusters) > 1:
        raw_pos = nx.spring_layout(Q, weight="weight", seed=42, k=1.8 / math.sqrt(len(sorted_clusters)))
    else:
        raw_pos = {sorted_clusters[0]["id"]: np.array([0.0, 0.0])}

    # Convert to mutable pos dict and scale to working units
    pos: Dict[int, np.ndarray] = {cid: np.array(p, dtype=float) * 6.0 for cid, p in raw_pos.items()}

    # Compute geometries per cluster (hub radius, ring radius, outer disc)
    hub_radius: Dict[int, float] = {}
    ring_radius: Dict[int, float] = {}
    outer_disc: Dict[int, float] = {}
    top_spokes: Dict[int, List[dict]] = {}
    truncated_counts: Dict[int, int] = {}

    for c in sorted_clusters:
        cid = c["id"]
        members = sorted(c.get("members", []), key=lambda m: m.get("weight", 1.0), reverse=True)
        spokes = members[:MAX_SPOKES]
        top_spokes[cid] = spokes
        truncated_counts[cid] = max(0, len(members) - len(spokes))

        n_mem = c.get("size", len(members))
        hr = 0.55 + 0.12 * math.sqrt(n_mem)
        hub_radius[cid] = hr

        n_spk = len(spokes)
        rr = hr + 0.55 + 0.10 * n_spk
        ring_radius[cid] = rr
        outer_disc[cid] = rr + 0.8  # ring + max spoke radius + text buffer

    # Relaxation loop to prevent cluster overlap (<= 200 iterations)
    cluster_ids = list(sorted_clusters)
    for _ in range(200):
        moved = False
        for i in range(len(cluster_ids)):
            id_i = cluster_ids[i]["id"]
            for j in range(i + 1, len(cluster_ids)):
                id_j = cluster_ids[j]["id"]
                delta = pos[id_j] - pos[id_i]
                dist = np.linalg.norm(delta)
                min_dist = outer_disc[id_i] + outer_disc[id_j]
                if dist < min_dist:
                    moved = True
                    if dist < 1e-4:
                        delta = np.array([0.01, 0.01])
                        dist = np.linalg.norm(delta)
                    push = (min_dist - dist) * 0.5 * (delta / dist)
                    pos[id_j] += push * 0.4
                    pos[id_i] -= push * 0.4
        if not moved:
            break

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 8), dpi=110)
    fig.patch.set_facecolor("#050507")
    ax.set_facecolor("#050507")
    ax.axis("off")

    cmap = plt.colormaps.get_cmap("tab20")

    # 1. Draw hub-to-hub links
    for lnk in active_links:
        cid_a, cid_b = lnk["a"], lnk["b"]
        w = lnk.get("weight", 1.0)
        p_a = pos[cid_a]
        p_b = pos[cid_b]
        delta = p_b - p_a
        dist = np.linalg.norm(delta)
        if dist < 1e-4:
            continue
        unit = delta / dist
        r_a = hub_radius[cid_a]
        r_b = hub_radius[cid_b]

        start_pt = p_a + unit * r_a
        end_pt = p_b - unit * r_b
        lw = min(4.5, 0.8 + 0.6 * math.log(1.0 + w))

        ax.plot(
            [start_pt[0], end_pt[0]],
            [start_pt[1], end_pt[1]],
            color="#a855f7",
            alpha=0.65,
            linewidth=lw,
            zorder=2,
            linestyle="-",
        )

        # Midpoint label with relation
        top_edge = lnk.get("top_edge", {})
        rel_label = top_edge.get("relation", "")
        if rel_label and dist > (r_a + r_b + 0.8):
            mid_pt = (start_pt + end_pt) / 2.0
            ax.text(
                mid_pt[0],
                mid_pt[1],
                rel_label,
                color="#c084fc",
                fontsize=7.5,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#13131a", edgecolor="#242436", alpha=0.85),
                zorder=4,
            )

    # 2. Draw hubs and spokes
    for idx, c in enumerate(sorted_clusters):
        cid = c["id"]
        center = pos[cid]
        color = cmap(idx % 20)
        hr = hub_radius[cid]
        rr = ring_radius[cid]
        spokes = top_spokes[cid]
        n_spk = len(spokes)

        # Draw spoke links & spoke nodes
        if n_spk > 0:
            weights = [m.get("weight", 1.0) for m in spokes]
            min_w, max_w = min(weights), max(weights)

            for s_idx, spk in enumerate(spokes):
                angle = (2.0 * math.pi * s_idx) / n_spk - (math.pi / 2.0)
                spk_x = center[0] + rr * math.cos(angle)
                spk_y = center[1] + rr * math.sin(angle)

                # Spoke radius normalized [0.18, 0.30]
                norm_w = (spk.get("weight", 1.0) - min_w) / (max_w - min_w) if max_w > min_w else 0.5
                spk_r = 0.18 + 0.12 * norm_w

                # Hub to spoke thin line
                ax.plot(
                    [center[0], spk_x],
                    [center[1], spk_y],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                    zorder=3,
                )

                # Cross-source check
                sources = spk.get("sources", [])
                is_cross = len(sources) >= 2
                edge_col = "#ffd700" if is_cross else color
                edge_w = 2.0 if is_cross else 1.0

                spk_circle = patches.Circle(
                    (spk_x, spk_y),
                    spk_r,
                    facecolor="#13131a",
                    edgecolor=edge_col,
                    linewidth=edge_w,
                    zorder=5,
                )
                ax.add_patch(spk_circle)

                # Spoke label radially offset
                label_dist = spk_r + 0.22
                lx = spk_x + label_dist * math.cos(angle)
                ly = spk_y + label_dist * math.sin(angle)
                ha = "center"
                if math.cos(angle) > 0.3:
                    ha = "left"
                elif math.cos(angle) < -0.3:
                    ha = "right"

                spk_name = spk.get("name", "")
                wrapped_spk = textwrap.fill(spk_name, width=12)
                ax.text(
                    lx,
                    ly,
                    wrapped_spk,
                    color="#e5e7eb",
                    fontsize=8,
                    ha=ha,
                    va="center",
                    zorder=6,
                )

        # Draw hub circle
        hub_circle = patches.Circle(
            (center[0], center[1]),
            hr,
            facecolor=color,
            alpha=0.88,
            edgecolor="#ffffff",
            linewidth=1.5,
            zorder=7,
        )
        ax.add_patch(hub_circle)

        # Hub label
        hub_name = c.get("hub", f"Cluster {cid}")
        wrapped_hub = textwrap.fill(hub_name, width=15)
        ax.text(
            center[0],
            center[1],
            wrapped_hub,
            color="#ffffff",
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )

        # Truncation annotation if members were truncated
        trunc = truncated_counts[cid]
        if trunc > 0:
            ax.text(
                center[0],
                center[1] - hr - 0.22,
                f"+{trunc} more",
                color="#9ca3af",
                fontsize=7.5,
                fontstyle="italic",
                ha="center",
                va="top",
                zorder=8,
            )

    # Rescale plot limits with margins
    all_x = [pos[c["id"]][0] for c in sorted_clusters]
    all_y = [pos[c["id"]][1] for c in sorted_clusters]
    max_d = max(outer_disc.values()) if outer_disc else 2.0

    min_x = min(all_x) - max_d - 0.5
    max_x = max(all_x) + max_d + 0.5
    min_y = min(all_y) - max_d - 0.5
    max_y = max(all_y) + max_d + 0.5

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")

    # Title
    n_cl = metrics.get("n_clusters", len(sorted_clusters))
    n_cp = metrics.get("n_concepts_clustered", 0)
    mod = metrics.get("modularity", 0.0)
    ax.set_title(
        f"Concept Map (clusters: {n_cl} | concepts: {n_cp} | modularity: {mod:.2f})",
        color="#f3f4f6",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=110, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
