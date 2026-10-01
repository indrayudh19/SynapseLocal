import streamlit as st
import pandas as pd
import numpy as np
import os

from backend.session_manager import (
    create_session, delete_session, list_sessions,
    save_uploaded_files, get_session_dir,
)
from backend.pipeline import hybrid_retrieve_and_answer
from backend.vector_store import LocalVectorStore
from backend.local_llm import get_embedding_dim
from representation import paths as rep_paths
from representation import embed_store
from representation import graph as rep_graph
from representation import cluster as rep_cluster
from representation import extract_concepts as rep_extract
from representation import build_concept_graph as rep_graph_build

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SynapseLocal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling: Matrix Dark (Black, Neon Green, Purple)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;600&display=swap');

:root {
    --bg-primary: #050507;
    --bg-secondary: #0d0d12;
    --bg-card: #13131a;
    --border-color: #242436;
    --accent-green: #00ff66;
    --accent-purple: #a855f7;
    --accent-purple-light: #c084fc;
    --text-primary: #f3f4f6;
    --text-secondary: #9ca3af;
    --text-green: #38ef7d;
    --text-purple: #c084fc;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.02em;
}

h1 { color: var(--accent-green) !important; font-weight: 700; }
h2, h3 { color: var(--accent-purple) !important; font-weight: 600; }
p, span, label, div { color: var(--text-primary); }

.badge-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    padding: 2px 8px;
    border-radius: 4px;
    background: #181824;
    border: 1px solid var(--border-color);
}
.badge-green { color: var(--accent-green); border-color: rgba(0, 255, 102, 0.3); }
.badge-purple { color: var(--accent-purple-light); border-color: rgba(168, 85, 247, 0.3); }

.stButton > button {
    background-color: #121124 !important;
    color: var(--accent-green) !important;
    border: 1px solid var(--accent-green) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    transition: all 0.2s ease-in-out !important;
    width: 100%;
}
.stButton > button:hover {
    background-color: var(--accent-green) !important;
    color: #050507 !important;
    box-shadow: 0 0 12px rgba(0, 255, 102, 0.4) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px; border-bottom: 1px solid var(--border-color); background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important; font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-bottom: none !important;
    border-radius: 4px 4px 0 0 !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-purple-light) !important;
    border-color: var(--accent-purple) !important;
    border-bottom: 2px solid var(--accent-purple) !important;
    background-color: #1a162b !important;
}

[data-testid="stChatMessage"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--accent-purple) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--accent-green) !important;
}

[data-testid="stChatInput"] textarea {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 8px rgba(168, 85, 247, 0.3) !important;
}

.streamlit-expanderHeader {
    background-color: #12121a !important;
    color: var(--accent-purple-light) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
}
.streamlit-expanderContent {
    background-color: #0b0b10 !important;
    border: 1px solid var(--border-color) !important;
    border-top: none !important;
}

[data-testid="stFileUploader"] section {
    background-color: #0e0e16 !important;
    border: 1px dashed var(--border-color) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent-green) !important;
}

div[data-baseweb="select"] > div {
    background-color: #12121a !important;
    border-color: var(--border-color) !important;
    color: var(--text-primary) !important;
}

.system-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    padding: 16px; border-radius: 6px; margin-bottom: 12px;
}
.system-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;
}
.system-card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem; color: var(--accent-green); font-weight: 700;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_stats" not in st.session_state:
    st.session_state.pipeline_stats = None
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False


# ---------------------------------------------------------
# Sidebar: Ingestion & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### SYNAPSE LOCAL")
    st.caption("Private Knowledge Graph & Hybrid RAG Engine")
    st.markdown("---")

    # ── Session Management ────────────────────────────────
    st.markdown("#### Session Management")
    
    sessions = list_sessions()
    session_options = {sid: meta["name"] for sid, meta in sessions.items()}
    
    col_new, col_del = st.columns(2)
    with col_new:
        if st.button("New Session"):
            new_id = create_session()
            st.session_state.session_id = new_id
            st.session_state.messages = []
            st.session_state.pipeline_stats = None
            st.session_state.is_processed = False
            st.rerun()
    
    with col_del:
        if st.button("Delete Session"):
            if st.session_state.session_id:
                delete_session(st.session_state.session_id)
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.pipeline_stats = None
                st.session_state.is_processed = False
                st.rerun()

    if session_options:
        selected_sid = st.selectbox(
            "Active Session",
            options=list(session_options.keys()),
            format_func=lambda x: f"{session_options[x]} [{x}]",
            index=(
                list(session_options.keys()).index(st.session_state.session_id)
                if st.session_state.session_id in session_options
                else 0
            )
        )
        if selected_sid != st.session_state.session_id:
            st.session_state.session_id = selected_sid
            st.session_state.messages = []
            # Check if this session has been processed
            meta = sessions.get(selected_sid, {})
            st.session_state.is_processed = meta.get("is_processed", False)
            st.session_state.pipeline_stats = None
            st.rerun()
    else:
        st.info("No sessions. Click 'New Session' to start.")
    
    st.markdown("---")

    # ── Document Ingestion ────────────────────────────────
    st.markdown("#### Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md", "pptx"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, Markdown, or PPTX files."
    )

    if uploaded_files:
        st.markdown(
            f"<div class='badge-mono badge-green'>{len(uploaded_files)} file(s) staged</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### Local Inference Model")
    selected_model = st.selectbox(
        "Select Qwen Model",
        options=["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5-coder:7b"],
        index=0,
        help="Select the local model running via Ollama."
    )
    
    embed_model = st.selectbox(
        "Embedding Model",
        options=["nomic-embed-text", "all-minilm (sentence-transformers fallback)"],
        index=0,
        help="Embedding model for vector indexing."
    )
    embed_model_name = "nomic-embed-text" if "nomic" in embed_model else "all-MiniLM-L6-v2"

    st.markdown("---")
    st.markdown("#### Services Pipeline")
    
    # Stage 1: Embed + Store
    sid = st.session_state.session_id
    has_stage1 = bool(
        sid
        and os.path.exists(rep_paths.embeddings_path(sid))
        and os.path.exists(rep_paths.chunks_jsonl_path(sid))
    )
    
    stage1_label = "Stage 1: Re-embed & Store" if has_stage1 else "Stage 1: Embed & Store"
    embed_btn = st.button(stage1_label, help="Run Stage 1: Parse documents, chunk, embed, and store in FAISS")

    if embed_btn:
        if not sid:
            st.error("Create or select a session first.")
        else:
            if uploaded_files:
                save_uploaded_files(sid, uploaded_files)
            
            with st.spinner("Stage 1: Embedding and storing chunks..."):
                res = embed_store.run(sid)
                if res.get("status") == "error":
                    st.error(res.get("message", "Stage 1 execution failed."))
                else:
                    st.session_state.is_processed = True
                    cached_tag = " (from cache)" if res.get("cached") else ""
                    st.success(
                        f"Stage 1 Complete{cached_tag}! {res['chunks_count']} chunks indexed (dim={res['embedding_dim']})."
                    )
                    st.rerun()

    # Sequential run in order button (embed_store -> graph -> cluster)
    run_all_btn = st.button(
        "Run Pipeline (1 -> 2 -> 3)",
        help="Executes embed_store, then graph, then cluster, strictly one at a time to full completion."
    )
    if run_all_btn:
        if not sid:
            st.error("Create or select a session first.")
        else:
            if uploaded_files:
                save_uploaded_files(sid, uploaded_files)
            
            with st.spinner("Stage 1/3: Running embed_store..."):
                r1 = embed_store.run(sid)
            if r1.get("status") == "error":
                st.error(f"Stage 1 failed: {r1.get('message')}")
            else:
                st.session_state.is_processed = True
                with st.spinner("Stage 2/3: Running knowledge graph..."):
                    r2 = rep_graph.run(sid)
                if r2.get("status") == "error":
                    st.error(f"Stage 2 failed: {r2.get('message')}")
                else:
                    with st.spinner("Stage 3/3: Running clustering..."):
                        r3 = rep_cluster.run(sid)
                    if r3.get("status") == "error":
                        st.error(f"Stage 3 failed: {r3.get('message')}")
                    else:
                        st.success("All 3 stages executed to completion in strict sequence!")
                        st.rerun()

    st.markdown("---")
    st.markdown("#### System Status")
    
    stage1_status = "Built" if has_stage1 else "Not built"
    stage1_color = "#00ff66" if has_stage1 else "#9ca3af"

    has_graph = bool(
        sid
        and os.path.exists(rep_paths.graph_png_path(sid))
        and os.path.exists(rep_paths.graph_meta_path(sid))
    )
    graph_status = "Built" if has_graph else "Not built"
    graph_color = "#00ff66" if has_graph else "#9ca3af"

    has_cluster = bool(
        sid
        and os.path.exists(rep_paths.cluster_map_png_path(sid))
        and os.path.exists(rep_paths.clusters_json_path(sid))
    )
    cluster_status = "Built" if has_cluster else "Not built"
    cluster_color = "#00ff66" if has_cluster else "#9ca3af"
    
    st.markdown(
        f"""
        <div class="system-card">
            <div class="system-card-title">Storage Status</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #f3f4f6; margin-top: 6px; line-height: 1.6;">
                <div>Stage 1 (Embed + Store): <span style="color:{stage1_color}; font-weight:600;">{stage1_status}</span></div>
                <div>Stage 2 (Knowledge Graph): <span style="color:{graph_color}; font-weight:600;">{graph_status}</span></div>
                <div>Stage 3 (Cluster Map): <span style="color:{cluster_color}; font-weight:600;">{cluster_status}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Header Area
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<h1 style='margin-bottom:0;'>SYNAPSE LOCAL</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#a855f7; font-family: \"JetBrains Mono\", monospace; font-size:0.9rem;'>"
        "TOPOLOGICAL HYBRID RAG ARCHITECTURE</p>",
        unsafe_allow_html=True
    )

with col_h2:
    sid_label = st.session_state.session_id or "none"
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 8px;">
            <span class="badge-mono badge-purple">{selected_model}</span>
            <span class="badge-mono badge-green">SESSION: {sid_label}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")


# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab_chat, tab_graph, tab_cluster = st.tabs([
    "Local RAG Chat",
    "Interactive Knowledge Graph",
    "Concept Clusters"
])


# ---------------------------------------------------------
# Tab 1: Local RAG Chat
# ---------------------------------------------------------
with tab_chat:
    if not st.session_state.session_id:
        st.info("Create a session from the sidebar to begin.")
    elif not st.session_state.is_processed:
        st.info("Upload documents and run the pipeline to enable chat.")
        # Still show chat input (disabled feel) for UI completeness
        if not st.session_state.messages:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "SynapseLocal engine standing by. Upload documents and process them to activate hybrid retrieval.",
                "citations": []
            }]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("citations"):
                with st.expander("Source Citations & Graph Nodes", expanded=False):
                    for idx, cit in enumerate(message["citations"]):
                        source_type = cit.get("source", "vector")
                        score_text = f" | Score: {cit['score']}" if cit.get("score") else ""
                        st.markdown(
                            f"""
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
                                        padding: 6px 0; border-bottom: 1px solid #1f1f2e;">
                                <span style="color: #00ff66;">[{idx+1}] {cit.get('doc', 'unknown')}</span>
                                &nbsp;|&nbsp;
                                <span style="color: #a855f7;">{cit.get('chunk_id', '')}</span>
                                &nbsp;|&nbsp;
                                <span style="color: #c084fc;">{cit.get('heading', '')}</span>
                                &nbsp;|&nbsp;
                                <span style="color: #9ca3af;">[{source_type}]{score_text}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Chat input
    user_query = st.chat_input("Enter your query for local retrieval...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query, "citations": []})
        with st.chat_message("user"):
            st.write(user_query)

        if st.session_state.is_processed and st.session_state.session_id:
            # Real hybrid retrieval
            with st.spinner("Retrieving context and generating response..."):
                try:
                    result = hybrid_retrieve_and_answer(
                        session_id=st.session_state.session_id,
                        query=user_query,
                        model=selected_model,
                        embed_model=embed_model_name,
                    )
                    answer = result["answer"]
                    citations = result["citations"]
                except Exception as e:
                    answer = f"Error during retrieval: {str(e)}"
                    citations = []
        else:
            answer = "Pipeline not initialized. Upload and process documents first."
            citations = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations
        })
        with st.chat_message("assistant"):
            st.write(answer)
            if citations:
                with st.expander("Source Citations & Graph Nodes", expanded=False):
                    for idx, cit in enumerate(citations):
                        source_type = cit.get("source", "vector")
                        score_text = f" | Score: {cit['score']}" if cit.get("score") else ""
                        st.markdown(
                            f"""
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
                                        padding: 6px 0; border-bottom: 1px solid #1f1f2e;">
                                <span style="color: #00ff66;">[{idx+1}] {cit.get('doc', 'unknown')}</span>
                                &nbsp;|&nbsp;
                                <span style="color: #a855f7;">{cit.get('chunk_id', '')}</span>
                                &nbsp;|&nbsp;
                                <span style="color: #c084fc;">{cit.get('heading', '')}</span>
                                &nbsp;|&nbsp;
                                <span style="color: #9ca3af;">[{source_type}]{score_text}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


# ---------------------------------------------------------
# Tab 2: Knowledge Graph
# ---------------------------------------------------------
with tab_graph:
    st.markdown("### Semantic Knowledge Graph")
    st.caption("kNN semantic network with Louvain communities and PageRank centrality.")

    sid = st.session_state.session_id
    if not sid:
        st.info("Create or select a session to view the knowledge graph.")
    else:
        has_stage1 = (
            os.path.exists(rep_paths.embeddings_path(sid))
            and os.path.exists(rep_paths.chunks_jsonl_path(sid))
        )
        graph_png = rep_paths.graph_png_path(sid)
        graph_meta = rep_paths.graph_meta_path(sid)
        graph_built = os.path.exists(graph_png) and os.path.exists(graph_meta)

        status_txt = "Built" if graph_built else "Not built"
        status_col = "#00ff66" if graph_built else "#9ca3af"

        col_gb1, col_gb2 = st.columns([3, 1])
        with col_gb1:
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.85rem;margin-bottom:12px;'>"
                f"Graph Status: <span style='color:{status_col};font-weight:600;'>{status_txt}</span></div>",
                unsafe_allow_html=True
            )

        btn_cols = st.columns([2, 1, 3])
        with btn_cols[0]:
            build_label = "Rebuild Graph" if graph_built else "Build Knowledge Graph"
            build_graph_btn = st.button(
                build_label,
                disabled=not has_stage1,
                help="Requires Stage 1 artifacts" if not has_stage1 else "Construct and render semantic knowledge graph"
            )

        if not has_stage1:
            st.warning("Stage 1 artifacts missing. Run 'Stage 1: Embed & Store' first.")

        if build_graph_btn:
            if graph_built and os.path.exists(rep_paths.graph_hash_path(sid)):
                try:
                    os.remove(rep_paths.graph_hash_path(sid))
                except OSError:
                    pass
            with st.spinner("Building Knowledge Graph (kNN + Louvain + PageRank)..."):
                res = rep_graph.run(sid)
                if res.get("status") == "error":
                    st.error(res.get("message", "Graph construction failed."))
                else:
                    cached_tag = " (from cache)" if res.get("cached") else ""
                    st.success(f"Knowledge Graph ready{cached_tag}!")
                    st.rerun()

        # Display saved artifacts if available
        if graph_built:
            try:
                import json
                with open(graph_meta, "r", encoding="utf-8") as mf:
                    meta_data = json.load(mf)

                g_stats = meta_data.get("global", {})
                st.markdown(
                    f"""
                    <div style="display:flex; flex-wrap:wrap; gap:12px; margin: 12px 0 16px 0; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
                        <span class="badge-mono badge-green">Nodes: {g_stats.get('nodes', 0)}</span>
                        <span class="badge-mono badge-purple">Edges: {g_stats.get('edges', 0)}</span>
                        <span class="badge-mono badge-purple">Communities: {g_stats.get('communities', 0)}</span>
                        <span class="badge-mono">Density: {g_stats.get('density', 0):.4f}</span>
                        <span class="badge-mono">Threshold Tau: {g_stats.get('threshold_tau', 0):.4f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.image(graph_png, use_container_width=True)

                col_t1, col_t2 = st.columns([3, 2])
                with col_t1:
                    st.markdown("#### Top Nodes by PageRank")
                    nodes_list = meta_data.get("nodes", [])
                    sorted_nodes = sorted(nodes_list, key=lambda x: x.get("pagerank", 0.0), reverse=True)[:10]
                    if sorted_nodes:
                        df_nodes = pd.DataFrame([{
                            "ID": n.get("id"),
                            "Snippet": n.get("text_snippet", ""),
                            "Source": n.get("source_file"),
                            "Type": n.get("source_type"),
                            "PageRank": f"{n.get('pagerank', 0.0):.5f}",
                            "Cross-Ratio": f"{n.get('cross_source_ratio', 0.0):.2f}",
                        } for n in sorted_nodes])
                        st.dataframe(df_nodes, use_container_width=True, hide_index=True)

                with col_t2:
                    st.markdown("#### Louvain Communities")
                    comms_list = meta_data.get("communities", [])
                    if comms_list:
                        df_comm = pd.DataFrame([{
                            "Community": f"#{c.get('community')}",
                            "Top Terms": c.get("label"),
                            "Size": c.get("size"),
                            "Sources": ", ".join(c.get("sources", [])),
                        } for c in comms_list])
                        st.dataframe(df_comm, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error displaying graph artifacts: {str(e)}")


# ---------------------------------------------------------
# Tab 3: Semantic Cluster Map
# ---------------------------------------------------------
with tab_cluster:
    st.markdown("### Concept Clusters")
    st.caption("LLM-powered concept extraction (Qwen 2.5 3B), graph consolidation, and hub-and-spoke community mapping.")

    sid = st.session_state.session_id
    if not sid:
        st.info("Create or select a session to view concept clusters.")
    else:
        chunks_file = rep_paths.chunks_jsonl_path(sid)
        has_chunks = os.path.exists(chunks_file)

        raw_jsonl = rep_paths.concepts_raw_path(sid)
        extract_log = rep_paths.extract_log_path(sid)
        has_stage1 = os.path.exists(raw_jsonl)

        concept_graph = rep_paths.concept_graph_path(sid)
        has_stage2 = os.path.exists(concept_graph)

        clusters_json = rep_paths.concept_clusters_path(sid)
        clusters_png = rep_paths.concept_map_path(sid)
        has_stage3 = os.path.exists(clusters_json)

        # Count total eligible parent chunks for default sample
        total_p_chunks = 0
        if has_chunks:
            try:
                import json
                seen_p = set()
                with open(chunks_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            c = json.loads(line)
                            seen_p.add(c.get("parent_id") or c.get("id"))
                total_p_chunks = len(seen_p)
            except Exception:
                total_p_chunks = 20

        # Status row
        s1_color = "#00ff66" if has_stage1 else "#9ca3af"
        s2_color = "#00ff66" if has_stage2 else "#9ca3af"
        s3_color = "#00ff66" if has_stage3 else "#9ca3af"

        st.markdown(
            f"""
            <div style='display:flex; gap:16px; font-family:JetBrains Mono,monospace; font-size:0.85rem; margin-bottom:14px;'>
                <span>Stage 1 (Extracted): <b style='color:{s1_color};'>{'Ready' if has_stage1 else 'Pending'}</b></span>
                <span>Stage 2 (Consolidated): <b style='color:{s2_color};'>{'Ready' if has_stage2 else 'Pending'}</b></span>
                <span>Stage 3 (Clustered): <b style='color:{s3_color};'>{'Ready' if has_stage3 else 'Pending'}</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Execution Pipeline")
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown("**Step 1: Extract Concepts (Qwen)**")
            def_sample = max(5, total_p_chunks) if total_p_chunks > 0 else 10
            sample_val = st.number_input(
                "Sample size (chunks)",
                min_value=5,
                max_value=max(5, total_p_chunks) if total_p_chunks > 0 else 1000,
                value=def_sample,
                step=5,
                key="concept_sample_size",
                help="Process first N eligible parent chunks (resumes if already partially run)."
            )

            # Read extract log if present to show estimated time
            avg_sec = 0.0
            if os.path.exists(extract_log):
                try:
                    import json
                    with open(extract_log, "r", encoding="utf-8") as lf:
                        avg_sec = json.load(lf).get("avg_seconds_per_chunk", 0.0)
                except Exception:
                    pass

            if avg_sec > 0:
                est_time = int(sample_val * avg_sec)
                st.caption(f"Estimated time: ~{est_time}s ({sample_val} chunks × {avg_sec:.1f}s/chunk)")

            b_col1, b_col2 = st.columns([2, 1])
            with b_col1:
                extract_btn = st.button(
                    "1. Extract concepts (Qwen)",
                    disabled=not has_chunks,
                    help="Extract concepts & relations sequentially with Qwen 2.5 3B"
                )
            with b_col2:
                reset_btn = st.button("Reset extraction", help="Delete concepts_raw.jsonl to start fresh")

            if reset_btn:
                for p in [rep_paths.concepts_raw_path(sid), rep_paths.extract_log_path(sid), rep_paths.extract_hash_path(sid)]:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                st.success("Extraction cache cleared.")
                st.rerun()

            if extract_btn:
                if os.path.exists(rep_paths.extract_hash_path(sid)):
                    try:
                        os.remove(rep_paths.extract_hash_path(sid))
                    except OSError:
                        pass
                progress_bar = st.progress(0, text="Starting Qwen concept extraction...")

                def update_progress(done: int, total: int):
                    frac = min(1.0, done / max(1, total))
                    progress_bar.progress(frac, text=f"Extracted {done}/{total} chunks...")

                with st.spinner("Extracting concepts with Qwen 2.5 3B..."):
                    res = rep_extract.run(sid, max_chunks=sample_val, on_progress=update_progress)
                    if res.get("status") == "error":
                        st.error(res.get("message", "Extraction failed."))
                    else:
                        st.success(f"Extraction complete! {res.get('concepts_kept', 0)} concepts kept.")
                        st.rerun()

        with col_c2:
            st.markdown("**Step 2: Consolidate Graph**")
            st.caption("Normalize terms, resolve acronyms, merge fuzzy keys & filter global topics.")
            consolidate_btn = st.button(
                "2. Consolidate",
                disabled=not has_stage1,
                help="Requires Stage 1 concepts_raw.jsonl"
            )
            if consolidate_btn:
                if os.path.exists(rep_paths.concept_graph_hash_path(sid)):
                    try:
                        os.remove(rep_paths.concept_graph_hash_path(sid))
                    except OSError:
                        pass
                with st.spinner("Consolidating concepts & typing relations..."):
                    res = rep_graph_build.run(sid)
                    if res.get("status") == "error":
                        st.error(res.get("message", "Consolidation failed."))
                    else:
                        st.success(f"Graph consolidated! ({res.get('n_concepts', 0)} concepts, {res.get('n_relations', 0)} relations)")
                        st.rerun()

        with col_c3:
            st.markdown("**Step 3: Cluster & Render**")
            st.caption("Community detection (Louvain), hub discovery, and hub-and-spoke map.")
            cluster_btn = st.button(
                "3. Build concept map",
                disabled=not has_stage2,
                help="Requires Stage 2 concept_graph.json"
            )
            if cluster_btn:
                if os.path.exists(rep_paths.concept_clusters_hash_path(sid)):
                    try:
                        os.remove(rep_paths.concept_clusters_hash_path(sid))
                    except OSError:
                        pass
                with st.spinner("Clustering concept communities and rendering map..."):
                    res = rep_cluster.run(sid)
                    if res.get("status") == "error":
                        st.error(res.get("message", "Clustering failed."))
                    elif res.get("status") == "insufficient_structure":
                        st.warning(res.get("message", "Insufficient structure detected."))
                    else:
                        st.success("Concept map rendered!")
                        st.rerun()

        st.markdown("---")

        # Display Stage 3 results if available
        if has_stage3:
            try:
                import json
                with open(clusters_json, "r", encoding="utf-8") as jf:
                    cluster_data = json.load(jf)

                if cluster_data.get("status") == "insufficient_structure":
                    st.warning(
                        cluster_data.get(
                            "message",
                            "Insufficient structure detected to generate a reliable concept map. "
                            "Please extract more chunks or add richer source documents."
                        )
                    )
                else:
                    metrics = cluster_data.get("metrics", {})
                    st.markdown(
                        f"""
                        <div style="display:flex; flex-wrap:wrap; gap:12px; margin: 8px 0 16px 0; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
                            <span class="badge-mono badge-green">Concepts: {metrics.get('n_concepts_clustered', 0)}</span>
                            <span class="badge-mono badge-purple">Clusters: {metrics.get('n_clusters', 0)}</span>
                            <span class="badge-mono badge-purple">Modularity: {metrics.get('modularity', 0.0):.3f}</span>
                            <span class="badge-mono badge-green">Cross-Source Concepts: {metrics.get('cross_source_concepts', 0)}</span>
                            <span class="badge-mono">Hub Coverage: {metrics.get('hub_coverage', 0.0)*100:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if os.path.exists(clusters_png):
                        st.image(clusters_png, use_container_width=True)

                    # Cluster breakdown table
                    st.markdown("#### Concept Clusters Breakdown")
                    clusters_list = cluster_data.get("clusters", [])
                    if clusters_list:
                        df_rows = []
                        for c in clusters_list:
                            members_preview = ", ".join([
                                f"{m.get('name', '')} [{m.get('link_to_hub', '')}]"
                                for m in c.get("members", [])
                            ])
                            mix_str = ", ".join([f"{sf}: {int(frac*100)}%" for sf, frac in c.get("source_mix", {}).items()])
                            df_rows.append({
                                "Cluster ID": f"#{c.get('id')}",
                                "Hub Concept": c.get("hub"),
                                "Size": c.get("size"),
                                "Members": members_preview,
                                "Source Mix": mix_str,
                            })
                        st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

                    # Expander for global topics, orphans, extraction log
                    with st.expander("Global Topics, Orphans & Extraction Log", expanded=False):
                        c_ex1, c_ex2 = st.columns(2)
                        with c_ex1:
                            st.markdown("##### Global Hub Topics (Frequency > 40%)")
                            gt = cluster_data.get("global_topics", [])
                            if gt:
                                st.write(", ".join(gt))
                            else:
                                st.caption("None detected")
                        with c_ex2:
                            st.markdown("##### Orphan Concepts")
                            orph = cluster_data.get("orphans", [])
                            if orph:
                                st.write(", ".join(orph))
                            else:
                                st.caption("None")

                        if os.path.exists(extract_log):
                            st.markdown("##### Extraction Stats")
                            try:
                                with open(extract_log, "r", encoding="utf-8") as elf:
                                    log_stats = json.load(elf)
                                st.json(log_stats)
                            except Exception:
                                pass
            except Exception as e:
                st.error(f"Error displaying concept clusters: {str(e)}")
        else:
            st.info("Concept map not built yet. Run the 3 steps above to generate the concept map.")

