"""
backend/graph_engine.py
Module 1.4: Topological Knowledge Graph & Centrality Algorithms

Constructs a mathematical concept graph from extracted entities/chunks
and computes structural importance metrics locally via NetworkX.

Algorithms:
  - Graph Construction: Nodes = concepts/headings, Edges = co-occurrence / shared chunks
  - PageRank Centrality: Iterative importance scoring for node sizing
  - Louvain Community Detection: Partition nodes into discrete clusters
"""
import os
import re
import pickle
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions")


class KnowledgeGraph:
    """
    Builds and maintains a topological knowledge graph from document chunks.
    
    Nodes represent extracted key concepts (headings, prominent noun phrases).
    Edges represent co-occurrence within the same semantic chunk.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = os.path.join(DATA_ROOT, session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        self._graph_path = os.path.join(self.session_dir, "knowledge_graph.pkl")
        self._entities_path = os.path.join(self.session_dir, "entity_map.pkl")
        
        self.graph: nx.Graph = nx.Graph()
        self.entity_to_chunks: Dict[str, Set[str]] = {}  # entity -> set of chunk_ids
        self.chunk_to_entities: Dict[str, Set[str]] = {}  # chunk_id -> set of entities
        
        self._load()
    
    def _load(self):
        """Load persisted graph state."""
        if os.path.exists(self._graph_path):
            with open(self._graph_path, "rb") as f:
                self.graph = pickle.load(f)
        if os.path.exists(self._entities_path):
            with open(self._entities_path, "rb") as f:
                data = pickle.load(f)
                self.entity_to_chunks = data.get("entity_to_chunks", {})
                self.chunk_to_entities = data.get("chunk_to_entities", {})
    
    def _save(self):
        """Persist graph state to disk."""
        with open(self._graph_path, "wb") as f:
            pickle.dump(self.graph, f)
        with open(self._entities_path, "wb") as f:
            pickle.dump({
                "entity_to_chunks": self.entity_to_chunks,
                "chunk_to_entities": self.chunk_to_entities
            }, f)
    
    def extract_entities_simple(self, text: str) -> List[str]:
        """
        Lightweight entity extraction using regex-based noun phrase heuristics.
        Extracts capitalized multi-word phrases and technical terms.
        """
        entities = set()
        
        # Extract capitalized phrases (2+ words starting with uppercase)
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(cap_pattern, text):
            entities.add(match.group(1).strip())
        
        # Extract technical terms: words with mixed case, acronyms, or containing digits
        tech_pattern = r'\b([A-Z]{2,}[a-z]*(?:\s*[A-Z]{2,}[a-z]*)*)\b'
        for match in re.finditer(tech_pattern, text):
            term = match.group(1).strip()
            if len(term) >= 2:
                entities.add(term)
        
        # Extract terms with common technical suffixes/patterns
        suffix_pattern = r'\b(\w+(?:ing|tion|ment|ity|ness|ance|ence|ism|ist|ics|ology|graphy))\b'
        for match in re.finditer(suffix_pattern, text, re.IGNORECASE):
            term = match.group(1).strip()
            if len(term) >= 5:
                entities.add(term.lower())
        
        return list(entities)
    
    def extract_entities_llm(self, text: str, ollama_fn=None) -> List[str]:
        """
        Use the local LLM (Qwen via Ollama) to extract key entities/concepts.
        Falls back to simple extraction if LLM is unavailable.
        """
        if ollama_fn is None:
            return self.extract_entities_simple(text)
        
        prompt = (
            "Extract the key concepts, entities, and technical terms from the following text. "
            "Return ONLY a comma-separated list of terms, nothing else.\n\n"
            f"Text: {text[:1500]}\n\n"
            "Entities:"
        )
        
        try:
            response = ollama_fn(prompt)
            raw = response.strip()
            entities = [e.strip() for e in raw.split(",") if e.strip() and len(e.strip()) >= 2]
            return entities if entities else self.extract_entities_simple(text)
        except Exception:
            return self.extract_entities_simple(text)
    
    def build_from_chunks(self, chunks: list, ollama_fn=None):
        """
        Build the knowledge graph from semantic chunks.
        
        1. Extract entities from each chunk
        2. Create nodes for each entity
        3. Create edges between entities that co-occur in the same chunk
        4. Edge weight = number of chunks where both entities co-occur
        """
        for chunk in chunks:
            entities = self.extract_entities_llm(chunk.text, ollama_fn)
            
            # Also add the chunk heading as an entity/node
            if chunk.heading and chunk.heading not in ("Section 1", "Page 1"):
                entities.append(chunk.heading)
            
            # Deduplicate within this chunk
            entities = list(set(entities))
            
            # Update mappings
            self.chunk_to_entities[chunk.chunk_id] = set(entities)
            for entity in entities:
                if entity not in self.entity_to_chunks:
                    self.entity_to_chunks[entity] = set()
                self.entity_to_chunks[entity].add(chunk.chunk_id)
            
            # Add nodes
            for entity in entities:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, frequency=0, source_files=set())
                self.graph.nodes[entity]["frequency"] += 1
                self.graph.nodes[entity]["source_files"].add(chunk.source_file)
            
            # Add edges for co-occurring entities within this chunk
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1, e2 = entities[i], entities[j]
                    if self.graph.has_edge(e1, e2):
                        self.graph[e1][e2]["weight"] += 1
                    else:
                        self.graph.add_edge(e1, e2, weight=1)
        
        self._save()
    
    def compute_pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """Compute PageRank centrality for all nodes."""
        if len(self.graph.nodes) == 0:
            return {}
        return nx.pagerank(self.graph, alpha=alpha)
    
    def compute_louvain_communities(self) -> Dict[str, int]:
        """
        Compute Louvain community partition.
        Returns mapping of node -> community_id.
        """
        if len(self.graph.nodes) == 0:
            return {}
        try:
            import community.community_louvain as community_louvain
            return community_louvain.best_partition(self.graph)
        except ImportError:
            # Fallback: use NetworkX greedy modularity
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(self.graph)
            partition = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    partition[node] = idx
            return partition
    
    def get_graph_stats(self) -> Dict:
        """Return summary statistics about the graph."""
        communities = self.compute_louvain_communities()
        n_communities = len(set(communities.values())) if communities else 0
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": n_communities,
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
    
    def get_neighbors(self, entity: str, depth: int = 2) -> Set[str]:
        """Get k-hop neighborhood of an entity for graph-augmented retrieval."""
        if entity not in self.graph:
            return set()
        
        visited = {entity}
        frontier = {entity}
        
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        
        return visited
    
    def get_chunks_for_entities(self, entities: Set[str]) -> Set[str]:
        """Get all chunk IDs associated with a set of entities."""
        chunk_ids = set()
        for entity in entities:
            chunk_ids.update(self.entity_to_chunks.get(entity, set()))
        return chunk_ids
