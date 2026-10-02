"""
representation/concept_schema.py
Constants and Pydantic models for concept extraction and graph clustering.
Single source of truth for Pass 3.
"""
from typing import Literal, List
from pydantic import BaseModel, Field

EXTRACT_MODEL = "qwen2.5:3b"
PROMPT_VERSION = "v2"                     # bump: forces one fresh extraction
MAX_CHARS = 1200          # chunk text sent to the LLM
MIN_CHARS = 200           # skip shorter chunks
MAX_CONCEPTS = 8
MAX_RELATIONS = 8
GLOBAL_FREQ = 0.50                        # concept in >50% of processed chunks = global topic
MIN_CLUSTER_SIZE = 3
MAX_CLUSTERS = 12         # drawn
MAX_SPOKES = 8            # drawn per hub
HIERARCHY_RELS = ("type_of", "part_of", "example_of")
REL_WEIGHT = {
    "type_of": 3.0,
    "part_of": 3.0,
    "example_of": 3.0,
    "uses": 1.5,
    "related_to": 1.0,
}

KINDS     = ("category","component","practice","attribute","example")
HUB_KINDS = ("category","component","practice")
SUFFIXES  = ("architecture","system","pattern","approach","framework","technique","methodology","concept")

# Stage 2 merge
MERGE_AUTO_SIM   = 0.88
MERGE_JUDGE_BAND = (0.78, 0.88)
MERGE_JUDGE_MAX  = 30
LLM_MERGE_JUDGE  = True

# Stage 3 affinity / clustering
AFFINITY_W  = {"relation": 0.5, "cooccur": 0.3, "semantic": 0.2}
SEM_MIN_SIM, SEM_TOPK = 0.55, 3
EDGE_MIN, NODE_TOPK   = 0.15, 5
RESOLUTIONS = [0.8, 1.0, 1.3, 1.7]
BRIDGE_RATIO, MAX_BRIDGES = 0.5, 6


class Concept(BaseModel):
    name: str
    kind: Literal["category","component","practice","attribute","example"]


class Relation(BaseModel):
    source: str
    relation: Literal["type_of", "part_of", "example_of", "uses", "related_to"]
    target: str


class Extraction(BaseModel):
    main_topic: str = ""                             # the single most central concept of this chunk
    concepts: List[Concept] = Field(default_factory=list, max_length=MAX_CONCEPTS)
    relations: List[Relation] = Field(default_factory=list, max_length=MAX_RELATIONS)
