"""
representation/concept_schema.py
Constants and Pydantic models for concept extraction and graph clustering.
Single source of truth for Pass 3.
"""
from typing import Literal, List
from pydantic import BaseModel, Field

EXTRACT_MODEL = "qwen2.5:3b"
PROMPT_VERSION = "v1"
MAX_CHARS = 1200          # chunk text sent to the LLM
MIN_CHARS = 200           # skip shorter chunks
MAX_CONCEPTS = 8
MAX_RELATIONS = 8
GLOBAL_FREQ = 0.40        # concept in >40% of processed chunks = global topic
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


class Concept(BaseModel):
    name: str


class Relation(BaseModel):
    source: str
    relation: Literal["type_of", "part_of", "example_of", "uses", "related_to"]
    target: str


class Extraction(BaseModel):
    concepts: List[Concept] = Field(default_factory=list, max_length=MAX_CONCEPTS)
    relations: List[Relation] = Field(default_factory=list, max_length=MAX_RELATIONS)
