"""
qa/schema.py
Pydantic models shared across QA stages.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Understanding(BaseModel):
    intent: Literal[
        "definition", "fact", "list", "comparison", "procedure",
        "explanation", "numeric", "yes_no", "summary", "other"
    ]
    answer_type: Literal["span", "list", "paragraph", "number", "boolean"]
    focus: str                                      # <= 6 words
    key_terms: list[str] = Field(max_length=6)
    queries: list[str] = Field(max_length=3)
    hypothetical_answer: str                        # <= 40 words
    sub_questions: list[str] = Field(max_length=3)
    doc_hint: Optional[str] = None


class Claim(BaseModel):
    text: str = Field(max_length=400)
    support: list[str] = Field(min_length=1, max_length=3)   # sentence ids like "2.3"


class Answer(BaseModel):
    found: bool
    claims: list[Claim] = Field(max_length=5)


class Statement(BaseModel):
    text: str = Field(max_length=300)
    cites: list[str] = Field(min_length=1, max_length=4)    # claim ids "c1".."c5"


class Consolidated(BaseModel):
    lead: Statement
    points: list[Statement] = Field(max_length=4)
