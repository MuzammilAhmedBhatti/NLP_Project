from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class Outcome(str, Enum):
    UPHELD = "upheld"
    OVERTURNED = "overturned"
    OTHER = "other"


class Decision(BaseModel):
    case_id: str
    title: str
    url: HttpUrl
    published_date: date | None = None
    outcome: Outcome = Outcome.OTHER
    decision_type: str = "Unknown"
    platform: list[str] = Field(default_factory=list)
    policies: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    summary: str = ""
    meta_action: str = "unknown"
    full_text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    majority: str = ""
    minority: str = ""
    scraped_at: str = ""


class Chunk(BaseModel):
    chunk_id: str
    case_id: str
    source_type: Literal["policy", "precedent"]
    title: str
    text: str
    url: str
    section: str = ""
    outcome: Outcome = Outcome.OTHER
    published_date: date | None = None
    policies: list[str] = Field(default_factory=list)


class PredictionRequest(BaseModel):
    post_description: str = Field(min_length=15)
    meta_action: Literal["removed", "left_up", "warning", "account_action"]
    user_guess: Literal["upheld", "overturned"] | None = None
    as_of_date: date | None = None


class Citation(BaseModel):
    chunk_id: str
    case_id: str
    title: str
    url: str
    excerpt: str
    score: float
    source_type: str


class Prediction(BaseModel):
    system: Literal["S0", "S1", "S2"]
    outcome: Literal["upheld", "overturned"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    policies: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    model: str
    fallback_used: bool = False
