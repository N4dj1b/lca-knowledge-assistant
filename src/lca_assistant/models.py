"""Validated data structures shared by ingestion, extraction, and recommendation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

EntityType = Literal[
    "Software",
    "Database",
    "ImpactAssessmentMethod",
    "ImpactCategory",
    "StudyPhase",
    "StudyRequirement",
    "Organization",
    "Other",
]


class DocumentChunk(BaseModel):
    text: str = Field(min_length=1)
    source: str
    page: int | None = None
    chunk_id: str


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    entity_type: EntityType
    description: str = Field(default="", max_length=1000)
    source: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    subject: str = Field(min_length=1, max_length=160)
    predicate: Literal[
        "mentions",
        "usesDatabase",
        "usesMethod",
        "coversCategory",
        "hasStudyPhase",
        "describedIn",
        "relatedTo",
    ]
    object: str = Field(min_length=1, max_length=160)
    source: str = ""


class KnowledgeExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    @field_validator("entities", "relations")
    @classmethod
    def limit_items(cls, value):
        if len(value) > 100:
            raise ValueError("At most 100 items are accepted per extraction.")
        return value


class Resource(BaseModel):
    id: str
    name: str
    resource_type: Literal["Software", "Database", "ImpactAssessmentMethod"]
    description: str
    tags: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    study_phases: list[str] = Field(default_factory=list)
    impact_categories: list[str] = Field(default_factory=list)
    source_note: str = "Illustrative demo entry; verify before real-world use."


class RecommendationRequest(BaseModel):
    requirements: str = Field(min_length=3)
    resource_type: Literal["Any", "Software", "Database", "ImpactAssessmentMethod"] = "Any"
    region: str = ""
    study_phase: str = ""
    impact_category: str = ""
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic": 0.55,
            "phase": 0.15,
            "region": 0.15,
            "category": 0.15,
        }
    )

    @field_validator("weights")
    @classmethod
    def valid_weights(cls, value):
        allowed = {"semantic", "phase", "region", "category"}
        if set(value) != allowed:
            raise ValueError(f"Weights must contain exactly: {sorted(allowed)}")
        if any(weight < 0 for weight in value.values()):
            raise ValueError("Weights cannot be negative.")
        if sum(value.values()) <= 0:
            raise ValueError("At least one weight must be positive.")
        return value
