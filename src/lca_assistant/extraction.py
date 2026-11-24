"""Deterministic vocabulary-based extraction fallback for demos without an LLM."""

from __future__ import annotations

import re

from .models import DocumentChunk, ExtractedEntity, KnowledgeExtraction

VOCABULARY = {
    "cradle-to-gate": ("StudyPhase", "A study boundary from raw material extraction to the factory gate."),
    "cradle-to-grave": ("StudyPhase", "A study boundary covering a product life cycle through end of life."),
    "climate change": ("ImpactCategory", "An environmental impact category concerning climate change."),
    "water use": ("ImpactCategory", "An environmental impact category concerning water use."),
    "openlca": ("Software", "LCA software name mentioned in the source; capabilities require verification."),
    "agribalyse": ("Database", "LCA database name mentioned in the source; scope and version require verification."),
    "ecoinvent": ("Database", "LCA database name mentioned in the source; scope and version require verification."),
    "environmental footprint": ("ImpactAssessmentMethod", "Impact assessment method name mentioned in the source; version requires verification."),
    "recipe 2016": ("ImpactAssessmentMethod", "Impact assessment method name mentioned in the source; version requires verification."),
}


def extract_from_chunks(chunks: list[DocumentChunk]) -> KnowledgeExtraction:
    """Extract known terms as candidates; this is not a substitute for expert validation."""
    entities_by_key: dict[tuple[str, str], ExtractedEntity] = {}
    for chunk in chunks:
        lowered = chunk.text.lower()
        for term, (entity_type, description) in VOCABULARY.items():
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered):
                key = (entity_type, term.casefold())
                entities_by_key[key] = ExtractedEntity(
                    name=term,
                    entity_type=entity_type,
                    description=description,
                    source=chunk.source + (f" (page {chunk.page})" if chunk.page else ""),
                    confidence=0.55,
                )
    return KnowledgeExtraction(entities=list(entities_by_key.values()), relations=[])
