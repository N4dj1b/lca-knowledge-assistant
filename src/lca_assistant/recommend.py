"""Explainable hybrid resource matching using text similarity and explicit criteria."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import RecommendationRequest, Resource


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _contains(candidate_values: list[str], requested: str) -> bool:
    if not requested.strip():
        return True
    target = _normalise(requested)
    values = [_normalise(value) for value in candidate_values]
    return any(target == value or target in value or value in target for value in values if value)


def _similarities(query: str, texts: list[str], backend: str = "tfidf") -> tuple[np.ndarray, str]:
    if backend in {"sentence-transformers", "embeddings", "dense"}:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            vectors = model.encode([query, *texts], normalize_embeddings=True)
            return np.asarray(vectors[1:] @ vectors[0]), "sentence-transformers"
        except Exception:
            pass
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([query, *texts])
    return cosine_similarity(matrix[0:1], matrix[1:]).ravel(), "tfidf"


@dataclass
class Recommendation:
    resource: Resource
    total_score: float
    semantic_score: float
    phase_match: float
    region_match: float
    category_match: float
    explanations: list[str]

    def to_dict(self):
        result = asdict(self)
        result["resource"] = self.resource.model_dump()
        return result


def recommend(
    request: RecommendationRequest,
    resources: list[Resource],
    embedding_backend: str = "tfidf",
) -> tuple[list[Recommendation], str]:
    candidates = [
        resource for resource in resources
        if request.resource_type == "Any" or resource.resource_type == request.resource_type
    ]
    if not candidates:
        return [], "none"

    query_texts = [
        " ".join([
            resource.name,
            resource.resource_type,
            resource.description,
            " ".join(resource.tags),
            " ".join(resource.regions),
            " ".join(resource.study_phases),
            " ".join(resource.impact_categories),
        ])
        for resource in candidates
    ]
    semantic_scores, backend_used = _similarities(request.requirements, query_texts, embedding_backend)
    weights = request.weights
    weight_sum = sum(weights.values())

    ranked: list[Recommendation] = []
    for resource, semantic in zip(candidates, semantic_scores):
        phase_match = 1.0 if _contains(resource.study_phases, request.study_phase) else 0.0
        is_global = any(_normalise(region) == "global" for region in resource.regions)
        region_match = 1.0 if (_contains(resource.regions, request.region) or is_global) else 0.0
        category_match = 1.0 if _contains(resource.impact_categories, request.impact_category) else 0.0

        # User-selected phase/region/category are treated as hard constraints.
        if request.study_phase and phase_match == 0:
            continue
        if request.region and region_match == 0:
            continue
        if request.impact_category and category_match == 0:
            continue

        active_criteria = {"semantic": weights["semantic"]}
        if request.study_phase:
            active_criteria["phase"] = weights["phase"]
        if request.region:
            active_criteria["region"] = weights["region"]
        if request.impact_category:
            active_criteria["category"] = weights["category"]
        weight_sum = sum(active_criteria.values()) or 1.0
        total = (
            active_criteria["semantic"] * max(0.0, float(semantic))
            + active_criteria.get("phase", 0.0) * phase_match
            + active_criteria.get("region", 0.0) * region_match
            + active_criteria.get("category", 0.0) * category_match
        ) / weight_sum
        explanations = [f"Text relevance score: {max(0.0, float(semantic)):.3f} ({backend_used})."]
        if request.study_phase:
            explanations.append(f"Meets the requested study phase: {request.study_phase}.")
        if request.region:
            explanations.append(f"Region constraint satisfied: {request.region}.")
        if request.impact_category:
            explanations.append(f"Impact-category constraint satisfied: {request.impact_category}.")
        explanations.append("Ranking is a prototype score, not a verified technical suitability assessment.")
        ranked.append(Recommendation(
            resource=resource,
            total_score=float(total),
            semantic_score=max(0.0, float(semantic)),
            phase_match=phase_match,
            region_match=region_match,
            category_match=category_match,
            explanations=explanations,
        ))

    ranked.sort(key=lambda item: item.total_score, reverse=True)
    return ranked, backend_used
